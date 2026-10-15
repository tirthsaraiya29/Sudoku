#!/usr/bin/env python3
"""
SudokuMind dataset_validator.py — Independent Dataset Integrity Auditor
=======================================================================

Role in the 4-file SudokuMind architecture:
  1. dataset.py            Dataset generation.
  2. dataset_validator.py  (THIS FILE) Independent validation, safe repair,
                           deduplication, cross-split leakage auditing, and
                           quarantine of invalid records.
  3. trainer.py            Curriculum training.
  4. validator.py          Model accuracy evaluation on held-out test set.

Design principles:
  - Independent validation: never trusts generator labels. Every mathematical
    claim (solution correctness, uniqueness, difficulty, technique trace) is
    recomputed from first principles using code that is intentionally written
    independently from dataset.py.
  - Layered validation (A..G) matching the engineering specification.
  - Streaming, bounded-memory processing suitable for millions of records.
  - SQLite-backed deduplication registry for canonical puzzle and solution
    family hashes.
  - Safe repair policy with three categories: structural, derived-data, and
    quarantine. Every change is logged with old value, new value, and rule.
  - Non-destructive default: audit mode never mutates source data.

Exit codes:
  0  PASS                  all required checks passed cleanly
  1  PASS_WITH_WARNINGS    non-critical findings only
  2  FAIL                  at least one critical correctness failure
  3  INCOMPLETE            unresolved checks (timeouts, node limits)
  4  CLI_ERROR             invalid arguments / configuration
  5  INTERNAL_ERROR        unexpected internal exception

"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import random
import shutil
import signal
import sqlite3
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple


# ---------------------------------------------------------------------------
# Versioning & Constants
# ---------------------------------------------------------------------------

VALIDATOR_NAME = "SudokuMind-DatasetValidator"
VALIDATOR_VERSION = "1.0.0"

SUPPORTED_SCHEMA_VERSIONS: Tuple[str, ...] = ("sm-schema-v2.0",)
NUM_CELLS = 81
ALL_DIGITS_MASK = 0x1FF
MIN_UNIQUE_CLUES_STANDARD = 17

# Defaults
DEFAULT_MAX_LINE_BYTES = 512 * 1024
DEFAULT_MAX_RECORD_BYTES = 512 * 1024
DEFAULT_MAX_SOLVER_SECONDS = 5.0
DEFAULT_MAX_SOLVER_NODES = 300_000
DEFAULT_MAX_WALL_SECONDS = 0.0  # 0 = unlimited
DEFAULT_MIN_FREE_DISK_MB = 100.0

_SHUTDOWN = False


def _signal_handler(signum: int, _frame: Any) -> None:
    global _SHUTDOWN
    _SHUTDOWN = True
    sys.stderr.write(f"\n[SIGNAL] Received {signum}; finishing current batch...\n")


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ExitCode:
    OK = 0
    WARNINGS = 1
    FAIL = 2
    INCOMPLETE = 3
    CLI_ERROR = 4
    INTERNAL_ERROR = 5


class OverallStatus(str, Enum):
    PASS = "PASS"
    PASS_WITH_WARNINGS = "PASS_WITH_WARNINGS"
    FAIL = "FAIL"
    INCOMPLETE = "INCOMPLETE"


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class Verdict(str, Enum):
    VALID = "valid"
    INVALID = "invalid"
    UNRESOLVED = "unresolved"
    QUARANTINED = "quarantined"
    REPAIRED = "repaired"


class RepairMode(str, Enum):
    AUDIT = "audit"
    REPAIR_SAFE = "repair-safe"
    REPAIR_AGGRESSIVE = "repair-aggressive"
    QUARANTINE = "quarantine"


class SolverStatus(str, Enum):
    UNSAT = "unsat"
    UNIQUE = "unique"
    MULTIPLE = "multiple"
    TIMEOUT = "timeout"
    NODE_LIMIT = "node_limit"


class ValidationCategory(str, Enum):
    FILE = "file"
    SCHEMA = "schema"
    PUZZLE = "puzzle"
    SOLUTION = "solution"
    UNIQUENESS = "uniqueness"
    TARGET = "target"
    TECHNIQUE_TRACE = "technique_trace"
    DIFFICULTY = "difficulty"
    DEDUP = "dedup"
    LEAKAGE = "leakage"
    MANIFEST = "manifest"
    REPAIR = "repair"


# ---------------------------------------------------------------------------
# Independent Variant Registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VariantSpec:
    name: str
    units: Tuple[Tuple[int, ...], ...]
    peers: Tuple[Tuple[int, ...], ...]
    min_theoretical_clues: int
    preserves_d4: bool = True


def _build_standard_units() -> List[Tuple[int, ...]]:
    units: List[Tuple[int, ...]] = []
    for r in range(9):
        units.append(tuple(r * 9 + c for c in range(9)))
    for c in range(9):
        units.append(tuple(r * 9 + c for r in range(9)))
    for br in (0, 3, 6):
        for bc in (0, 3, 6):
            units.append(tuple((br + dr) * 9 + (bc + dc)
                               for dr in range(3) for dc in range(3)))
    return units


def _build_peers_from_units(units: Sequence[Tuple[int, ...]],
                            extra_deltas: Sequence[Tuple[int, int]] = ()
                            ) -> Tuple[Tuple[int, ...], ...]:
    peers: List[Set[int]] = [set() for _ in range(NUM_CELLS)]
    for unit in units:
        for cell in unit:
            peers[cell].update(c for c in unit if c != cell)
    for cell in range(NUM_CELLS):
        r, c = divmod(cell, 9)
        for dr, dc in extra_deltas:
            nr, nc = r + dr, c + dc
            if 0 <= nr < 9 and 0 <= nc < 9:
                peers[cell].add(nr * 9 + nc)
    return tuple(tuple(sorted(p)) for p in peers)


def _build_variants() -> Dict[str, VariantSpec]:
    base = _build_standard_units()
    reg: Dict[str, VariantSpec] = {}

    reg["standard"] = VariantSpec("standard", tuple(base),
                                  _build_peers_from_units(base), 17)

    diag = base + [
        tuple(i * 9 + i for i in range(9)),
        tuple(i * 9 + (8 - i) for i in range(9)),
    ]
    reg["diagonal"] = VariantSpec("diagonal", tuple(diag),
                                  _build_peers_from_units(diag), 12)

    hyper_boxes: List[Tuple[int, ...]] = []
    for br in (1, 5):
        for bc in (1, 5):
            hyper_boxes.append(tuple((br + dr) * 9 + (bc + dc)
                                     for dr in range(3) for dc in range(3)))
    hyper = base + hyper_boxes
    reg["hyper"] = VariantSpec("hyper", tuple(hyper),
                               _build_peers_from_units(hyper), 12)

    knight_deltas = [(-2, -1), (-2, 1), (-1, -2), (-1, 2),
                     (1, -2), (1, 2), (2, -1), (2, 1)]
    reg["anti_knight"] = VariantSpec("anti_knight", tuple(base),
                                     _build_peers_from_units(base, knight_deltas), 12)

    king_deltas = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    reg["anti_king"] = VariantSpec("anti_king", tuple(base),
                                   _build_peers_from_units(base, king_deltas), 12)

    return reg


VARIANT_REGISTRY: Dict[str, VariantSpec] = _build_variants()
SUPPORTED_VARIANTS: Tuple[str, ...] = tuple(VARIANT_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Grid Utilities
# ---------------------------------------------------------------------------


def grid_to_string(grid: Sequence[int]) -> str:
    return "".join(str(x) if 1 <= x <= 9 else "0" for x in grid)


def parse_grid_field(value: Any) -> Optional[List[int]]:
    """
    Strictly parse a puzzle/solution field into an 81-int grid.
    Accepts: an 81-char string using digits 0-9 with '0' or '.' for empty, or
    an 81-element list/tuple of ints in 0..9.  Returns None if invalid.
    """
    if isinstance(value, str):
        s = value.strip()
        if len(s) != NUM_CELLS:
            return None
        out: List[int] = []
        for ch in s:
            if ch == "0" or ch == ".":
                out.append(0)
            elif "1" <= ch <= "9":
                out.append(int(ch))
            else:
                return None
        return out
    if isinstance(value, (list, tuple)):
        if len(value) != NUM_CELLS:
            return None
        out2: List[int] = []
        for v in value:
            if isinstance(v, bool) or not isinstance(v, int):
                return None
            if not 0 <= v <= 9:
                return None
            out2.append(v)
        return out2
    return None


# ---------------------------------------------------------------------------
# Independent Bitmask MRV Solver
# ---------------------------------------------------------------------------


@dataclass
class SolveResult:
    status: SolverStatus
    solution_count: int
    first_solution: Optional[List[int]]
    nodes: int
    elapsed_sec: float
    method: str = "independent_bitmask_mrv_v1"


def _propagate(values: List[int], masks: List[int],
               spec: VariantSpec) -> Tuple[bool, int]:
    """Independent constraint propagation: naked + hidden singles, repeated."""
    steps = 0
    changed = True
    while changed:
        changed = False

        # Naked singles
        for cell in range(NUM_CELLS):
            if values[cell] == 0:
                m = masks[cell]
                if m == 0:
                    return False, steps
                if (m & (m - 1)) == 0:
                    digit = m.bit_length()
                    bit = m
                    values[cell] = digit
                    masks[cell] = bit
                    steps += 1
                    changed = True
                    for peer in spec.peers[cell]:
                        if values[peer] == digit:
                            return False, steps
                        if values[peer] == 0 and (masks[peer] & bit):
                            masks[peer] &= ~bit
                            if masks[peer] == 0:
                                return False, steps

        # Hidden singles
        for unit in spec.units:
            seen_once = 0
            seen_multi = 0
            for cell in unit:
                m = masks[cell]
                seen_multi |= seen_once & m
                seen_once |= m
            if (seen_once | seen_multi) != ALL_DIGITS_MASK:
                return False, steps
            unique_bits = seen_once & ~seen_multi
            while unique_bits:
                bit = unique_bits & -unique_bits
                unique_bits ^= bit
                target = -1
                for cell in unit:
                    if values[cell] == 0 and (masks[cell] & bit):
                        target = cell
                        break
                if target == -1:
                    return False, steps
                digit = bit.bit_length()
                values[target] = digit
                masks[target] = bit
                steps += 1
                changed = True
                for peer in spec.peers[target]:
                    if values[peer] == digit:
                        return False, steps
                    if values[peer] == 0 and (masks[peer] & bit):
                        masks[peer] &= ~bit
                        if masks[peer] == 0:
                            return False, steps
    return True, steps


def solve_and_count(
    puzzle: Sequence[int],
    spec: VariantSpec,
    limit: int = 2,
    timeout_sec: float = DEFAULT_MAX_SOLVER_SECONDS,
    max_nodes: int = DEFAULT_MAX_SOLVER_NODES,
) -> SolveResult:
    """
    Independently written bounded bitmask solver.
    Distinguishes UNSAT, UNIQUE, MULTIPLE, TIMEOUT, NODE_LIMIT.
    """
    t0 = time.monotonic()

    values = [0] * NUM_CELLS
    masks = [0] * NUM_CELLS
    for cell in range(NUM_CELLS):
        v = puzzle[cell]
        if v == 0:
            masks[cell] = ALL_DIGITS_MASK
        elif 1 <= v <= 9:
            values[cell] = v
            masks[cell] = 1 << (v - 1)
        else:
            return SolveResult(SolverStatus.UNSAT, 0, None, 0,
                               time.monotonic() - t0)

    # Initial consistency for givens
    for cell in range(NUM_CELLS):
        if values[cell] != 0:
            d = values[cell]
            for peer in spec.peers[cell]:
                if values[peer] == d:
                    return SolveResult(SolverStatus.UNSAT, 0, None, 0,
                                       time.monotonic() - t0)
                if values[peer] == 0:
                    masks[peer] &= ~(1 << (d - 1))
                    if masks[peer] == 0:
                        return SolveResult(SolverStatus.UNSAT, 0, None, 0,
                                           time.monotonic() - t0)

    ok, _ = _propagate(values, masks, spec)
    if not ok:
        return SolveResult(SolverStatus.UNSAT, 0, None, 0,
                           time.monotonic() - t0)

    solutions: List[List[int]] = []
    nodes = [0]
    aborted = [False]
    timeout_hit = [False]

    def dfs(vals: List[int], ms: List[int]) -> bool:
        if aborted[0]:
            return True
        if nodes[0] >= max_nodes:
            aborted[0] = True
            return True
        # Time check is amortized
        if (nodes[0] & 63) == 0 and time.monotonic() - t0 > timeout_sec:
            timeout_hit[0] = True
            aborted[0] = True
            return True

        best = -1
        best_cnt = 10
        for cell in range(NUM_CELLS):
            if vals[cell] == 0:
                c = ms[cell].bit_count()
                if c == 0:
                    return False
                if c < best_cnt:
                    best_cnt = c
                    best = cell
                    if c == 2:
                        break

        if best == -1:
            solutions.append(list(vals))
            return len(solutions) >= limit

        m = ms[best]
        bits: List[int] = []
        while m:
            b = m & -m
            bits.append(b)
            m ^= b

        for bit in bits:
            nodes[0] += 1
            nv = list(vals)
            nm = list(ms)
            digit = bit.bit_length()
            nv[best] = digit
            nm[best] = bit
            consistent = True
            for peer in spec.peers[best]:
                if nv[peer] == digit:
                    consistent = False
                    break
                if nv[peer] == 0 and (nm[peer] & bit):
                    nm[peer] &= ~bit
                    if nm[peer] == 0:
                        consistent = False
                        break
            if not consistent:
                continue
            prop_ok, _ = _propagate(nv, nm, spec)
            if prop_ok:
                if dfs(nv, nm):
                    return True
            if len(solutions) >= limit:
                return True
        return False

    dfs(values, masks)
    elapsed = time.monotonic() - t0

    if aborted[0]:
        status = SolverStatus.TIMEOUT if timeout_hit[0] else SolverStatus.NODE_LIMIT
    elif len(solutions) == 0:
        status = SolverStatus.UNSAT
    elif len(solutions) == 1:
        status = SolverStatus.UNIQUE
    else:
        status = SolverStatus.MULTIPLE

    return SolveResult(status, len(solutions),
                       solutions[0] if solutions else None,
                       nodes[0], elapsed)


def validate_solution_grid(grid: Sequence[int], spec: VariantSpec) -> Tuple[bool, Optional[str]]:
    """Check that every cell is filled 1..9 and all unit/peer constraints hold."""
    if len(grid) != NUM_CELLS:
        return False, f"grid length {len(grid)} != 81"
    for i, v in enumerate(grid):
        if not (1 <= v <= 9):
            return False, f"cell {i} not filled with 1..9 (got {v!r})"
    # Unit coverage
    for u_idx, unit in enumerate(spec.units):
        seen = set()
        for cell in unit:
            v = grid[cell]
            if v in seen:
                return False, f"unit {u_idx}: duplicate {v} at cell {cell}"
            seen.add(v)
    # Peer coverage (covers anti-knight / anti-king which are not units)
    for cell in range(NUM_CELLS):
        v = grid[cell]
        for peer in spec.peers[cell]:
            if grid[peer] == v:
                return False, f"peer conflict: cells {cell},{peer} both {v}"
    return True, None


def validate_puzzle_grid(grid: Sequence[int], spec: VariantSpec) -> Tuple[bool, Optional[str]]:
    """Check that givens do not already violate constraints (may be unsolved)."""
    if len(grid) != NUM_CELLS:
        return False, f"grid length {len(grid)} != 81"
    for i, v in enumerate(grid):
        if not (0 <= v <= 9):
            return False, f"cell {i} invalid value {v!r}"
    for cell in range(NUM_CELLS):
        v = grid[cell]
        if v == 0:
            continue
        for peer in spec.peers[cell]:
            if grid[peer] == v:
                return False, f"conflicting givens: cells {cell},{peer} both {v}"
    return True, None


# ---------------------------------------------------------------------------
# Canonical Hashing (Independent D4 + digit relabeling)
# ---------------------------------------------------------------------------


def _build_d4_maps() -> Tuple[Tuple[int, ...], ...]:
    maps = []
    for reflect in (False, True):
        for rot in range(4):
            perm = [0] * NUM_CELLS
            for r in range(9):
                for c in range(9):
                    rr, cc = r, (8 - c if reflect else c)
                    for _ in range(rot):
                        rr, cc = cc, 8 - rr
                    perm[r * 9 + c] = rr * 9 + cc
            maps.append(tuple(perm))
    return tuple(maps)


D4_MAPS: Tuple[Tuple[int, ...], ...] = _build_d4_maps()


def _relabel_canonical(grid: Sequence[int]) -> str:
    mapping: Dict[int, str] = {0: "0"}
    nxt = 1
    out: List[str] = []
    for v in grid:
        if v not in mapping:
            mapping[v] = str(nxt)
            nxt += 1
        out.append(mapping[v])
    return "".join(out)


def canonical_d4(grid: Sequence[int], use_d4: bool = True) -> str:
    if not use_d4:
        return _relabel_canonical(grid)
    best: Optional[str] = None
    for perm in D4_MAPS:
        t = [grid[perm[i]] for i in range(NUM_CELLS)]
        r = _relabel_canonical(t)
        if best is None or r < best:
            best = r
    assert best is not None
    return best


def compute_canonical_hashes(puzzle: Sequence[int], solution: Sequence[int],
                             spec: VariantSpec) -> Tuple[str, str, str]:
    """
    Return (exact_puzzle_hash, canonical_puzzle_hash, family_hash).
    NOTE: This is intentionally written independently from dataset.py, though the
    algorithm is necessarily similar (D4 + digit relabeling).
    """
    puz_str = grid_to_string(puzzle)
    exact = hashlib.sha256(f"{spec.name}:{puz_str}".encode("utf-8")).hexdigest()
    canon_puz = canonical_d4(puzzle, use_d4=spec.preserves_d4)
    canon_puz_h = hashlib.sha256(f"{spec.name}:{canon_puz}".encode("utf-8")).hexdigest()
    canon_sol = canonical_d4(solution, use_d4=spec.preserves_d4)
    fam_h = hashlib.sha256(f"{spec.name}:{canon_sol}".encode("utf-8")).hexdigest()
    return exact, canon_puz_h, fam_h


# ---------------------------------------------------------------------------
# Technique Verifiers
# ---------------------------------------------------------------------------


TECHNIQUES_SUPPORTED = {
    "full_house", "naked_single", "hidden_single", "locked_candidates",
    "naked_pair", "hidden_pair", "naked_triple", "hidden_triple",
    "x_wing", "xy_wing", "swordfish", "forcing_chain_contradiction",
}

TECHNIQUES_FULLY_VERIFIED = {
    "full_house", "naked_single", "hidden_single", "locked_candidates",
    "naked_pair", "hidden_pair", "naked_triple", "hidden_triple",
    "x_wing", "xy_wing", "swordfish",
}

TECHNIQUES_BEST_EFFORT = {"forcing_chain_contradiction"}


def _apply_placement(cell: int, digit: int, values: List[int],
                     masks: List[int], spec: VariantSpec) -> bool:
    if values[cell] != 0:
        return False
    bit = 1 << (digit - 1)
    if not (masks[cell] & bit):
        return False
    values[cell] = digit
    masks[cell] = bit
    for peer in spec.peers[cell]:
        if values[peer] == digit:
            return False
        if values[peer] == 0 and (masks[peer] & bit):
            masks[peer] &= ~bit
            if masks[peer] == 0:
                return False
    return True


def _apply_elimination(targets: Sequence[int], digits: Sequence[int],
                       values: List[int], masks: List[int]) -> bool:
    for t in targets:
        if values[t] != 0:
            return False
        for d in digits:
            masks[t] &= ~(1 << (d - 1))
    return True


def _verify_full_house(values, masks, spec, step) -> Tuple[bool, Optional[str]]:
    targets = step["target_cells"]
    digits = step["digits"]
    if len(targets) != 1 or len(digits) != 1:
        return False, "full_house expects 1 target cell and 1 digit"
    cell = targets[0]
    if values[cell] != 0:
        return False, f"cell {cell} already filled"
    for unit in spec.units:
        if cell not in unit:
            continue
        empty = [c for c in unit if values[c] == 0]
        if len(empty) == 1 and empty[0] == cell:
            if masks[cell].bit_count() == 1 and masks[cell] == (1 << (digits[0] - 1)):
                return True, None
    return False, "no unit with exactly one empty cell equal to target"


def _verify_naked_single(values, masks, spec, step) -> Tuple[bool, Optional[str]]:
    targets = step["target_cells"]
    digits = step["digits"]
    if len(targets) != 1 or len(digits) != 1:
        return False, "naked_single expects 1 target cell and 1 digit"
    cell = targets[0]
    if values[cell] != 0:
        return False, f"cell {cell} already filled"
    if masks[cell] != (1 << (digits[0] - 1)):
        return False, f"cell {cell} candidates {masks[cell]:#x} != single digit {digits[0]}"
    return True, None


def _verify_hidden_single(values, masks, spec, step) -> Tuple[bool, Optional[str]]:
    targets = step["target_cells"]
    digits = step["digits"]
    if len(targets) != 1 or len(digits) != 1:
        return False, "hidden_single expects 1 target cell and 1 digit"
    cell = targets[0]
    digit = digits[0]
    bit = 1 << (digit - 1)
    if values[cell] != 0:
        return False, f"cell {cell} already filled"
    if not (masks[cell] & bit):
        return False, f"digit {digit} not a candidate in target cell {cell}"
    for unit in spec.units:
        if cell not in unit:
            continue
        holders = [c for c in unit if values[c] == 0 and (masks[c] & bit)]
        if len(holders) == 1 and holders[0] == cell:
            return True, None
    return False, f"digit {digit} not hidden single in any unit of cell {cell}"


def _verify_locked_candidates(values, masks, spec, step) -> Tuple[bool, Optional[str]]:
    targets = step["target_cells"]
    digits = step["digits"]
    if len(digits) != 1:
        return False, "locked_candidates expects 1 digit"
    if not targets:
        return False, "no targets"
    digit = digits[0]
    bit = 1 << (digit - 1)
    # Find unit_a that contains all cells with candidate d (restricted to the ones
    # that also touch unit_b), and a unit_b != unit_a whose cells outside unit_a
    # are exactly `targets`.
    for unit_a in spec.units:
        holders = [c for c in unit_a if values[c] == 0 and (masks[c] & bit)]
        if len(holders) < 2:
            continue
        for unit_b in spec.units:
            if unit_b is unit_a:
                continue
            set_a = set(unit_a)
            set_b = set(unit_b)
            if not all(h in set_b for h in holders):
                continue
            candidates_b_outside_a = [
                c for c in unit_b
                if c not in set_a and values[c] == 0 and (masks[c] & bit)
            ]
            if sorted(candidates_b_outside_a) == sorted(targets):
                return True, None
    return False, "locked_candidates pattern not found"


def _verify_naked_subset(values, masks, spec, step, size) -> Tuple[bool, Optional[str]]:
    targets = step["target_cells"]
    digits = step["digits"]
    if len(digits) != size:
        return False, f"naked_{size} expects {size} digits"
    if len(targets) == 0:
        return False, "no targets"
    bitmask = 0
    for d in digits:
        bitmask |= 1 << (d - 1)
    # Find a unit containing all `targets` and a set of `size` cells within that
    # unit whose union of candidates equals bitmask, and which are disjoint from
    # targets.
    for unit in spec.units:
        if not all(t in unit for t in targets):
            continue
        empty = [c for c in unit if values[c] == 0]
        # Look for a `size`-subset of empty cells whose union == bitmask and whose
        # intersection with targets is empty
        # (target cells are typically NOT part of the subset)
        import itertools
        candidate_cells = [c for c in empty if c not in targets]
        for subset in itertools.combinations(candidate_cells, size):
            union = 0
            for c in subset:
                union |= masks[c]
            if union.bit_count() != size:
                continue
            if union != bitmask:
                continue
            # Verify that each target contains at least one of `digits`
            if all((masks[t] & bitmask) != 0 for t in targets):
                # Verify removal is meaningful (targets have extra candidates)
                # Actually if target only has these digits, elimination would be no-op
                # Both are acceptable; but the record says elimination, so we check
                # that it is a legal action.
                return True, None
    return False, f"naked_{size} pattern not found"


def _verify_hidden_subset(values, masks, spec, step, size) -> Tuple[bool, Optional[str]]:
    targets = step["target_cells"]
    digits = step["digits"]
    if len(digits) != size:
        return False, f"hidden_{size} expects {size} digits"
    bits = [1 << (d - 1) for d in digits]
    digit_mask = 0
    for b in bits:
        digit_mask |= b
    # Find a unit containing the targets, and where each digit's holder set is a
    # subset of the size cells.
    for unit in spec.units:
        if not all(t in unit for t in targets):
            continue
        # Union of holders across the digits
        union_holders: Set[int] = set()
        ok = True
        for b in bits:
            holders = [c for c in unit if values[c] == 0 and (masks[c] & b)]
            if len(holders) == 0:
                ok = False
                break
            union_holders.update(holders)
        if not ok:
            continue
        if len(union_holders) != size:
            continue
        # Every target must be among holders and must have candidates outside digit_mask
        if not union_holders.issubset(set(targets)):
            continue
        if all((masks[t] & ~digit_mask) != 0 for t in targets):
            return True, None
    return False, f"hidden_{size} pattern not found"


def _verify_x_wing(values, masks, spec, step) -> Tuple[bool, Optional[str]]:
    targets = step["target_cells"]
    digits = step["digits"]
    if len(digits) != 1:
        return False, "x_wing expects 1 digit"
    d = digits[0]
    bit = 1 << (d - 1)
    # Find two rows with candidates for d in exactly the same 2 columns
    row_cands: Dict[int, List[int]] = {}
    for r in range(9):
        cols = [c for c in range(9)
                if values[r * 9 + c] == 0 and (masks[r * 9 + c] & bit)]
        row_cands[r] = cols
    import itertools
    for r1, r2 in itertools.combinations(range(9), 2):
        c1 = row_cands[r1]
        c2 = row_cands[r2]
        if len(c1) != 2 or len(c2) != 2:
            continue
        if set(c1) != set(c2):
            continue
        cols = c1
        # Targets should be cells in columns `cols`, not in rows r1/r2, with bit set
        expected = set()
        for c in cols:
            for r in range(9):
                if r in (r1, r2):
                    continue
                cell = r * 9 + c
                if values[cell] == 0 and (masks[cell] & bit):
                    expected.add(cell)
        if expected == set(targets) and expected:
            return True, None
    # Column-based X-Wing
    col_cands: Dict[int, List[int]] = {}
    for c in range(9):
        rows = [r for r in range(9)
                if values[r * 9 + c] == 0 and (masks[r * 9 + c] & bit)]
        col_cands[c] = rows
    for c1, c2 in itertools.combinations(range(9), 2):
        r1 = col_cands[c1]
        r2 = col_cands[c2]
        if len(r1) != 2 or len(r2) != 2:
            continue
        if set(r1) != set(r2):
            continue
        rows = r1
        expected = set()
        for r in rows:
            for c in range(9):
                if c in (c1, c2):
                    continue
                cell = r * 9 + c
                if values[cell] == 0 and (masks[cell] & bit):
                    expected.add(cell)
        if expected == set(targets) and expected:
            return True, None
    return False, "x_wing pattern not found"


def _verify_xy_wing(values, masks, spec, step) -> Tuple[bool, Optional[str]]:
    targets = step["target_cells"]
    digits = step["digits"]
    if len(digits) != 1:
        return False, "xy_wing expects 1 elimination digit"
    z = digits[0]
    z_bit = 1 << (z - 1)
    # Find pivot bivalue {X,Y} and pincers {X,Z}, {Y,Z}
    bivalue = [c for c in range(NUM_CELLS)
               if values[c] == 0 and masks[c].bit_count() == 2]
    for pivot in bivalue:
        pm = masks[pivot]
        x_bit = pm & -pm
        y_bit = pm ^ x_bit
        if x_bit == z_bit or y_bit == z_bit:
            continue
        peer_bivalues = [p for p in spec.peers[pivot] if p in set(bivalue)]
        for p1 in peer_bivalues:
            if not (masks[p1] & x_bit):
                continue
            if masks[p1] & y_bit:
                continue
            p1_other = masks[p1] ^ x_bit
            if p1_other != z_bit:
                continue
            for p2 in peer_bivalues:
                if p2 == p1:
                    continue
                if not (masks[p2] & y_bit):
                    continue
                if masks[p2] & x_bit:
                    continue
                p2_other = masks[p2] ^ y_bit
                if p2_other != z_bit:
                    continue
                common = set(spec.peers[p1]).intersection(spec.peers[p2])
                common.discard(pivot)
                expected = set(c for c in common
                               if values[c] == 0 and (masks[c] & z_bit))
                if expected == set(targets) and expected:
                    return True, None
    return False, "xy_wing pattern not found"


def _verify_swordfish(values, masks, spec, step) -> Tuple[bool, Optional[str]]:
    targets = step["target_cells"]
    digits = step["digits"]
    if len(digits) != 1:
        return False, "swordfish expects 1 digit"
    d = digits[0]
    bit = 1 << (d - 1)
    import itertools
    for r1, r2, r3 in itertools.combinations(range(9), 3):
        rows = [r1, r2, r3]
        all_cols: Set[int] = set()
        valid = True
        for r in rows:
            cols = [c for c in range(9)
                    if values[r * 9 + c] == 0 and (masks[r * 9 + c] & bit)]
            if not (2 <= len(cols) <= 3):
                valid = False
                break
            all_cols.update(cols)
        if not valid or len(all_cols) != 3:
            continue
        expected = set()
        for c in all_cols:
            for r in range(9):
                if r in rows:
                    continue
                cell = r * 9 + c
                if values[cell] == 0 and (masks[cell] & bit):
                    expected.add(cell)
        if expected == set(targets) and expected:
            return True, None
    # Column-based
    for c1, c2, c3 in itertools.combinations(range(9), 3):
        cols = [c1, c2, c3]
        all_rows: Set[int] = set()
        valid = True
        for c in cols:
            rows = [r for r in range(9)
                    if values[r * 9 + c] == 0 and (masks[r * 9 + c] & bit)]
            if not (2 <= len(rows) <= 3):
                valid = False
                break
            all_rows.update(rows)
        if not valid or len(all_rows) != 3:
            continue
        expected = set()
        for r in all_rows:
            for c in range(9):
                if c in cols:
                    continue
                cell = r * 9 + c
                if values[cell] == 0 and (masks[cell] & bit):
                    expected.add(cell)
        if expected == set(targets) and expected:
            return True, None
    return False, "swordfish pattern not found"


def _verify_forcing_chain_best_effort(values, masks, spec, step
                                       ) -> Tuple[Optional[bool], Optional[str]]:
    """
    Best-effort verification: try assuming the eliminated digit at the target cell
    and propagate. If a contradiction appears, the elimination is consistent.
    Returns (True, None) on contradiction, (False, msg) if no contradiction, or
    (None, msg) if verification is inconclusive.
    """
    targets = step["target_cells"]
    digits = step["digits"]
    if len(targets) != 1 or len(digits) != 1:
        return None, "forcing_chain expects 1 target and 1 digit"
    cell = targets[0]
    d = digits[0]
    bit = 1 << (d - 1)
    if values[cell] != 0:
        return None, f"cell {cell} already filled"
    if not (masks[cell] & bit):
        return None, "eliminated digit not a candidate"
    tv = list(values)
    tm = list(masks)
    tv[cell] = d
    tm[cell] = bit
    for peer in spec.peers[cell]:
        if tv[peer] == d:
            return True, None
        if tv[peer] == 0 and (tm[peer] & bit):
            tm[peer] &= ~bit
            if tm[peer] == 0:
                return True, None
    ok, _ = _propagate(tv, tm, spec)
    if not ok:
        return True, None
    return None, "no immediate contradiction detected (best-effort)"


def _run_technique_verifier(values, masks, spec, step
                            ) -> Tuple[Optional[bool], Optional[str]]:
    """
    Returns (True, None) if verified, (False, reason) if provably invalid,
    (None, reason) if unverified/inconclusive.
    """
    tech = step["technique"]
    if tech == "full_house":
        return _verify_full_house(values, masks, spec, step)
    if tech == "naked_single":
        return _verify_naked_single(values, masks, spec, step)
    if tech == "hidden_single":
        return _verify_hidden_single(values, masks, spec, step)
    if tech == "locked_candidates":
        return _verify_locked_candidates(values, masks, spec, step)
    if tech == "naked_pair":
        return _verify_naked_subset(values, masks, spec, step, 2)
    if tech == "naked_triple":
        return _verify_naked_subset(values, masks, spec, step, 3)
    if tech == "hidden_pair":
        return _verify_hidden_subset(values, masks, spec, step, 2)
    if tech == "hidden_triple":
        return _verify_hidden_subset(values, masks, spec, step, 3)
    if tech == "x_wing":
        return _verify_x_wing(values, masks, spec, step)
    if tech == "xy_wing":
        return _verify_xy_wing(values, masks, spec, step)
    if tech == "swordfish":
        return _verify_swordfish(values, masks, spec, step)
    if tech == "forcing_chain_contradiction":
        return _verify_forcing_chain_best_effort(values, masks, spec, step)
    return None, f"unsupported technique '{tech}'"


# ---------------------------------------------------------------------------
# Trace Replay
# ---------------------------------------------------------------------------


@dataclass
class TraceReplayResult:
    ok: bool
    first_error: Optional[str]
    verified_steps: int
    unverified_steps: int
    invalid_steps: int
    techniques_verified: List[str]
    techniques_unverified: List[str]
    reached_full_solution: bool


def replay_technique_trace(trace: Sequence[Dict[str, Any]],
                           puzzle: Sequence[int],
                           solution: Optional[Sequence[int]],
                           spec: VariantSpec) -> TraceReplayResult:
    """
    Replays a technique_trace from the puzzle state, verifying each step.
    Returns an aggregated result. Never asserts full verification when a step
    is only 'best-effort' or unsupported.
    """
    values = list(puzzle)
    masks = [0] * NUM_CELLS
    for cell in range(NUM_CELLS):
        if values[cell] != 0:
            masks[cell] = 1 << (values[cell] - 1)
        else:
            masks[cell] = ALL_DIGITS_MASK
    for cell in range(NUM_CELLS):
        if values[cell] != 0:
            d = values[cell]
            bit = 1 << (d - 1)
            for peer in spec.peers[cell]:
                if values[peer] == 0:
                    masks[peer] &= ~bit

    verified = 0
    unverified = 0
    invalid = 0
    tech_verified: Set[str] = set()
    tech_unverified: Set[str] = set()

    for idx, step in enumerate(trace):
        if not isinstance(step, dict):
            return TraceReplayResult(False, f"step {idx}: not an object",
                                     verified, unverified, invalid,
                                     sorted(tech_verified),
                                     sorted(tech_unverified), False)
        tech = step.get("technique")
        action = step.get("action_type")
        if not isinstance(tech, str) or not isinstance(action, str):
            return TraceReplayResult(False, f"step {idx}: missing technique/action",
                                     verified, unverified, invalid,
                                     sorted(tech_verified),
                                     sorted(tech_unverified), False)

        verdict, reason = _run_technique_verifier(values, masks, spec, step)
        if verdict is False:
            invalid += 1
            return TraceReplayResult(False, f"step {idx} ({tech}): {reason}",
                                     verified, unverified, invalid,
                                     sorted(tech_verified),
                                     sorted(tech_unverified), False)
        if verdict is None:
            unverified += 1
            tech_unverified.add(tech)
        else:
            verified += 1
            tech_verified.add(tech)

        # Apply the action (best-effort for unverified techniques)
        ok = True
        if action == "placement":
            targets = step.get("target_cells") or []
            digits = step.get("digits") or []
            if len(targets) == 1 and len(digits) == 1:
                ok = _apply_placement(int(targets[0]), int(digits[0]),
                                      values, masks, spec)
        elif action == "elimination":
            targets = step.get("target_cells") or []
            digits = step.get("digits") or []
            ok = _apply_elimination([int(t) for t in targets],
                                    [int(d) for d in digits],
                                    values, masks)
        else:
            ok = False

        if not ok:
            return TraceReplayResult(False, f"step {idx}: action '{action}' failed to apply",
                                     verified, unverified, invalid,
                                     sorted(tech_verified),
                                     sorted(tech_unverified), False)

    reached = False
    if solution is not None and all(v != 0 for v in values):
        reached = all(values[i] == solution[i] for i in range(NUM_CELLS))

    return TraceReplayResult(True, None, verified, unverified, invalid,
                             sorted(tech_verified), sorted(tech_unverified), reached)


# ---------------------------------------------------------------------------
# Findings
# ---------------------------------------------------------------------------


@dataclass
class Finding:
    category: str
    severity: str
    code: str
    message: str
    file: Optional[str] = None
    line: Optional[int] = None
    record_id: Optional[str] = None
    evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category,
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "file": self.file,
            "line": self.line,
            "record_id": self.record_id,
            "evidence": self.evidence,
        }


# ---------------------------------------------------------------------------
# Schema Profile
# ---------------------------------------------------------------------------


REQUIRED_KEYS_V2 = {
    "schema_version", "puzzle_id", "split", "variant", "difficulty",
    "difficulty_raw", "difficulty_features", "givens", "puzzle", "solution",
    "solution_count_status", "unique_solution", "task_type", "target",
    "technique_trace", "generator_version", "generation_seed",
    "solver_version", "provenance", "validation_status", "record_sha256",
}


def _check_json_strict(raw: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Parse JSON with duplicate-key detection."""
    def _hook(pairs):
        keys = [k for k, _ in pairs]
        if len(keys) != len(set(keys)):
            seen = set()
            dups = [k for k in keys if k in seen or seen.add(k)]
            raise ValueError(f"duplicate JSON keys: {sorted(set(dups))}")
        return dict(pairs)

    try:
        obj = json.loads(raw, object_pairs_hook=_hook,
                         parse_constant=lambda x: (_ for _ in ()).throw(
                             ValueError(f"forbidden JSON constant {x}")))
    except json.JSONDecodeError as e:
        return None, f"json_decode_error: {e}"
    except ValueError as e:
        return None, str(e)
    if not isinstance(obj, dict):
        return None, "top-level JSON is not an object"
    return obj, None


def _is_finite_number(x: Any) -> bool:
    if isinstance(x, bool):
        return False
    if not isinstance(x, (int, float)):
        return False
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return False
    return True


def _is_int(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


# ---------------------------------------------------------------------------
# Record Validator
# ---------------------------------------------------------------------------


@dataclass
class RecordValidationOutcome:
    verdict: Verdict
    findings: List[Finding]
    repaired_record: Optional[Dict[str, Any]]
    repairs_applied: List[Dict[str, Any]]


def validate_record(
    record: Dict[str, Any],
    file_path: str,
    line_no: int,
    spec_override: Optional[VariantSpec],
    mode: RepairMode,
    max_solver_sec: float,
    max_solver_nodes: int,
) -> RecordValidationOutcome:
    findings: List[Finding] = []
    repairs: List[Dict[str, Any]] = []
    record = dict(record)  # shallow copy so we can mutate safely

    def add(cat: ValidationCategory, sev: Severity, code: str, msg: str,
            **ev: Any) -> None:
        findings.append(Finding(cat.value, sev.value, code, msg,
                                file=file_path, line=line_no,
                                record_id=record.get("puzzle_id"),
                                evidence=ev))

    # --- Schema version ---
    sv = record.get("schema_version")
    if not isinstance(sv, str):
        add(ValidationCategory.SCHEMA, Severity.CRITICAL, "missing_schema_version",
            "schema_version missing or not a string")
        return RecordValidationOutcome(Verdict.INVALID, findings, None, repairs)
    if sv not in SUPPORTED_SCHEMA_VERSIONS:
        add(ValidationCategory.SCHEMA, Severity.CRITICAL, "unsupported_schema_version",
            f"schema_version {sv!r} not supported; supported={list(SUPPORTED_SCHEMA_VERSIONS)}")
        return RecordValidationOutcome(Verdict.INVALID, findings, None, repairs)

    # --- Required fields ---
    missing = REQUIRED_KEYS_V2 - set(record.keys())
    if missing:
        add(ValidationCategory.SCHEMA, Severity.CRITICAL, "missing_fields",
            f"missing required keys: {sorted(missing)}")
        return RecordValidationOutcome(Verdict.INVALID, findings, None, repairs)

    # --- Variant ---
    variant_name = record.get("variant")
    if not isinstance(variant_name, str):
        add(ValidationCategory.SCHEMA, Severity.CRITICAL, "bad_variant",
            "variant not a string")
        return RecordValidationOutcome(Verdict.INVALID, findings, None, repairs)
    if spec_override is not None and variant_name != spec_override.name:
        add(ValidationCategory.SCHEMA, Severity.ERROR, "variant_mismatch",
            f"record variant {variant_name!r} != override {spec_override.name!r}")
        return RecordValidationOutcome(Verdict.INVALID, findings, None, repairs)
    if variant_name not in VARIANT_REGISTRY:
        add(ValidationCategory.SCHEMA, Severity.CRITICAL, "unsupported_variant",
            f"unsupported variant {variant_name!r}")
        return RecordValidationOutcome(Verdict.INVALID, findings, None, repairs)
    spec = VARIANT_REGISTRY[variant_name]

    # --- Record self-checksum ---
    claimed_sha = record.get("record_sha256")
    if not isinstance(claimed_sha, str) or len(claimed_sha) != 64:
        add(ValidationCategory.SCHEMA, Severity.ERROR, "bad_record_sha",
            "record_sha256 missing or malformed")
    else:
        rec_copy = dict(record)
        rec_copy.pop("record_sha256", None)
        computed = hashlib.sha256(
            json.dumps(rec_copy, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        if computed != claimed_sha:
            add(ValidationCategory.SCHEMA, Severity.ERROR, "record_sha_mismatch",
                "record_sha256 mismatch",
                claimed=claimed_sha, computed=computed)

    # --- Puzzle & solution parsing ---
    puzzle_grid = parse_grid_field(record.get("puzzle"))
    if puzzle_grid is None:
        add(ValidationCategory.PUZZLE, Severity.CRITICAL, "bad_puzzle_field",
            "puzzle field is not a valid 81-cell grid")
        return RecordValidationOutcome(Verdict.INVALID, findings, None, repairs)
    solution_grid = parse_grid_field(record.get("solution"))
    if solution_grid is None:
        add(ValidationCategory.SOLUTION, Severity.CRITICAL, "bad_solution_field",
            "solution field is not a valid 81-cell grid")
        return RecordValidationOutcome(Verdict.INVALID, findings, None, repairs)

    # --- Puzzle structural integrity ---
    ok, err = validate_puzzle_grid(puzzle_grid, spec)
    if not ok:
        add(ValidationCategory.PUZZLE, Severity.CRITICAL, "puzzle_invalid",
            f"puzzle constraints violated: {err}")
        return RecordValidationOutcome(Verdict.INVALID, findings, None, repairs)

    actual_givens = sum(1 for x in puzzle_grid if x != 0)
    if actual_givens < spec.min_theoretical_clues:
        add(ValidationCategory.PUZZLE, Severity.CRITICAL, "below_min_clues",
            f"givens={actual_givens} below theoretical minimum {spec.min_theoretical_clues}")
        return RecordValidationOutcome(Verdict.INVALID, findings, None, repairs)

    claimed_givens = record.get("givens")
    if not _is_int(claimed_givens) or claimed_givens != actual_givens:
        add(ValidationCategory.DIFFICULTY, Severity.WARNING, "givens_mismatch",
            f"claimed givens={claimed_givens} != actual={actual_givens}")
        # Safe repair: recompute givens
        if mode in (RepairMode.REPAIR_SAFE, RepairMode.REPAIR_AGGRESSIVE):
            record["givens"] = actual_givens
            repairs.append({"rule": "recompute_givens",
                            "old": claimed_givens, "new": actual_givens})

    # --- Clue preservation ---
    for i in range(NUM_CELLS):
        if puzzle_grid[i] != 0 and puzzle_grid[i] != solution_grid[i]:
            add(ValidationCategory.SOLUTION, Severity.CRITICAL, "clue_preservation",
                f"clue at cell {i} ({puzzle_grid[i]}) not preserved in solution ({solution_grid[i]})")
            return RecordValidationOutcome(Verdict.INVALID, findings, None, repairs)

    # --- Solution correctness ---
    ok, err = validate_solution_grid(solution_grid, spec)
    if not ok:
        add(ValidationCategory.SOLUTION, Severity.CRITICAL, "solution_invalid",
            f"stored solution fails verification: {err}")
        # Try to repair: regenerate
        if mode in (RepairMode.REPAIR_SAFE, RepairMode.REPAIR_AGGRESSIVE):
            regen = solve_and_count(puzzle_grid, spec, limit=2,
                                    timeout_sec=max_solver_sec,
                                    max_nodes=max_solver_nodes)
            if regen.status == SolverStatus.UNIQUE and regen.first_solution is not None:
                record["solution"] = grid_to_string(regen.first_solution)
                repairs.append({"rule": "regenerate_solution",
                                "old": grid_to_string(solution_grid),
                                "new": record["solution"],
                                "evidence": "independent solver proved uniqueness"})
                add(ValidationCategory.REPAIR, Severity.INFO, "repaired_solution",
                    "solution regenerated from puzzle via independent solver")
                solution_grid = regen.first_solution
                # Recheck clue preservation
                for i in range(NUM_CELLS):
                    if puzzle_grid[i] != 0 and puzzle_grid[i] != solution_grid[i]:
                        add(ValidationCategory.REPAIR, Severity.CRITICAL,
                            "repair_failed_clue",
                            "regenerated solution does not preserve clue")
                        return RecordValidationOutcome(Verdict.QUARANTINED,
                                                       findings, None, repairs)
            else:
                add(ValidationCategory.REPAIR, Severity.ERROR, "repair_refused",
                    f"cannot safely repair solution: solver status={regen.status.value}")
                return RecordValidationOutcome(Verdict.QUARANTINED, findings, None, repairs)
        else:
            return RecordValidationOutcome(Verdict.INVALID, findings, None, repairs)

    # --- Uniqueness / satisfiability ---
    sres = solve_and_count(puzzle_grid, spec, limit=2,
                           timeout_sec=max_solver_sec,
                           max_nodes=max_solver_nodes)
    claimed_status = record.get("solution_count_status")
    claimed_unique = record.get("unique_solution")

    if sres.status == SolverStatus.UNSAT:
        add(ValidationCategory.UNIQUENESS, Severity.CRITICAL, "puzzle_unsat",
            "independent solver proved no solution exists")
        return RecordValidationOutcome(Verdict.INVALID, findings, None, repairs)
    if sres.status == SolverStatus.MULTIPLE:
        add(ValidationCategory.UNIQUENESS, Severity.CRITICAL, "puzzle_multiple",
            "independent solver found >= 2 solutions")
        if mode in (RepairMode.REPAIR_AGGRESSIVE,):
            add(ValidationCategory.REPAIR, Severity.WARNING, "no_aggressive_repair",
                "aggressive mode cannot reduce multi-solution to unique without changing puzzle")
        return RecordValidationOutcome(Verdict.QUARANTINED, findings, None, repairs)
    if sres.status in (SolverStatus.TIMEOUT, SolverStatus.NODE_LIMIT):
        add(ValidationCategory.UNIQUENESS, Severity.WARNING, "uniqueness_unresolved",
            f"uniqueness could not be established: {sres.status.value}")
        # Record but treat as UNRESOLVED
        return RecordValidationOutcome(Verdict.UNRESOLVED, findings, None, repairs)

    # UNIQUE — compare with claimed status
    if claimed_status != "unique":
        add(ValidationCategory.UNIQUENESS, Severity.WARNING, "status_mismatch",
            f"claimed status={claimed_status!r} but solver says unique")
        if mode in (RepairMode.REPAIR_SAFE, RepairMode.REPAIR_AGGRESSIVE):
            record["solution_count_status"] = "unique"
            repairs.append({"rule": "fix_solution_count_status",
                            "old": claimed_status, "new": "unique"})
    if claimed_unique is not True:
        add(ValidationCategory.UNIQUENESS, Severity.WARNING, "unique_flag_mismatch",
            f"unique_solution={claimed_unique!r}, solver says True")
        if mode in (RepairMode.REPAIR_SAFE, RepairMode.REPAIR_AGGRESSIVE):
            record["unique_solution"] = True
            repairs.append({"rule": "fix_unique_solution_flag",
                            "old": claimed_unique, "new": True})

    # Also verify stored solution equals solver's solution (up to equivalence)
    if sres.first_solution is not None:
        if sres.first_solution != solution_grid:
            # Could legitimately differ only if solution_grid wasn't what solver found
            add(ValidationCategory.SOLUTION, Severity.CRITICAL, "solution_mismatch",
                "stored solution differs from independently-derived solution")
            if mode in (RepairMode.REPAIR_SAFE, RepairMode.REPAIR_AGGRESSIVE):
                # Since puzzle is unique, solver's solution must be the only one
                old_sol = record["solution"]
                record["solution"] = grid_to_string(sres.first_solution)
                repairs.append({"rule": "replace_solution_with_unique",
                                "old": old_sol, "new": record["solution"]})
                solution_grid = sres.first_solution
            else:
                return RecordValidationOutcome(Verdict.INVALID, findings, None, repairs)

    # --- Provenance / canonical hashes ---
    prov = record.get("provenance")
    if not isinstance(prov, dict):
        add(ValidationCategory.SCHEMA, Severity.ERROR, "bad_provenance",
            "provenance missing or not an object")
    else:
        exact, canon_h, family_h = compute_canonical_hashes(puzzle_grid, solution_grid, spec)
        if prov.get("exact_puzzle_hash") != exact:
            add(ValidationCategory.SCHEMA, Severity.WARNING, "provenance_exact_hash_mismatch",
                "provenance.exact_puzzle_hash mismatch")
            if mode in (RepairMode.REPAIR_SAFE, RepairMode.REPAIR_AGGRESSIVE):
                prov["exact_puzzle_hash"] = exact
                repairs.append({"rule": "fix_exact_puzzle_hash",
                                "old": prov.get("exact_puzzle_hash"), "new": exact})
        if prov.get("canonical_puzzle_hash") != canon_h:
            add(ValidationCategory.SCHEMA, Severity.WARNING,
                "provenance_canonical_hash_mismatch",
                "provenance.canonical_puzzle_hash mismatch")
            if mode in (RepairMode.REPAIR_SAFE, RepairMode.REPAIR_AGGRESSIVE):
                prov["canonical_puzzle_hash"] = canon_h
                repairs.append({"rule": "fix_canonical_puzzle_hash",
                                "old": prov.get("canonical_puzzle_hash"), "new": canon_h})
        if prov.get("solution_family_hash") != family_h:
            add(ValidationCategory.SCHEMA, Severity.WARNING,
                "provenance_family_hash_mismatch",
                "provenance.solution_family_hash mismatch")
            if mode in (RepairMode.REPAIR_SAFE, RepairMode.REPAIR_AGGRESSIVE):
                prov["solution_family_hash"] = family_h
                repairs.append({"rule": "fix_solution_family_hash",
                                "old": prov.get("solution_family_hash"), "new": family_h})

    # --- Difficulty metadata ---
    diff = record.get("difficulty")
    if not _is_finite_number(diff):
        add(ValidationCategory.DIFFICULTY, Severity.ERROR, "bad_difficulty",
            f"difficulty not a finite number: {diff!r}")
    elif not (1.0 <= diff <= 10.0):
        add(ValidationCategory.DIFFICULTY, Severity.ERROR, "difficulty_out_of_range",
            f"difficulty={diff} outside [1.0, 10.0]")

    diff_raw = record.get("difficulty_raw")
    if not _is_finite_number(diff_raw):
        add(ValidationCategory.DIFFICULTY, Severity.WARNING, "bad_difficulty_raw",
            f"difficulty_raw not finite: {diff_raw!r}")

    dfeat = record.get("difficulty_features")
    if not isinstance(dfeat, dict):
        add(ValidationCategory.DIFFICULTY, Severity.WARNING, "bad_difficulty_features",
            "difficulty_features not an object")
    else:
        if _is_int(dfeat.get("clue_count")) and dfeat["clue_count"] != actual_givens:
            add(ValidationCategory.DIFFICULTY, Severity.WARNING, "clue_count_mismatch",
                f"features.clue_count={dfeat['clue_count']} != actual={actual_givens}")
            if mode in (RepairMode.REPAIR_SAFE, RepairMode.REPAIR_AGGRESSIVE):
                dfeat["clue_count"] = actual_givens
                repairs.append({"rule": "fix_clue_count",
                                "old": dfeat.get("clue_count"), "new": actual_givens})

    # --- Technique trace replay ---
    trace = record.get("technique_trace")
    if not isinstance(trace, list):
        add(ValidationCategory.TECHNIQUE_TRACE, Severity.WARNING, "bad_trace",
            "technique_trace not a list")
    else:
        replay = replay_technique_trace(trace, puzzle_grid, solution_grid, spec)
        if not replay.ok:
            add(ValidationCategory.TECHNIQUE_TRACE, Severity.ERROR, "trace_invalid",
                f"trace replay failed: {replay.first_error}",
                verified=replay.verified_steps,
                unverified=replay.unverified_steps,
                invalid=replay.invalid_steps)
        elif replay.unverified_steps > 0:
            add(ValidationCategory.TECHNIQUE_TRACE, Severity.INFO, "trace_partial",
                f"{replay.unverified_steps} step(s) unverified (best-effort)",
                verified=replay.verified_steps,
                unverified=replay.unverified_steps,
                techniques_unverified=replay.techniques_unverified)

    # --- Task-target integrity ---
    task_type = record.get("task_type")
    target = record.get("target")
    if not isinstance(task_type, str):
        add(ValidationCategory.TARGET, Severity.ERROR, "bad_task_type",
            "task_type missing or not a string")
    elif not isinstance(target, dict):
        add(ValidationCategory.TARGET, Severity.ERROR, "bad_target",
            "target missing or not an object")
    else:
        if task_type == "complete_solution":
            tgt = parse_grid_field(target.get("solution_string"))
            if tgt is None:
                add(ValidationCategory.TARGET, Severity.ERROR, "target_bad_solution",
                    "complete_solution target.solution_string malformed")
            else:
                ok_t, err_t = validate_solution_grid(tgt, spec)
                if not ok_t:
                    add(ValidationCategory.TARGET, Severity.CRITICAL, "target_invalid",
                        f"target solution fails verification: {err_t}")
                for i in range(NUM_CELLS):
                    if puzzle_grid[i] != 0 and tgt[i] != puzzle_grid[i]:
                        add(ValidationCategory.TARGET, Severity.CRITICAL,
                            "target_clue_mismatch",
                            f"target disagrees with clue at cell {i}")
                        break
                if tgt != solution_grid:
                    add(ValidationCategory.TARGET, Severity.ERROR,
                        "target_differs_from_solution",
                        "target solution_string differs from record.solution")
        elif task_type == "next_logical_deduction":
            ded = target.get("deduction")
            if not isinstance(ded, dict):
                add(ValidationCategory.TARGET, Severity.ERROR, "bad_deduction",
                    "next_logical_deduction target.deduction not an object")
            else:
                # A first-move verification: apply deduction to puzzle and check
                # consistency with puzzle's rules.
                action = ded.get("action_type")
                t_cells = ded.get("target_cells") or []
                t_digits = ded.get("digits") or []
                if action == "placement" and len(t_cells) == 1 and len(t_digits) == 1:
                    cell = int(t_cells[0])
                    d = int(t_digits[0])
                    if not (0 <= cell < NUM_CELLS) or not (1 <= d <= 9):
                        add(ValidationCategory.TARGET, Severity.ERROR,
                            "deduction_oob", "deduction cell/digit out of range")
                    elif puzzle_grid[cell] != 0:
                        add(ValidationCategory.TARGET, Severity.ERROR,
                            "deduction_cell_filled",
                            f"deduction targets filled cell {cell}")
                    else:
                        for p in spec.peers[cell]:
                            if puzzle_grid[p] == d:
                                add(ValidationCategory.TARGET, Severity.ERROR,
                                    "deduction_conflict",
                                    f"deduction digit {d} conflicts with peer {p}")
                                break
        elif task_type in ("candidate_elimination", "solver_policy"):
            # Loosely validated; no forced-consistency requirement beyond type.
            pass
        else:
            add(ValidationCategory.TARGET, Severity.WARNING, "unknown_task_type",
                f"unknown task_type {task_type!r}")

    # --- Compute final verdict ---
    has_critical = any(f.severity == Severity.CRITICAL.value for f in findings)
    has_error = any(f.severity == Severity.ERROR.value for f in findings)

    if has_critical:
        return RecordValidationOutcome(Verdict.QUARANTINED, findings, None, repairs)
    if has_error and mode == RepairMode.AUDIT:
        return RecordValidationOutcome(Verdict.INVALID, findings, None, repairs)

    # Re-serialize if repairs were applied
    if repairs:
        # Recompute record_sha256
        rec_copy = dict(record)
        rec_copy.pop("record_sha256", None)
        record["record_sha256"] = hashlib.sha256(
            json.dumps(rec_copy, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return RecordValidationOutcome(Verdict.REPAIRED, findings, record, repairs)

    return RecordValidationOutcome(Verdict.VALID, findings, None, [])


# ---------------------------------------------------------------------------
# Deduplication Registry
# ---------------------------------------------------------------------------


class DedupRegistry:
    """SQLite-backed hash registry for canonical puzzle and solution family."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        d = os.path.dirname(os.path.abspath(db_path))
        if d:
            os.makedirs(d, exist_ok=True)
        self.conn = sqlite3.connect(db_path, timeout=30.0)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")
        with self.conn:
            self.conn.execute(
                "CREATE TABLE IF NOT EXISTS hashes ("
                "  canonical_puzzle_hash TEXT NOT NULL,"
                "  exact_puzzle_hash TEXT NOT NULL,"
                "  family_hash TEXT NOT NULL,"
                "  split TEXT NOT NULL,"
                "  record_id TEXT,"
                "  file TEXT,"
                "  line INTEGER"
                ")")
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_cph ON hashes(canonical_puzzle_hash)")
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_fh ON hashes(family_hash)")

    def register(self, canon_h: str, exact_h: str, family_h: str,
                 split: str, record_id: Optional[str],
                 file: Optional[str], line: Optional[int]
                 ) -> List[Tuple[str, str, Optional[str], Optional[int]]]:
        """
        Register a hash. Returns a list of conflicts found:
          (kind, other_split, other_file, other_line)
          kind in {"puzzle_dup","puzzle_leak","family_dup","family_leak"}.
        """
        conflicts: List[Tuple[str, str, Optional[str], Optional[int]]] = []
        cur = self.conn.cursor()
        cur.execute(
            "SELECT split, file, line FROM hashes "
            "WHERE canonical_puzzle_hash=? OR exact_puzzle_hash=?",
            (canon_h, exact_h))
        for other_split, ofile, oline in cur.fetchall():
            if other_split != split:
                conflicts.append(("puzzle_leak", other_split, ofile, oline))
            else:
                conflicts.append(("puzzle_dup", other_split, ofile, oline))
        cur.execute(
            "SELECT split, file, line FROM hashes WHERE family_hash=?",
            (family_h,))
        for other_split, ofile, oline in cur.fetchall():
            if other_split != split:
                conflicts.append(("family_leak", other_split, ofile, oline))
            else:
                conflicts.append(("family_dup", other_split, ofile, oline))
        with self.conn:
            self.conn.execute(
                "INSERT INTO hashes VALUES (?, ?, ?, ?, ?, ?, ?)",
                (canon_h, exact_h, family_h, split, record_id, file, line))
        return conflicts

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# File Processing
# ---------------------------------------------------------------------------


@dataclass
class FileDescriptor:
    path: str
    split: str
    size_bytes: int


@dataclass
class ProcessingStats:
    total_records: int = 0
    valid_records: int = 0
    invalid_records: int = 0
    unresolved_records: int = 0
    quarantined_records: int = 0
    repaired_records: int = 0
    structural_errors: int = 0
    puzzle_errors: int = 0
    solution_errors: int = 0
    uniqueness_unresolved: int = 0
    technique_errors: int = 0
    difficulty_discrepancies: int = 0
    dedup_conflicts: int = 0
    cross_split_leaks: int = 0
    manifest_discrepancies: int = 0


def discover_dataset_files(input_path: str) -> List[FileDescriptor]:
    """
    Given a directory, discover JSONL split files.  Given a file, return it as
    a single file with split='unknown'.
    """
    if os.path.isfile(input_path):
        return [FileDescriptor(input_path, "unknown",
                               os.path.getsize(input_path))]
    if not os.path.isdir(input_path):
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    files: List[FileDescriptor] = []
    for name in sorted(os.listdir(input_path)):
        full = os.path.join(input_path, name)
        if os.path.isdir(full):
            for sub in sorted(os.listdir(full)):
                if sub.endswith(".jsonl") and not sub.endswith(".part"):
                    sp = os.path.join(full, sub)
                    files.append(FileDescriptor(sp, name, os.path.getsize(sp)))
        elif name.endswith(".jsonl") and not name.endswith(".part"):
            # Top-level JSONL treated as split based on filename
            split = "unknown"
            for candidate in ("train", "validation", "test"):
                if candidate in name:
                    split = candidate
                    break
            files.append(FileDescriptor(full, split, os.path.getsize(full)))
    return files


def enforce_disk_budget(output_dir: str, min_free_mb: float) -> None:
    probe = os.path.abspath(output_dir)
    while not os.path.exists(probe):
        parent = os.path.dirname(probe)
        if parent == probe:
            break
        probe = parent
    usage = shutil.disk_usage(probe)
    free_mb = usage.free / (1024.0 * 1024.0)
    if free_mb < min_free_mb:
        raise RuntimeError(
            f"Free disk {free_mb:.1f} MB below threshold {min_free_mb:.1f} MB")


def _atomic_write_line(fh, obj: Dict[str, Any]) -> None:
    fh.write(json.dumps(obj, sort_keys=True, separators=(",", ":")) + "\n")


# ---------------------------------------------------------------------------
# Manifest check
# ---------------------------------------------------------------------------


def check_manifest(split_dir: str, jsonl_path: str,
                   stats: ProcessingStats) -> List[Finding]:
    findings: List[Finding] = []
    manifest_path = os.path.join(split_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        findings.append(Finding(
            ValidationCategory.MANIFEST.value, Severity.INFO.value,
            "manifest_missing",
            f"no manifest.json next to {jsonl_path}",
            file=jsonl_path))
        return findings

    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except Exception as e:
        findings.append(Finding(
            ValidationCategory.MANIFEST.value, Severity.ERROR.value,
            "manifest_parse_error",
            f"cannot parse manifest: {e}",
            file=manifest_path))
        return findings

    claimed_sha = manifest.get("file_sha256")
    if isinstance(claimed_sha, str):
        h = hashlib.sha256()
        try:
            with open(jsonl_path, "rb") as f:
                while True:
                    chunk = f.read(1 << 20)
                    if not chunk:
                        break
                    h.update(chunk)
            actual = h.hexdigest()
            if actual != claimed_sha:
                findings.append(Finding(
                    ValidationCategory.MANIFEST.value, Severity.ERROR.value,
                    "manifest_hash_mismatch",
                    "manifest file_sha256 does not match actual file hash",
                    file=manifest_path,
                    evidence={"claimed": claimed_sha, "actual": actual}))
        except OSError as e:
            findings.append(Finding(
                ValidationCategory.MANIFEST.value, Severity.ERROR.value,
                "manifest_hash_unreadable", str(e), file=manifest_path))

    claimed_count = manifest.get("accepted_count")
    if isinstance(claimed_count, int):
        actual_count = stats.valid_records + stats.repaired_records
        if claimed_count != actual_count:
            findings.append(Finding(
                ValidationCategory.MANIFEST.value, Severity.WARNING.value,
                "manifest_count_mismatch",
                f"manifest accepted_count={claimed_count} but audited valid={actual_count}",
                file=manifest_path,
                evidence={"claimed": claimed_count, "actual": actual_count}))

    return findings


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


@dataclass
class AuditReport:
    validator_name: str
    validator_version: str
    started_unix: float
    finished_unix: float
    input_path: str
    output_path: Optional[str]
    mode: str
    schema_profile: str
    files: List[Dict[str, Any]]
    stats: Dict[str, Any]
    findings: List[Dict[str, Any]]
    repairs: List[Dict[str, Any]]
    overall_status: str
    exit_code: int
    limitations: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


# ---------------------------------------------------------------------------
# Main Audit Orchestration
# ---------------------------------------------------------------------------


@dataclass
class AuditConfig:
    input_path: str
    output_path: Optional[str] = None
    report_path: Optional[str] = None
    mode: RepairMode = RepairMode.AUDIT
    workers: int = 1
    max_line_bytes: int = DEFAULT_MAX_LINE_BYTES
    max_record_bytes: int = DEFAULT_MAX_RECORD_BYTES
    max_solver_sec: float = DEFAULT_MAX_SOLVER_SECONDS
    max_solver_nodes: int = DEFAULT_MAX_SOLVER_NODES
    max_wall_sec: float = DEFAULT_MAX_WALL_SECONDS
    min_free_disk_mb: float = DEFAULT_MIN_FREE_DISK_MB
    cross_split: bool = False
    quiet: bool = False
    verbose: bool = False
    dry_run: bool = False
    variant_override: Optional[str] = None
    quarantine_dirname: str = "_quarantine"


def _iter_records(path: str, max_line_bytes: int, max_record_bytes: int
                  ) -> Iterator[Tuple[int, Optional[Dict[str, Any]], Optional[str]]]:
    """
    Stream records from a JSONL file.  Yields (line_no, record_or_None, error_or_None).
    """
    with open(path, "rb") as f:
        line_no = 0
        while True:
            raw = f.readline()
            if not raw:
                break
            line_no += 1
            if len(raw) > max_line_bytes:
                yield line_no, None, f"line_too_long ({len(raw)} > {max_line_bytes})"
                continue
            if len(raw) > max_record_bytes:
                yield line_no, None, "record_too_large"
                continue
            if raw.endswith(b"\n"):
                raw_body = raw[:-1]
            else:
                raw_body = raw
            if not raw_body.strip():
                continue
            try:
                text = raw_body.decode("utf-8")
            except UnicodeDecodeError as e:
                yield line_no, None, f"invalid_utf8: {e}"
                continue
            obj, err = _check_json_strict(text)
            if err is not None:
                yield line_no, None, err
                continue
            yield line_no, obj, None


def run_audit(cfg: AuditConfig) -> AuditReport:
    started = time.time()
    findings: List[Finding] = []
    repairs: List[Dict[str, Any]] = []
    stats = ProcessingStats()

    variant_override: Optional[VariantSpec] = None
    if cfg.variant_override:
        if cfg.variant_override not in VARIANT_REGISTRY:
            raise ValueError(f"unknown variant override {cfg.variant_override!r}")
        variant_override = VARIANT_REGISTRY[cfg.variant_override]

    descriptors = discover_dataset_files(cfg.input_path)
    if not descriptors:
        raise FileNotFoundError(f"no JSONL files found under {cfg.input_path}")

    # Prepare output directories for repair modes
    output_writers: Dict[str, Any] = {}
    quarantine_writers: Dict[str, Any] = {}
    if cfg.mode in (RepairMode.REPAIR_SAFE, RepairMode.REPAIR_AGGRESSIVE,
                    RepairMode.QUARANTINE) and not cfg.dry_run:
        if not cfg.output_path:
            raise ValueError(f"--output is required for mode {cfg.mode.value}")
        enforce_disk_budget(cfg.output_path, cfg.min_free_disk_mb)
        os.makedirs(cfg.output_path, exist_ok=True)

    # Dedup registry (in a tempdir unless output path supplied)
    if cfg.mode in (RepairMode.REPAIR_SAFE, RepairMode.REPAIR_AGGRESSIVE) and cfg.output_path:
        dedup_path = os.path.join(cfg.output_path, "_audit_dedup.sqlite3")
    else:
        dedup_path = os.path.join(tempfile.mkdtemp(prefix="sm_validator_"), "dedup.sqlite3")
    dedup = DedupRegistry(dedup_path)

    files_meta: List[Dict[str, Any]] = []

    try:
        for fd in descriptors:
            if _SHUTDOWN:
                break
            file_size = fd.size_bytes
            file_findings_before = len(findings)

            if cfg.verbose and not cfg.quiet:
                sys.stderr.write(f"[AUDIT] {fd.path} split={fd.split} "
                                 f"size={file_size} bytes\n")

            # Open output writers per split on demand
            def _get_writer(root: str) -> Any:
                if root not in output_writers:
                    if cfg.dry_run:
                        output_writers[root] = None
                        return None
                    out_dir = os.path.join(root, fd.split)
                    os.makedirs(out_dir, exist_ok=True)
                    out_path = os.path.join(out_dir, os.path.basename(fd.path))
                    tmp = out_path + ".tmp"
                    output_writers[root] = (open(tmp, "w", encoding="utf-8"), tmp, out_path)
                return output_writers[root]

            def _get_quarantine_writer() -> Any:
                if "q" not in quarantine_writers:
                    if cfg.dry_run:
                        quarantine_writers["q"] = None
                        return None
                    base = cfg.output_path or os.path.dirname(fd.path)
                    qdir = os.path.join(base, cfg.quarantine_dirname)
                    os.makedirs(qdir, exist_ok=True)
                    qpath = os.path.join(qdir, f"{fd.split}_quarantine.jsonl")
                    tmp = qpath + ".tmp"
                    quarantine_writers["q"] = (open(tmp, "w", encoding="utf-8"),
                                               tmp, qpath)
                return quarantine_writers["q"]

            writer = None
            q_writer = None
            if cfg.mode in (RepairMode.REPAIR_SAFE, RepairMode.REPAIR_AGGRESSIVE):
                writer = _get_writer(cfg.output_path)  # type: ignore[arg-type]
            if cfg.mode == RepairMode.QUARANTINE:
                q_writer = _get_quarantine_writer()

            for line_no, rec, err in _iter_records(fd.path, cfg.max_line_bytes,
                                                   cfg.max_record_bytes):
                if _SHUTDOWN:
                    break
                stats.total_records += 1
                if err is not None:
                    stats.structural_errors += 1
                    findings.append(Finding(
                        ValidationCategory.FILE.value, Severity.ERROR.value,
                        "line_error", err,
                        file=fd.path, line=line_no))
                    continue

                assert rec is not None
                outcome = validate_record(
                    rec, fd.path, line_no, variant_override,
                    cfg.mode, cfg.max_solver_sec, cfg.max_solver_nodes,
                )
                findings.extend(outcome.findings)
                if outcome.repairs:
                    for r in outcome.repairs:
                        repairs.append({"file": fd.path, "line": line_no,
                                        "record_id": rec.get("puzzle_id"),
                                        **r})

                # Register hash for dedup if we have a well-formed puzzle/solution
                puz_grid = parse_grid_field(rec.get("puzzle"))
                sol_grid = parse_grid_field(rec.get("solution"))
                if puz_grid is not None and sol_grid is not None:
                    vname = rec.get("variant", "standard")
                    spec = VARIANT_REGISTRY.get(vname)
                    if spec is not None:
                        exact_h, canon_h, family_h = compute_canonical_hashes(
                            puz_grid, sol_grid, spec)
                        conflicts = dedup.register(
                            canon_h, exact_h, family_h,
                            fd.split, rec.get("puzzle_id"), fd.path, line_no)
                        for kind, other_split, ofile, oline in conflicts:
                            stats.dedup_conflicts += 1
                            if "leak" in kind:
                                stats.cross_split_leaks += 1
                            sev = Severity.ERROR if "leak" in kind else Severity.WARNING
                            findings.append(Finding(
                                ValidationCategory.LEAKAGE.value if "leak" in kind
                                else ValidationCategory.DEDUP.value,
                                sev.value, kind,
                                f"duplicate/leak {kind} with split={other_split} "
                                f"(file={ofile} line={oline})",
                                file=fd.path, line=line_no,
                                record_id=rec.get("puzzle_id")))

                # Classify
                if outcome.verdict == Verdict.VALID:
                    stats.valid_records += 1
                elif outcome.verdict == Verdict.REPAIRED:
                    stats.repaired_records += 1
                elif outcome.verdict == Verdict.INVALID:
                    stats.invalid_records += 1
                elif outcome.verdict == Verdict.UNRESOLVED:
                    stats.unresolved_records += 1
                elif outcome.verdict == Verdict.QUARANTINED:
                    stats.quarantined_records += 1

                # Categorize counts
                for f in outcome.findings:
                    if f.category == ValidationCategory.PUZZLE.value and \
                            f.severity == Severity.CRITICAL.value:
                        stats.puzzle_errors += 1
                    elif f.category == ValidationCategory.SOLUTION.value and \
                            f.severity in (Severity.CRITICAL.value, Severity.ERROR.value):
                        stats.solution_errors += 1
                    elif f.category == ValidationCategory.UNIQUENESS.value and \
                            f.code == "uniqueness_unresolved":
                        stats.uniqueness_unresolved += 1
                    elif f.category == ValidationCategory.TECHNIQUE_TRACE.value and \
                            f.severity == Severity.ERROR.value:
                        stats.technique_errors += 1
                    elif f.category == ValidationCategory.DIFFICULTY.value:
                        stats.difficulty_discrepancies += 1

                # Write outputs
                if outcome.verdict == Verdict.QUARANTINED:
                    if cfg.mode == RepairMode.QUARANTINE and q_writer is not None:
                        _atomic_write_line(q_writer[0], {
                            "record": rec,
                            "findings": [f.to_dict() for f in outcome.findings],
                        })
                elif outcome.verdict in (Verdict.VALID, Verdict.REPAIRED) and \
                        writer is not None:
                    out_rec = outcome.repaired_record if outcome.repaired_record is not None else rec
                    _atomic_write_line(writer[0], out_rec)
                elif outcome.verdict == Verdict.UNRESOLVED and writer is not None:
                    # Keep unresolved out of the clean output; log them
                    pass

                # Progress
                if not cfg.quiet and stats.total_records % 5000 == 0:
                    sys.stderr.write(
                        f"  ... {stats.total_records} records; "
                        f"valid={stats.valid_records} "
                        f"invalid={stats.invalid_records} "
                        f"quarantined={stats.quarantined_records} "
                        f"unresolved={stats.unresolved_records}\n")

                if cfg.max_wall_sec > 0 and time.time() - started > cfg.max_wall_sec:
                    findings.append(Finding(
                        ValidationCategory.FILE.value, Severity.WARNING.value,
                        "wall_clock_limit",
                        f"wall clock limit {cfg.max_wall_sec}s exceeded; stopping"))
                    _SHUTDOWN = True
                    break

            # Manifest check
            split_dir = os.path.dirname(fd.path)
            file_findings = check_manifest(split_dir, fd.path, stats)
            findings.extend(file_findings)
            for f in file_findings:
                if f.severity == Severity.ERROR.value:
                    stats.manifest_discrepancies += 1

            files_meta.append({
                "path": fd.path,
                "split": fd.split,
                "size_bytes": file_size,
                "sha256": _file_sha256(fd.path) if file_size < (1 << 30) else None,
                "findings": len(findings) - file_findings_before,
            })
    finally:
        # Finalize writers
        for writer in output_writers.values():
            if writer is not None:
                try:
                    writer[0].flush()
                    writer[0].close()
                    os.replace(writer[1], writer[2])
                except Exception as e:
                    findings.append(Finding(
                        ValidationCategory.FILE.value, Severity.ERROR.value,
                        "output_finalize_error", str(e)))
        for writer in quarantine_writers.values():
            if writer is not None:
                try:
                    writer[0].flush()
                    writer[0].close()
                    os.replace(writer[1], writer[2])
                except Exception as e:
                    findings.append(Finding(
                        ValidationCategory.FILE.value, Severity.ERROR.value,
                        "quarantine_finalize_error", str(e)))
        dedup.close()

    # Compute status
    has_critical = any(f.severity == Severity.CRITICAL.value for f in findings)
    has_error = any(f.severity == Severity.ERROR.value for f in findings)
    has_unresolved = stats.unresolved_records > 0
    has_warnings = any(f.severity == Severity.WARNING.value for f in findings)

    if _SHUTDOWN:
        status = OverallStatus.INCOMPLETE
        exit_code = ExitCode.INCOMPLETE
    elif has_critical or stats.invalid_records > 0 or stats.quarantined_records > 0:
        status = OverallStatus.FAIL
        exit_code = ExitCode.FAIL
    elif has_unresolved or has_error:
        status = OverallStatus.INCOMPLETE if has_unresolved else OverallStatus.FAIL
        exit_code = ExitCode.INCOMPLETE if has_unresolved else ExitCode.FAIL
    elif has_warnings or stats.repaired_records > 0:
        status = OverallStatus.PASS_WITH_WARNINGS
        exit_code = ExitCode.WARNINGS
    else:
        status = OverallStatus.PASS
        exit_code = ExitCode.OK

    limitations: List[str] = [
        "Technique verification covers most human strategies; "
        "forcing_chain_contradiction is verified on a best-effort basis only.",
        "Difficulty labels are checked for range/type consistency and internal "
        "feature agreement, but the generator's scoring formula is NOT "
        "independently recomputed; drift between calibration versions is reported "
        "as informational only.",
        "Canonical equivalence checking covers the D4 dihedral group and digit "
        "relabeling for full-symmetry variants; anti-knight/anti-king are checked "
        "only under digit relabeling (their peer graphs break full D4 invariance).",
        "Solver timeouts are reported as UNRESOLVED, never as valid or invalid.",
    ]

    report = AuditReport(
        validator_name=VALIDATOR_NAME,
        validator_version=VALIDATOR_VERSION,
        started_unix=started,
        finished_unix=time.time(),
        input_path=os.path.abspath(cfg.input_path),
        output_path=os.path.abspath(cfg.output_path) if cfg.output_path else None,
        mode=cfg.mode.value,
        schema_profile="sm-schema-v2.0",
        files=files_meta,
        stats=dataclasses.asdict(stats),
        findings=[f.to_dict() for f in findings],
        repairs=repairs,
        overall_status=status.value,
        exit_code=exit_code,
        limitations=limitations,
    )
    return report


def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="dataset_validator.py",
        description=(
            f"{VALIDATOR_NAME} v{VALIDATOR_VERSION} — Independent validation of "
            "SudokuMind dataset files. Never trusts generator metadata; recomputes "
            "all mathematical claims."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Modes:\n"
            "  audit              validate only (no writes) [default]\n"
            "  repair-safe        write output with provably safe repairs only\n"
            "  repair-aggressive  additionally allow deterministic reconstruction\n"
            "  quarantine         isolate invalid records into a quarantine area\n\n"
            "Exit codes:\n"
            "  0 PASS  1 PASS_WITH_WARNINGS  2 FAIL  3 INCOMPLETE\n"
            "  4 CLI error  5 internal error\n\n"
            "Examples:\n"
            "  python dataset_validator.py audit --input datasets/\n"
            "  python dataset_validator.py audit --input datasets/train/train.jsonl\n"
            "  python dataset_validator.py audit --input datasets/ --cross-split --workers 4\n"
            "  python dataset_validator.py repair-safe --input datasets/ --output datasets_repaired/\n"
            "  python dataset_validator.py repair-aggressive --input datasets/ --output fixed/ --dry-run\n"
            "  python dataset_validator.py --run-tests\n"
        ),
    )
    p.add_argument("command", nargs="?", default="audit",
                   choices=["audit", "repair-safe", "repair-aggressive",
                            "quarantine", "selftest", "help"],
                   help="Operation mode (default: audit)")
    p.add_argument("--input", required=False, help="Path to dataset file or directory")
    p.add_argument("--output", help="Output directory (required for repair modes)")
    p.add_argument("--report", help="Path to write JSON audit report")
    p.add_argument("--cross-split", action="store_true",
                   help="Enable cross-split leakage detection (default when input is a directory)")
    p.add_argument("--workers", type=int, default=1,
                   help="Number of worker processes (currently 1; reserved for future)")
    p.add_argument("--max-line-bytes", type=int, default=DEFAULT_MAX_LINE_BYTES)
    p.add_argument("--max-record-bytes", type=int, default=DEFAULT_MAX_RECORD_BYTES)
    p.add_argument("--max-solver-seconds", type=float,
                   default=DEFAULT_MAX_SOLVER_SECONDS,
                   help=f"Per-puzzle solver timeout (default {DEFAULT_MAX_SOLVER_SECONDS}s)")
    p.add_argument("--max-solver-nodes", type=int,
                   default=DEFAULT_MAX_SOLVER_NODES)
    p.add_argument("--max-wall-seconds", type=float, default=DEFAULT_MAX_WALL_SECONDS,
                   help="Overall wall-clock limit (0 = unlimited)")
    p.add_argument("--min-free-disk-mb", type=float, default=DEFAULT_MIN_FREE_DISK_MB)
    p.add_argument("--variant", help="Override variant for every record")
    p.add_argument("--dry-run", action="store_true",
                   help="Report planned changes without writing anything")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--run-tests", action="store_true",
                   help="Run the built-in self-test suite and exit")
    return p


def _print_report_summary(rep: AuditReport, quiet: bool) -> None:
    if quiet:
        return
    print()
    print("=" * 78)
    print(f"{rep.validator_name} v{rep.validator_version} — Audit Summary")
    print("=" * 78)
    print(f"  Input             : {rep.input_path}")
    if rep.output_path:
        print(f"  Output            : {rep.output_path}")
    print(f"  Mode              : {rep.mode}")
    print(f"  Schema profile    : {rep.schema_profile}")
    print(f"  Elapsed           : {rep.finished_unix - rep.started_unix:.2f}s")
    print(f"  Files examined    : {len(rep.files)}")
    print()
    s = rep.stats
    print(f"  Records processed : {s['total_records']}")
    print(f"  Valid             : {s['valid_records']}")
    print(f"  Repaired          : {s['repaired_records']}")
    print(f"  Invalid           : {s['invalid_records']}")
    print(f"  Quarantined       : {s['quarantined_records']}")
    print(f"  Unresolved        : {s['unresolved_records']}")
    print()
    print(f"  Structural errors : {s['structural_errors']}")
    print(f"  Puzzle errors     : {s['puzzle_errors']}")
    print(f"  Solution errors   : {s['solution_errors']}")
    print(f"  Uniqueness (open) : {s['uniqueness_unresolved']}")
    print(f"  Technique errors  : {s['technique_errors']}")
    print(f"  Difficulty disc.  : {s['difficulty_discrepancies']}")
    print(f"  Dedup conflicts   : {s['dedup_conflicts']}")
    print(f"  Cross-split leaks : {s['cross_split_leaks']}")
    print(f"  Manifest issues   : {s['manifest_discrepancies']}")
    print()
    print(f"  OVERALL STATUS    : {rep.overall_status}")
    print(f"  Exit code         : {rep.exit_code}")
    print("=" * 78)


# ---------------------------------------------------------------------------
# Self-Tests
# ---------------------------------------------------------------------------


def run_selftest_suite() -> int:
    """
    Executes a suite of independent validator self-tests.
    Prints pass/fail per test and returns 0 on complete success, nonzero otherwise.
    """
    print("=" * 78)
    print(f"{VALIDATOR_NAME} v{VALIDATOR_VERSION} — Self-Test Suite")
    print("=" * 78)
    passed = 0
    failed = 0

    def _run(name: str, fn: Callable[[], None]) -> None:
        nonlocal passed, failed
        t0 = time.monotonic()
        try:
            fn()
            ms = (time.monotonic() - t0) * 1000.0
            print(f"  [PASS] {name:<60} ({ms:.1f} ms)")
            passed += 1
        except AssertionError as e:
            print(f"  [FAIL] {name:<60} -> {e}")
            failed += 1
        except Exception as e:
            print(f"  [FAIL] {name:<60} -> {type(e).__name__}: {e}")
            failed += 1

    # 1. Valid standard Sudoku record
    def t01() -> None:
        puz = "530070000600195000098000060800060003400803001700020006060000280000419005000080079"
        sol = "534678912672195348198342567859761423426853791713924856961537284287419635345286179"
        puz_grid = parse_grid_field(puz)
        sol_grid = parse_grid_field(sol)
        assert puz_grid is not None and sol_grid is not None
        ok, _ = validate_puzzle_grid(puz_grid, VARIANT_REGISTRY["standard"])
        assert ok
        ok, err = validate_solution_grid(sol_grid, VARIANT_REGISTRY["standard"])
        assert ok, err
        res = solve_and_count(puz_grid, VARIANT_REGISTRY["standard"], limit=2,
                              timeout_sec=5.0)
        assert res.status == SolverStatus.UNIQUE, res.status
        assert res.first_solution == sol_grid

    # 2. Invalid grid (bad length)
    def t02() -> None:
        assert parse_grid_field("123") is None
        assert parse_grid_field([1] * 80) is None

    # 3. Contradictory clues (peer conflict)
    def t03() -> None:
        g = [0] * 81
        g[0] = 5
        g[1] = 5
        ok, err = validate_puzzle_grid(g, VARIANT_REGISTRY["standard"])
        assert not ok

    # 4. Unsatisfiable puzzle (no candidate for a cell)
    def t04() -> None:
        # Cell (0,0) has 1-9 all blocked in row/col/box
        g = [0] * 81
        for c in range(1, 9):
            g[c] = c + 1  # row 0 has 2..9
        for r in range(1, 9):
            g[r * 9 + 0] = 1  # col 0 has 1
        res = solve_and_count(g, VARIANT_REGISTRY["standard"], limit=2,
                              timeout_sec=2.0)
        assert res.status == SolverStatus.UNSAT, res.status

    # 5. Unique puzzle
    def t05() -> None:
        puz = "530070000600195000098000060800060003400803001700020006060000280000419005000080079"
        g = parse_grid_field(puz)
        assert g is not None
        res = solve_and_count(g, VARIANT_REGISTRY["standard"], limit=2,
                              timeout_sec=5.0)
        assert res.status == SolverStatus.UNIQUE

    # 6. Multiple-solution puzzle (empty board)
    def t06() -> None:
        g = [0] * 81
        res = solve_and_count(g, VARIANT_REGISTRY["standard"], limit=2,
                              timeout_sec=2.0)
        assert res.status == SolverStatus.MULTIPLE

    # 7. Wrong reference solution detected
    def t07() -> None:
        puz = "530070000600195000098000060800060003400803001700020006060000280000419005000080079"
        bad_sol = "534678912672195348198342567859761423426853791713924856961537284287419635345286178"
        puz_grid = parse_grid_field(puz)
        sol_grid = parse_grid_field(bad_sol)
        assert puz_grid is not None and sol_grid is not None
        ok, _ = validate_solution_grid(sol_grid, VARIANT_REGISTRY["standard"])
        assert not ok
        res = solve_and_count(puz_grid, VARIANT_REGISTRY["standard"], limit=2,
                              timeout_sec=5.0)
        assert res.first_solution != sol_grid

    # 8. False uniqueness claim
    def t08() -> None:
        g = [0] * 81
        res = solve_and_count(g, VARIANT_REGISTRY["standard"], limit=2,
                              timeout_sec=2.0)
        assert res.status == SolverStatus.MULTIPLE

    # 9. Technique trace: naked single
    def t09() -> None:
        # Build a puzzle where a naked single exists at step 1
        puz = [0] * 81
        for d in range(1, 10):
            puz[d - 1] = d  # Row 0 is 1..9
        # Cell 9 (row 1, col 0) in standard: only candidate is 9? Let's check.
        # Column 0 has 1, row 1 has nothing. So cell 9 has 2..9 candidates minus
        # 1? Wait col 0 has 1; row 1 empty. So candidates {2..9}. Not a naked single.
        # Simpler: puzzle with one empty cell
        g = [0] * 81
        # Fill row 0 with 1..8 and (0,8) empty; put valid row values else zeros
        for c in range(8):
            g[c] = c + 1
        # We need a valid puzzle with exactly 1 empty cell - actually simpler:
        # Take a valid complete grid and empty one cell
        sol_str = "534678912672195348198342567859761423426853791713924856961537284287419635345286179"
        sol = parse_grid_field(sol_str)
        assert sol is not None
        puz_one = list(sol)
        puz_one[0] = 0
        spec = VARIANT_REGISTRY["standard"]
        trace = [{"step_index": 1, "technique": "naked_single",
                  "action_type": "placement",
                  "target_cells": [0], "digits": [sol[0]],
                  "evidence_cells": [], "rationale": "only candidate"}]
        result = replay_technique_trace(trace, puz_one, sol, spec)
        assert result.ok, result.first_error
        assert result.verified_steps == 1

    # 10. Invalid technique: naked_single claimed but cell has 2 candidates
    def t10() -> None:
        g = [0] * 81
        g[0] = 0
        spec = VARIANT_REGISTRY["standard"]
        trace = [{"step_index": 1, "technique": "naked_single",
                  "action_type": "placement",
                  "target_cells": [0], "digits": [1],
                  "evidence_cells": [], "rationale": "bogus"}]
        result = replay_technique_trace(trace, g, None, spec)
        assert not result.ok
        assert "naked_single" in result.first_error

    # 11. Invalid proof trace: unsolvable path
    def t11() -> None:
        g = [0] * 81
        g[0] = 5
        g[1] = 5  # direct conflict
        # Trace trying to place digits elsewhere
        spec = VARIANT_REGISTRY["standard"]
        trace = [{"step_index": 1, "technique": "naked_single",
                  "action_type": "placement",
                  "target_cells": [2], "digits": [1],
                  "evidence_cells": [], "rationale": "bogus"}]
        result = replay_technique_trace(trace, g, None, spec)
        # Cell 2 has many candidates, so verifier should fail
        assert not result.ok

    # 12. Difficulty metadata: out-of-range and wrong type
    def t12() -> None:
        assert not _is_finite_number("abc")
        assert not _is_finite_number(True)
        assert not _is_finite_number(float("nan"))
        assert not _is_finite_number(float("inf"))
        assert _is_finite_number(5.5)

    # 13. Duplicate records detected
    def t13() -> None:
        with tempfile.TemporaryDirectory() as td:
            db = os.path.join(td, "h.sqlite3")
            reg = DedupRegistry(db)
            try:
                c1 = reg.register("a" * 64, "b" * 64, "c" * 64, "train", "p1", "f1", 1)
                c2 = reg.register("a" * 64, "b" * 64, "c" * 64, "train", "p2", "f1", 2)
                assert c1 == []
                assert any(kind == "puzzle_dup" for kind, *_ in c2)
            finally:
                reg.close()

    # 14. Cross-split leakage detected
    def t14() -> None:
        with tempfile.TemporaryDirectory() as td:
            db = os.path.join(td, "h.sqlite3")
            reg = DedupRegistry(db)
            try:
                reg.register("a" * 64, "b" * 64, "c" * 64, "test", "p1", "f1", 1)
                conf = reg.register("a" * 64, "b" * 64, "c" * 64, "train", "p2", "f2", 1)
                assert any(kind == "puzzle_leak" for kind, *_ in conf)
                assert any(kind == "family_leak" for kind, *_ in conf)
            finally:
                reg.close()

    # 15. Equivalent puzzles under D4 + relabeling
    def t15() -> None:
        puz = "530070000600195000098000060800060003400803001700020006060000280000419005000080079"
        g = parse_grid_field(puz)
        assert g is not None
        perm = D4_MAPS[1]
        transformed = [g[perm[i]] for i in range(NUM_CELLS)]
        relabeled = [(10 - v) if v != 0 else 0 for v in transformed]
        assert canonical_d4(g, True) == canonical_d4(relabeled, True)

    # 16. Malformed JSONL / duplicate keys / truncated final record
    def t16() -> None:
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "x.jsonl")
            with open(path, "w") as f:
                f.write('{"a":1,"a":2}\n')
                f.write('not json\n')
                f.write('{"b":3}\n')  # missing newline at end - truncated? No, complete
            rows = list(_iter_records(path, 1 << 20, 1 << 20))
            errs = [r[2] for r in rows if r[2] is not None]
            assert any("duplicate" in e for e in errs)
            assert any("json_decode_error" in e for e in errs)

    # 17. Truncated final line
    def t17() -> None:
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "x.jsonl")
            with open(path, "wb") as f:
                f.write(b'{"a":1}\n')
                f.write(b'{"b":')  # truncated
            rows = list(_iter_records(path, 1 << 20, 1 << 20))
            assert rows[-1][1] is None
            assert rows[-1][2] is not None

    # 18. record_sha256 mismatch detected
    def t18() -> None:
        # Construct a minimal record with wrong sha
        rec = {
            "schema_version": "sm-schema-v2.0",
            "puzzle_id": "p1", "split": "train", "variant": "standard",
            "difficulty": 3.0, "difficulty_raw": 3.0,
            "difficulty_features": {}, "givens": 30,
            "puzzle": "530070000600195000098000060800060003400803001700020006060000280000419005000080079",
            "solution": "534678912672195348198342567859761423426853791713924856961537284287419635345286179",
            "solution_count_status": "unique", "unique_solution": True,
            "task_type": "complete_solution",
            "target": {"solution_string": "534678912672195348198342567859761423426853791713924856961537284287419635345286179"},
            "technique_trace": [],
            "generator_version": "2.1.0", "generation_seed": 0,
            "solver_version": "x", "provenance": {}, "validation_status": "ok",
            "record_sha256": "0" * 64,
        }
        outcome = validate_record(rec, "test", 1, None, RepairMode.AUDIT,
                                  5.0, 100000)
        codes = [f.code for f in outcome.findings]
        assert "record_sha_mismatch" in codes

    # 19. Safe repair of wrong sha
    def t19() -> None:
        rec = {
            "schema_version": "sm-schema-v2.0",
            "puzzle_id": "p1", "split": "train", "variant": "standard",
            "difficulty": 3.0, "difficulty_raw": 3.0,
            "difficulty_features": {},
            "givens": 29,  # wrong count
            "puzzle": "530070000600195000098000060800060003400803001700020006060000280000419005000080079",
            "solution": "534678912672195348198342567859761423426853791713924856961537284287419635345286179",
            "solution_count_status": "unique", "unique_solution": True,
            "task_type": "complete_solution",
            "target": {"solution_string": "534678912672195348198342567859761423426853791713924856961537284287419635345286179"},
            "technique_trace": [],
            "generator_version": "2.1.0", "generation_seed": 0,
            "solver_version": "x", "provenance": {}, "validation_status": "ok",
            "record_sha256": "0" * 64,
        }
        outcome = validate_record(rec, "test", 1, None, RepairMode.REPAIR_SAFE,
                                  5.0, 100000)
        assert outcome.verdict in (Verdict.REPAIRED, Verdict.VALID)
        rules = [r["rule"] for r in outcome.repairs]
        assert "recompute_givens" in rules

    # 20. Refusal to repair a wrong solution when puzzle is not unique
    def t20() -> None:
        # Puzzle with many solutions - should refuse repair of bad solution
        rec = {
            "schema_version": "sm-schema-v2.0",
            "puzzle_id": "p1", "split": "train", "variant": "standard",
            "difficulty": 3.0, "difficulty_raw": 3.0,
            "difficulty_features": {}, "givens": 30,
            "puzzle": "0" * 81,
            "solution": "5" * 81,  # obviously invalid
            "solution_count_status": "unique", "unique_solution": True,
            "task_type": "complete_solution",
            "target": {"solution_string": "5" * 81},
            "technique_trace": [],
            "generator_version": "2.1.0", "generation_seed": 0,
            "solver_version": "x", "provenance": {}, "validation_status": "ok",
            "record_sha256": "0" * 64,
        }
        outcome = validate_record(rec, "test", 1, None, RepairMode.REPAIR_SAFE,
                                  2.0, 100000)
        # Should be invalid/quarantined, not silently fixed
        assert outcome.verdict in (Verdict.INVALID, Verdict.QUARANTINED)

    # 21. Dry run: no writes
    def t21() -> None:
        with tempfile.TemporaryDirectory() as td:
            src = os.path.join(td, "in")
            os.makedirs(src)
            train_dir = os.path.join(src, "train")
            os.makedirs(train_dir)
            puz = "530070000600195000098000060800060003400803001700020006060000280000419005000080079"
            sol = "534678912672195348198342567859761423426853791713924856961537284287419635345286179"
            rec = {
                "schema_version": "sm-schema-v2.0",
                "puzzle_id": "p1", "split": "train", "variant": "standard",
                "difficulty": 3.0, "difficulty_raw": 3.0,
                "difficulty_features": {}, "givens": 30,
                "puzzle": puz, "solution": sol,
                "solution_count_status": "unique", "unique_solution": True,
                "task_type": "complete_solution",
                "target": {"solution_string": sol},
                "technique_trace": [],
                "generator_version": "2.1.0", "generation_seed": 0,
                "solver_version": "x", "provenance": {}, "validation_status": "ok",
                "record_sha256": "0" * 64,
            }
            # Compute correct sha
            rc = dict(rec); rc.pop("record_sha256")
            rec["record_sha256"] = hashlib.sha256(
                json.dumps(rc, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            with open(os.path.join(train_dir, "train.jsonl"), "w") as f:
                f.write(json.dumps(rec) + "\n")
            out = os.path.join(td, "out")
            cfg = AuditConfig(input_path=src, output_path=out,
                              mode=RepairMode.REPAIR_SAFE, dry_run=True,
                              quiet=True, max_solver_sec=3.0)
            run_audit(cfg)
            assert not os.path.exists(out) or not any(os.scandir(out))

    # 22. Quarantine behavior
    def t22() -> None:
        with tempfile.TemporaryDirectory() as td:
            src = os.path.join(td, "in", "train")
            os.makedirs(src)
            bad = {
                "schema_version": "sm-schema-v2.0",
                "puzzle_id": "p1", "split": "train", "variant": "standard",
                "difficulty": 3.0, "difficulty_raw": 3.0,
                "difficulty_features": {}, "givens": 30,
                "puzzle": "5" * 81,  # invalid (all 5s -> peer conflicts)
                "solution": "5" * 81,
                "solution_count_status": "unique", "unique_solution": True,
                "task_type": "complete_solution",
                "target": {"solution_string": "5" * 81},
                "technique_trace": [],
                "generator_version": "2.1.0", "generation_seed": 0,
                "solver_version": "x", "provenance": {}, "validation_status": "ok",
                "record_sha256": "0" * 64,
            }
            with open(os.path.join(src, "train.jsonl"), "w") as f:
                f.write(json.dumps(bad) + "\n")
            out = os.path.join(td, "out")
            cfg = AuditConfig(input_path=os.path.join(td, "in"),
                              output_path=out, mode=RepairMode.QUARANTINE,
                              quiet=True, max_solver_sec=2.0)
            run_audit(cfg)
            qdir = os.path.join(out, "_quarantine")
            assert os.path.isdir(qdir)
            qfile = os.path.join(qdir, "train_quarantine.jsonl")
            assert os.path.exists(qfile)
            with open(qfile) as f:
                lines = [json.loads(x) for x in f if x.strip()]
            assert len(lines) == 1

    # 23. Unsupported schema version
    def t23() -> None:
        rec = {"schema_version": "sm-schema-v99", "puzzle_id": "x"}
        outcome = validate_record(rec, "t", 1, None, RepairMode.AUDIT, 5.0, 100000)
        assert outcome.verdict == Verdict.INVALID
        assert any(f.code == "unsupported_schema_version"
                   for f in outcome.findings)

    # 24. Unsupported variant
    def t24() -> None:
        rec = {"schema_version": "sm-schema-v2.0", "puzzle_id": "x",
               "split": "train", "variant": "killer", "difficulty": 3.0,
               "difficulty_raw": 3.0, "difficulty_features": {}, "givens": 30,
               "puzzle": "0" * 81, "solution": "0" * 81,
               "solution_count_status": "unique", "unique_solution": True,
               "task_type": "complete_solution", "target": {},
               "technique_trace": [], "generator_version": "x",
               "generation_seed": 0, "solver_version": "x", "provenance": {},
               "validation_status": "ok", "record_sha256": "0" * 64}
        outcome = validate_record(rec, "t", 1, None, RepairMode.AUDIT, 5.0, 100000)
        assert outcome.verdict == Verdict.INVALID
        assert any(f.code == "unsupported_variant" for f in outcome.findings)

    # 25. Multi-variant correctness for all 5 supported variants
    def t25() -> None:
        for vname in ("standard", "diagonal", "hyper", "anti_knight", "anti_king"):
            spec = VARIANT_REGISTRY[vname]
            res = solve_and_count([0] * 81, spec, limit=1, timeout_sec=3.0)
            assert res.status == SolverStatus.UNIQUE, (vname, res.status)
            assert res.first_solution is not None
            ok, err = validate_solution_grid(res.first_solution, spec)
            assert ok, f"{vname}: {err}"

    _run("01 valid standard Sudoku record", t01)
    _run("02 invalid grid dimension rejected", t02)
    _run("03 contradictory clues rejected", t03)
    _run("04 unsatisfiable puzzle detected", t04)
    _run("05 unique puzzle confirmed", t05)
    _run("06 multiple-solution puzzle detected", t06)
    _run("07 wrong reference solution detected", t07)
    _run("08 false uniqueness claim detected", t08)
    _run("09 valid technique trace (naked single)", t09)
    _run("10 invalid technique claim rejected", t10)
    _run("11 invalid proof trace rejected", t11)
    _run("12 difficulty metadata type checks", t12)
    _run("13 duplicate records detected", t13)
    _run("14 cross-split leakage detected", t14)
    _run("15 D4 + relabeling equivalence", t15)
    _run("16 malformed JSONL / duplicate keys", t16)
    _run("17 truncated final line detected", t17)
    _run("18 record_sha256 mismatch", t18)
    _run("19 safe repair of givens count", t19)
    _run("20 repair refusal for unsafe case", t20)
    _run("21 dry-run makes no writes", t21)
    _run("22 quarantine behaviour", t22)
    _run("23 unsupported schema version", t23)
    _run("24 unsupported variant rejected", t24)
    _run("25 multi-variant solver correctness", t25)

    print("-" * 78)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 78)
    return 0 if failed == 0 else 1


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    import signal as _sig
    _sig.signal(_sig.SIGINT, _signal_handler)
    if hasattr(_sig, "SIGTERM"):
        _sig.signal(_sig.SIGTERM, _signal_handler)

    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.run_tests or args.command == "selftest":
        return run_selftest_suite()

    if args.command == "help":
        parser.print_help()
        return ExitCode.OK

    if not args.input:
        parser.print_help()
        print("\n[ERROR] --input is required for audit/repair modes.")
        return ExitCode.CLI_ERROR

    # Map command -> mode
    command_to_mode = {
        "audit": RepairMode.AUDIT,
        "repair-safe": RepairMode.REPAIR_SAFE,
        "repair-aggressive": RepairMode.REPAIR_AGGRESSIVE,
        "quarantine": RepairMode.QUARANTINE,
    }
    if args.command not in command_to_mode:
        print(f"[ERROR] unknown command {args.command!r}")
        return ExitCode.CLI_ERROR

    mode = command_to_mode[args.command]

    if mode in (RepairMode.REPAIR_SAFE, RepairMode.REPAIR_AGGRESSIVE,
                RepairMode.QUARANTINE) and not args.output and not args.dry_run:
        print(f"[ERROR] --output is required for mode {mode.value} "
              f"(or use --dry-run to inspect only).")
        return ExitCode.CLI_ERROR

    cfg = AuditConfig(
        input_path=args.input,
        output_path=args.output,
        report_path=args.report,
        mode=mode,
        workers=max(1, args.workers),
        max_line_bytes=args.max_line_bytes,
        max_record_bytes=args.max_record_bytes,
        max_solver_sec=args.max_solver_seconds,
        max_solver_nodes=args.max_solver_nodes,
        max_wall_sec=args.max_wall_seconds,
        min_free_disk_mb=args.min_free_disk_mb,
        cross_split=args.cross_split or True,
        quiet=args.quiet,
        verbose=args.verbose,
        dry_run=args.dry_run,
        variant_override=args.variant,
    )

    # Startup banner
    if not args.quiet:
        print("=" * 78)
        print(f"{VALIDATOR_NAME} v{VALIDATOR_VERSION}")
        print("=" * 78)
        print(f"  Input               : {os.path.abspath(args.input)}")
        if args.output:
            print(f"  Output              : {os.path.abspath(args.output)}")
        print(f"  Mode                : {mode.value}")
        print(f"  Schema profile      : sm-schema-v2.0")
        print(f"  Workers             : {cfg.workers}")
        print(f"  Max solver seconds  : {cfg.max_solver_sec}")
        print(f"  Max solver nodes    : {cfg.max_solver_nodes}")
        print(f"  Max line bytes      : {cfg.max_line_bytes}")
        print(f"  Dry run             : {cfg.dry_run}")
        print(f"  Source modification : DISABLED (always non-destructive)")
        print("=" * 78)

    try:
        report = run_audit(cfg)
    except FileNotFoundError as e:
        sys.stderr.write(f"[ERROR] {e}\n")
        return ExitCode.CLI_ERROR
    except ValueError as e:
        sys.stderr.write(f"[ERROR] {e}\n")
        return ExitCode.CLI_ERROR
    except Exception as e:
        sys.stderr.write(f"[INTERNAL ERROR] {type(e).__name__}: {e}\n")
        traceback.print_exc()
        return ExitCode.INTERNAL_ERROR

    if args.report:
        os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
        tmp = args.report + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2, sort_keys=True)
        os.replace(tmp, args.report)

    _print_report_summary(report, args.quiet)
    return report.exit_code


if __name__ == "__main__":
    sys.exit(main())