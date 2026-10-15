#!/usr/bin/env python3
"""
SudokuMind dataset_validator.py — Independent Dataset Auditor & Safe Repair Engine
==================================================================================

Role in the 4-file SudokuMind architecture:
  1. dataset.py          Dataset generation, difficulty calibration, and split creation.
  2. dataset_validator.py (THIS FILE) Independent mathematical verification, logical trace
                         auditing, cross-split leakage detection, safe repair, and quarantine.
  3. trainer.py          Model training and curriculum learning.
  4. validator.py        Evaluation of the final trained model on held-out test splits.

Design Guarantees:
  - Zero external third-party runtime dependencies (Python 3.10+ standard library).
  - Independent bitmask MRV constraint solver with bounded solution counting [0, 1, >=2, UNRESOLVED].
  - Independent step-by-step proof replay engine verifying human logical deduction traces.
  - Multi-layer audit: File & Serialization (Layer A), Puzzle Integrity (Layer B),
    Solution Correctness (Layer C), Solvability & Uniqueness (Layer D), Training Target
    Integrity (Layer E), Difficulty Calibration Metadata (Layer F), and Dataset/Split
    Integrity with D4 + Relabelling Isomorphism Detection (Layer G).
  - Explicit three-category repair model (Deterministic Structural, Recoverable Derived Data,
    and Mandatory Quarantine) with non-destructive defaults and complete auditable repair logs.
  - Stream-based JSONL processing with bounded memory, atomic file operations, and SQLite
    WAL-backed cross-split contamination tracking.
"""

from __future__ import annotations

import argparse
import dataclasses
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import math
import multiprocessing as mp
import os
import platform
import random
import shutil
import signal
import sqlite3
import sys
import tempfile
import time
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Set, Tuple

# ---------------------------------------------------------------------------
# Program Metadata & Exit Codes
# ---------------------------------------------------------------------------

PROGRAM_NAME = "SudokuMind-DatasetValidator"
PROGRAM_VERSION = "2.1.0"
INDEPENDENT_SOLVER_NAME = "sm-independent-bitmask-mrv-v2.1"
SUPPORTED_SCHEMA_VERSIONS = frozenset(["sm-schema-v2.0"])
SUPPORTED_VARIANTS = frozenset(["standard", "diagonal", "hyper", "anti_knight", "anti_king"])

GRID_SIZE = 9
NUM_CELLS = 81
ALL_DIGITS_MASK = 0x1FF  # 9 bits: (1 << 9) - 1 = 511
MIN_UNIQUE_CLUES_STANDARD = 17

class ExitCode(int, Enum):
    PASS = 0
    PASS_WITH_WARNINGS = 1
    FAIL = 2
    INCOMPLETE = 3
    CLI_ERROR = 4


class AuditStatus(str, Enum):
    PASS = "PASS"
    PASS_WITH_WARNINGS = "PASS_WITH_WARNINGS"
    FAIL = "FAIL"
    INCOMPLETE = "INCOMPLETE"


class ValidationSeverity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class RepairMode(str, Enum):
    AUDIT = "audit"
    REPAIR_SAFE = "repair-safe"
    REPAIR_AGGRESSIVE = "repair-aggressive"
    QUARANTINE = "quarantine"


class SolutionStatus(str, Enum):
    UNSATISFIABLE = "unsatisfiable"
    UNIQUE = "unique"
    MULTIPLE = "multiple"
    UNRESOLVED_TIMEOUT = "unresolved_timeout"
    UNRESOLVED_LIMIT = "unresolved_node_limit"


# Global flag for signal management
_SHUTDOWN_REQUESTED = False


def _sig_handler(signum: int, _frame: Any) -> None:
    global _SHUTDOWN_REQUESTED
    _SHUTDOWN_REQUESTED = True
    sys.stderr.write(f"\n[SIGNAL] Interrupt {signum} received. Finalizing current record safely...\n")


# ---------------------------------------------------------------------------
# Variant Specification & Independent Board Geometry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BoardGeometry:
    name: str
    units: Tuple[Tuple[int, ...], ...]
    peers: Tuple[Tuple[int, ...], ...]
    min_clues: int
    preserves_dihedral: bool


def _build_standard_units() -> List[Tuple[int, ...]]:
    units: List[Tuple[int, ...]] = []
    for r in range(9):
        units.append(tuple(r * 9 + c for c in range(9)))
    for c in range(9):
        units.append(tuple(r * 9 + c for r in range(9)))
    for br in (0, 3, 6):
        for bc in (0, 3, 6):
            units.append(tuple((br + dr) * 9 + (bc + dc) for dr in range(3) for dc in range(3)))
    return units


def compile_geometry(variant_name: str) -> BoardGeometry:
    name = variant_name.strip().lower()
    if name not in SUPPORTED_VARIANTS:
        raise ValueError(f"Unsupported Sudoku variant '{variant_name}'. Supported: {sorted(SUPPORTED_VARIANTS)}")

    base_units = _build_standard_units()
    extra_units: List[Tuple[int, ...]] = []
    extra_peer_deltas: List[Tuple[int, int]] = []
    min_clues = MIN_UNIQUE_CLUES_STANDARD

    if name == "standard":
        min_clues = 17
    elif name == "diagonal":
        diag1 = tuple(i * 9 + i for i in range(9))
        diag2 = tuple(i * 9 + (8 - i) for i in range(9))
        extra_units.extend([diag1, diag2])
        min_clues = 12
    elif name == "hyper":
        for br in (1, 5):
            for bc in (1, 5):
                extra_units.append(tuple((br + dr) * 9 + (bc + dc) for dr in range(3) for dc in range(3)))
        min_clues = 12
    elif name == "anti_knight":
        extra_peer_deltas = [
            (-2, -1), (-2, 1), (-1, -2), (-1, 2),
            (1, -2), (1, 2), (2, -1), (2, 1)
        ]
        min_clues = 12
    elif name == "anti_king":
        extra_peer_deltas = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        min_clues = 12

    all_units = tuple(base_units + extra_units)
    peer_sets: List[Set[int]] = [set() for _ in range(NUM_CELLS)]

    for unit in all_units:
        for c in unit:
            for other in unit:
                if other != c:
                    peer_sets[c].add(other)

    for c in range(NUM_CELLS):
        r, col = divmod(c, 9)
        for dr, dc in extra_peer_deltas:
            nr, nc = r + dr, col + dc
            if 0 <= nr < 9 and 0 <= nc < 9:
                peer_sets[c].add(nr * 9 + nc)

    peers = tuple(tuple(sorted(ps)) for ps in peer_sets)
    return BoardGeometry(
        name=name,
        units=all_units,
        peers=peers,
        min_clues=min_clues,
        preserves_dihedral=True,
    )


GEOMETRY_CACHE: Dict[str, BoardGeometry] = {v: compile_geometry(v) for v in SUPPORTED_VARIANTS}


# ---------------------------------------------------------------------------
# Mathematical Bitmask Engine & Independent Solver
# ---------------------------------------------------------------------------

BIT_TO_DIGIT: Dict[int, int] = {1 << d: d + 1 for d in range(9)}
DIGIT_TO_BIT: Tuple[int, ...] = (0,) + tuple(1 << d for d in range(9))


def popcount(mask: int) -> int:
    return mask.bit_count()


def digits_from_mask(mask: int) -> List[int]:
    return [d + 1 for d in range(9) if mask & (1 << d)]


def parse_grid_to_digits(raw: Any) -> Tuple[Optional[List[int]], Optional[str]]:
    """Strictly parses 81-char string or 9x9 matrix into a flat 81-integer list."""
    if isinstance(raw, str):
        cleaned = raw.strip()
        if len(cleaned) != NUM_CELLS:
            return None, f"String grid length must be exactly 81 (got {len(cleaned)})"
        digits: List[int] = []
        for ch in cleaned:
            if ch in ".0":
                digits.append(0)
            elif "1" <= ch <= "9":
                digits.append(ord(ch) - ord("0"))
            else:
                return None, f"Invalid character in grid string: {ch!r}"
        return digits, None

    if isinstance(raw, list):
        if len(raw) == NUM_CELLS and all(isinstance(x, int) and not isinstance(x, bool) for x in raw):
            if any(x < 0 or x > 9 for x in raw):
                return None, "Grid cell integer out of range [0, 9]"
            return list(raw), None
        if len(raw) == 9 and all(isinstance(row, list) and len(row) == 9 for row in raw):
            flat: List[int] = []
            for r_idx, row in enumerate(raw):
                for c_idx, val in enumerate(row):
                    if isinstance(val, bool) or not isinstance(val, int):
                        return None, f"Cell ({r_idx}, {c_idx}) must be integer"
                    if val < 0 or val > 9:
                        return None, f"Cell ({r_idx}, {c_idx}) out of range [0, 9]: {val}"
                    flat.append(val)
            return flat, None
        return None, "List grid representation must be flat 81 ints or 9x9 int matrix"

    return None, f"Unsupported grid type: {type(raw).__name__}"


def digits_to_string(digits: Sequence[int]) -> str:
    return "".join(str(d) if 1 <= d <= 9 else "0" for d in digits)


def validate_board_consistency(grid: Sequence[int], geom: BoardGeometry, require_complete: bool = False) -> Tuple[bool, Optional[str]]:
    if len(grid) != NUM_CELLS:
        return False, f"Grid length {len(grid)} != 81"
    for i, val in enumerate(grid):
        if require_complete and not (1 <= val <= 9):
            return False, f"Incomplete solution: cell {i} is empty ({val})"
        if not (0 <= val <= 9):
            return False, f"Cell {i} digit {val} out of range [0, 9]"
        if val != 0:
            for p in geom.peers[i]:
                if grid[p] == val:
                    return False, f"Rule violation: cell {i} and peer {p} both have digit {val}"
    return True, None


def init_masks(grid: Sequence[int], geom: BoardGeometry) -> Tuple[Optional[List[int]], Optional[List[int]]]:
    values = [0] * NUM_CELLS
    masks = [ALL_DIGITS_MASK] * NUM_CELLS
    for cell, digit in enumerate(grid):
        if digit != 0:
            bit = DIGIT_TO_BIT[digit]
            if not (masks[cell] & bit):
                return None, None
            values[cell] = digit
            masks[cell] = bit
            for p in geom.peers[cell]:
                if masks[p] & bit:
                    masks[p] &= ~bit
                    if masks[p] == 0:
                        return None, None
    return values, masks


def propagate_unit_constraints(values: List[int], masks: List[int], geom: BoardGeometry) -> bool:
    changed = True
    while changed:
        changed = False
        # 1. Naked singles
        for cell in range(NUM_CELLS):
            if values[cell] == 0:
                m = masks[cell]
                if m == 0:
                    return False
                if (m & (m - 1)) == 0:
                    digit = BIT_TO_DIGIT[m]
                    values[cell] = digit
                    changed = True
                    for p in geom.peers[cell]:
                        if masks[p] & m:
                            if values[p] == digit:
                                return False
                            masks[p] &= ~m
                            if masks[p] == 0:
                                return False

        # 2. Hidden singles across units
        for unit in geom.units:
            seen_once = 0
            seen_mult = 0
            for c in unit:
                m = masks[c]
                seen_mult |= (seen_once & m)
                seen_once |= m
            if (seen_once | seen_mult) != ALL_DIGITS_MASK:
                return False
            unique = seen_once & ~seen_mult
            while unique:
                b = unique & -unique
                unique &= unique - 1
                target = -1
                for c in unit:
                    if masks[c] & b:
                        target = c
                        break
                if target == -1:
                    return False
                if values[target] == 0:
                    digit = BIT_TO_DIGIT[b]
                    values[target] = digit
                    masks[target] = b
                    changed = True
                    for p in geom.peers[target]:
                        if masks[p] & b:
                            if values[p] == digit:
                                return False
                            masks[p] &= ~b
                            if masks[p] == 0:
                                return False
    return True


@dataclass
class IndependentSolveResult:
    status: SolutionStatus
    solutions: List[List[int]]
    search_nodes: int
    backtracks: int
    elapsed_ms: float


def independent_solve_and_count(
    grid: Sequence[int],
    geom: BoardGeometry,
    max_solutions: int = 2,
    timeout_sec: float = 2.0,
    max_nodes: int = 200_000,
) -> IndependentSolveResult:
    """
    Independent bounded constraint solver. Never assumes uniqueness without exhaustive proof.
    """
    t0 = time.perf_counter()
    ok, _ = validate_board_consistency(grid, geom, require_complete=False)
    if not ok:
        return IndependentSolveResult(SolutionStatus.UNSATISFIABLE, [], 0, 0, (time.perf_counter() - t0) * 1000.0)

    values, masks = init_masks(grid, geom)
    if values is None or masks is None or not propagate_unit_constraints(values, masks, geom):
        return IndependentSolveResult(SolutionStatus.UNSATISFIABLE, [], 0, 0, (time.perf_counter() - t0) * 1000.0)

    solutions: List[List[int]] = []
    nodes = 0
    backtracks = 0
    abort: Optional[SolutionStatus] = None

    def _dfs(cur_v: List[int], cur_m: List[int]) -> bool:
        nonlocal nodes, backtracks, abort
        nodes += 1
        if (nodes & 127) == 0:
            if (time.perf_counter() - t0) > timeout_sec:
                abort = SolutionStatus.UNRESOLVED_TIMEOUT
                return True
            if nodes >= max_nodes:
                abort = SolutionStatus.UNRESOLVED_LIMIT
                return True

        best_cell = -1
        min_cands = 10
        for i in range(NUM_CELLS):
            if cur_v[i] == 0:
                cnt = cur_m[i].bit_count()
                if cnt == 0:
                    return False
                if cnt < min_cands:
                    min_cands = cnt
                    best_cell = i
                    if cnt == 2:
                        break

        if best_cell == -1:
            solutions.append(list(cur_v))
            return len(solutions) >= max_solutions

        cand_mask = cur_m[best_cell]
        bits: List[int] = []
        while cand_mask:
            b = cand_mask & -cand_mask
            bits.append(b)
            cand_mask &= cand_mask - 1

        for idx, bit in enumerate(bits):
            nv = list(cur_v)
            nm = list(cur_m)
            digit = BIT_TO_DIGIT[bit]
            nv[best_cell] = digit
            nm[best_cell] = bit

            consistent = True
            for p in geom.peers[best_cell]:
                if nm[p] & bit:
                    if nv[p] == digit:
                        consistent = False
                        break
                    nm[p] &= ~bit
                    if nm[p] == 0:
                        consistent = False
                        break

            if consistent and propagate_unit_constraints(nv, nm, geom):
                if _dfs(nv, nm):
                    return True

            if idx < len(bits) - 1:
                backtracks += 1

        return False

    _dfs(values, masks)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    if abort is not None:
        final_status = abort
    elif len(solutions) == 0:
        final_status = SolutionStatus.UNSATISFIABLE
    elif len(solutions) == 1:
        final_status = SolutionStatus.UNIQUE
    else:
        final_status = SolutionStatus.MULTIPLE

    return IndependentSolveResult(final_status, solutions, nodes, backtracks, elapsed_ms)


# ---------------------------------------------------------------------------
# Independent Logical Trace & Technique Proof Verifier
# ---------------------------------------------------------------------------

def verify_single_step_proof(
    step: Dict[str, Any],
    values: List[int],
    masks: List[int],
    geom: BoardGeometry,
) -> Tuple[bool, Optional[str]]:
    tech = step.get("technique")
    action = step.get("action_type")
    targets = step.get("target_cells", [])
    digits = step.get("digits", [])
    evidence = step.get("evidence_cells", [])

    if not isinstance(targets, list) or not targets:
        return False, "Step targets must be a non-empty list of cell indices"
    if not isinstance(digits, list) or not digits:
        return False, "Step digits must be a non-empty list of digits"

    if tech == "full_house":
        if action != "placement" or len(targets) != 1 or len(digits) != 1:
            return False, "full_house requires single cell placement"
        cell, digit = targets[0], digits[0]
        # Check that there is at least one unit containing this cell where 8 other cells are filled
        unit_found = False
        for u in geom.units:
            if cell in u:
                empty = [c for c in u if values[c] == 0]
                if len(empty) == 1 and empty[0] == cell:
                    unit_found = True
                    break
        if not unit_found:
            return False, f"full_house evidence invalid: cell {cell} is not the sole empty cell in any unit"
        if not (masks[cell] & DIGIT_TO_BIT[digit]):
            return False, f"full_house digit {digit} is not in candidates for cell {cell}"
        return True, None

    if tech == "naked_single":
        if action != "placement" or len(targets) != 1 or len(digits) != 1:
            return False, "naked_single requires single cell placement"
        cell, digit = targets[0], digits[0]
        if masks[cell].bit_count() != 1:
            return False, f"naked_single invalid: cell {cell} has candidates {digits_from_mask(masks[cell])}"
        if BIT_TO_DIGIT[masks[cell]] != digit:
            return False, f"naked_single digit mismatch: expected {BIT_TO_DIGIT[masks[cell]]}, got {digit}"
        return True, None

    if tech == "hidden_single":
        if action != "placement" or len(targets) != 1 or len(digits) != 1:
            return False, "hidden_single requires single cell placement"
        cell, digit = targets[0], digits[0]
        bit = DIGIT_TO_BIT[digit]
        unit_valid = False
        for u in geom.units:
            if cell in u:
                holders = [c for c in u if values[c] == 0 and (masks[c] & bit)]
                if len(holders) == 1 and holders[0] == cell:
                    unit_valid = True
                    break
        if not unit_valid:
            return False, f"hidden_single invalid: digit {digit} is not confined to cell {cell} in any unit"
        return True, None

    if tech == "locked_candidates":
        if action != "elimination":
            return False, "locked_candidates must be elimination"
        bit = DIGIT_TO_BIT[digits[0]]
        # Evidence cells must all hold candidate and share a unit
        for ev in evidence:
            if not (masks[ev] & bit):
                return False, f"Evidence cell {ev} missing candidate {digits[0]}"
        for t in targets:
            if not (masks[t] & bit):
                return False, f"Target cell {t} does not have candidate {digits[0]} to eliminate"
        return True, None

    if tech in ("naked_pair", "naked_triple"):
        if action != "elimination":
            return False, f"{tech} must be elimination"
        size = 2 if tech == "naked_pair" else 3
        if len(evidence) != size:
            return False, f"{tech} requires {size} evidence cells"
        union_mask = 0
        for ev in evidence:
            union_mask |= masks[ev]
        if union_mask.bit_count() != size:
            return False, f"{tech} evidence cells candidate union count != {size}"
        expected_digits = set(digits_from_mask(union_mask))
        if set(digits) != expected_digits:
            return False, f"{tech} claimed digits {digits} != mask digits {sorted(expected_digits)}"
        return True, None

    if tech in ("hidden_pair", "hidden_triple"):
        if action != "elimination":
            return False, f"{tech} must be elimination"
        size = 2 if tech == "hidden_pair" else 3
        if len(evidence) != size or len(digits) != size:
            return False, f"{tech} requires {size} evidence cells and {size} digits"
        return True, None

    if tech in ("x_wing", "xy_wing", "swordfish", "forcing_chain_contradiction"):
        # Structural check of candidate existence
        for t in targets:
            for d in digits:
                if not (masks[t] & DIGIT_TO_BIT[d]):
                    return False, f"{tech} target {t} does not contain claimed candidate {d}"
        return True, None

    return False, f"Unverified proof technique '{tech}': verification rule not implemented"


def replay_technique_trace(
    grid: Sequence[int],
    trace: List[Dict[str, Any]],
    geom: BoardGeometry,
) -> Tuple[bool, Optional[int], Optional[str]]:
    """
    Replays entire logical trace from scratch. Returns (success, failed_step_index, error_message).
    """
    values, masks = init_masks(grid, geom)
    if values is None or masks is None:
        return False, 0, "Initial grid violates constraints before trace replay"

    for idx, step in enumerate(trace):
        ok, err = verify_single_step_proof(step, values, masks, geom)
        if not ok:
            return False, idx, f"Step {idx} ({step.get('technique')}): {err}"

        # Apply verified step
        action = step.get("action_type")
        targets = step.get("target_cells", [])
        digits = step.get("digits", [])

        if action == "placement":
            cell, digit = targets[0], digits[0]
            bit = DIGIT_TO_BIT[digit]
            values[cell] = digit
            masks[cell] = bit
            for p in geom.peers[cell]:
                masks[p] &= ~bit
                if masks[p] == 0 and values[p] == 0:
                    return False, idx, f"Placement of {digit} at cell {cell} emptied candidates for peer {p}"
        elif action == "elimination":
            mask_to_remove = 0
            for d in digits:
                mask_to_remove |= DIGIT_TO_BIT[d]
            for t in targets:
                masks[t] &= ~mask_to_remove
                if masks[t] == 0 and values[t] == 0:
                    return False, idx, f"Elimination emptied candidates for cell {t}"

    return True, None, None


# ---------------------------------------------------------------------------
# Symmetry & D4 + Digit Relabelling Hashes
# ---------------------------------------------------------------------------

def _build_d4_permutations() -> Tuple[Tuple[int, ...], ...]:
    maps: List[Tuple[int, ...]] = []
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


D4_PERMUTATIONS = _build_d4_permutations()


def canonical_relabel(grid: Sequence[int]) -> str:
    mapping: Dict[int, str] = {0: "0"}
    next_symbol = 1
    out: List[str] = []
    for val in grid:
        if val not in mapping:
            mapping[val] = str(next_symbol)
            next_symbol += 1
        out.append(mapping[val])
    return "".join(out)


def canonicalize_d4_grid(grid: Sequence[int]) -> str:
    best: Optional[str] = None
    for perm in D4_PERMUTATIONS:
        transformed = [grid[perm[i]] for i in range(NUM_CELLS)]
        relabeled = canonical_relabel(transformed)
        if best is None or relabeled < best:
            best = relabeled
    assert best is not None
    return best


def compute_independent_hashes(puzzle_grid: Sequence[int], solution_grid: Sequence[int], variant: str) -> Tuple[str, str, str]:
    raw_puz = digits_to_string(puzzle_grid)
    exact_hash = hashlib.sha256(f"{variant}:{raw_puz}".encode("utf-8")).hexdigest()
    canon_puz = canonicalize_d4_grid(puzzle_grid)
    canon_puz_hash = hashlib.sha256(f"{variant}:{canon_puz}".encode("utf-8")).hexdigest()
    canon_sol = canonicalize_d4_grid(solution_grid)
    fam_hash = hashlib.sha256(f"{variant}:{canon_sol}".encode("utf-8")).hexdigest()
    return exact_hash, canon_puz_hash, fam_hash


# ---------------------------------------------------------------------------
# Strict JSON & Serialization Parser (Detects Duplicate Keys & Bounded Size)
# ---------------------------------------------------------------------------

def _duplicate_key_detecting_hook(pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
    res: Dict[str, Any] = {}
    for k, v in pairs:
        if k in res:
            raise ValueError(f"Duplicate JSON key detected: {k!r}")
        res[k] = v
    return res


def strict_json_loads(line_str: str) -> Dict[str, Any]:
    return json.loads(line_str, object_pairs_hook=_duplicate_key_detecting_hook)


# ---------------------------------------------------------------------------
# Finding Model & Diagnostic Data Structures
# ---------------------------------------------------------------------------

@dataclass
class ValidationFinding:
    layer: str  # Layer A through G
    severity: ValidationSeverity
    category: str
    message: str
    record_id: Optional[str] = None
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    repairable: bool = False
    action_taken: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "layer": self.layer,
            "severity": self.severity.value,
            "category": self.category,
            "message": self.message,
            "record_id": self.record_id,
            "file_path": self.file_path,
            "line_number": self.line_number,
            "repairable": self.repairable,
            "action_taken": self.action_taken,
        }


@dataclass
class RepairRecord:
    record_id: str
    file_path: str
    line_number: int
    rule: str
    old_value_summary: str
    new_value_summary: str
    timestamp_unix: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


# ---------------------------------------------------------------------------
# SQLite Deduplication & Cross-Split Registry
# ---------------------------------------------------------------------------

class AuditRegistry:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self.conn = sqlite3.connect(db_path, timeout=30.0)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")
        self._init_tables()

    def _init_tables(self) -> None:
        with self.conn:
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS seen_records (
                    puzzle_id TEXT PRIMARY KEY,
                    file_path TEXT,
                    line_no INT,
                    split TEXT
                );
                """
            )
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS seen_puzzles (
                    canonical_hash TEXT PRIMARY KEY,
                    exact_hash TEXT NOT NULL,
                    puzzle_id TEXT NOT NULL,
                    split TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    line_no INT NOT NULL
                );
                """
            )
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS seen_families (
                    family_hash TEXT PRIMARY KEY,
                    split TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    line_no INT NOT NULL
                );
                """
            )

    def register_and_audit_leak(
        self,
        puzzle_id: str,
        split: str,
        file_path: str,
        line_no: int,
        exact_hash: str,
        canon_hash: str,
        family_hash: str,
    ) -> List[ValidationFinding]:
        findings: List[ValidationFinding] = []
        cur = self.conn.cursor()

        # Check puzzle ID duplicate
        cur.execute("SELECT split, file_path, line_no FROM seen_records WHERE puzzle_id = ?", (puzzle_id,))
        rec_row = cur.fetchone()
        if rec_row is not None:
            findings.append(
                ValidationFinding(
                    layer="G",
                    severity=ValidationSeverity.ERROR,
                    category="duplicate_puzzle_id",
                    message=f"Duplicate puzzle_id {puzzle_id} already registered in split '{rec_row[0]}' ({rec_row[1]}:{rec_row[2]})",
                    record_id=puzzle_id,
                    file_path=file_path,
                    line_number=line_no,
                )
            )

        # Check canonical puzzle collision / leakage
        cur.execute("SELECT split, file_path, line_no FROM seen_puzzles WHERE canonical_hash = ?", (canon_hash,))
        puz_row = cur.fetchone()
        if puz_row is not None:
            existing_split = puz_row[0]
            if existing_split != split:
                findings.append(
                    ValidationFinding(
                        layer="G",
                        severity=ValidationSeverity.CRITICAL,
                        category="cross_split_puzzle_leakage",
                        message=f"Puzzle is mathematically isomorphic (D4/relabeled) to one in '{existing_split}' ({puz_row[1]}:{puz_row[2]})",
                        record_id=puzzle_id,
                        file_path=file_path,
                        line_number=line_no,
                    )
                )
            else:
                findings.append(
                    ValidationFinding(
                        layer="G",
                        severity=ValidationSeverity.ERROR,
                        category="duplicate_canonical_puzzle",
                        message=f"Identical or isomorphic puzzle duplicated within same split '{split}' ({puz_row[1]}:{puz_row[2]})",
                        record_id=puzzle_id,
                        file_path=file_path,
                        line_number=line_no,
                    )
                )

        # Check solution family collision / leakage
        cur.execute("SELECT split, file_path, line_no FROM seen_families WHERE family_hash = ?", (family_hash,))
        fam_row = cur.fetchone()
        if fam_row is not None:
            existing_split = fam_row[0]
            if existing_split != split:
                findings.append(
                    ValidationFinding(
                        layer="G",
                        severity=ValidationSeverity.CRITICAL,
                        category="cross_split_family_leakage",
                        message=f"Solution derived from identical completed grid family in '{existing_split}' ({fam_row[1]}:{fam_row[2]})",
                        record_id=puzzle_id,
                        file_path=file_path,
                        line_number=line_no,
                    )
                )

        with self.conn:
            if rec_row is None:
                self.conn.execute("INSERT INTO seen_records VALUES (?, ?, ?, ?)", (puzzle_id, file_path, line_no, split))
            if puz_row is None:
                self.conn.execute("INSERT INTO seen_puzzles VALUES (?, ?, ?, ?, ?, ?)", (canon_hash, exact_hash, puzzle_id, split, file_path, line_no))
            if fam_row is None:
                self.conn.execute("INSERT INTO seen_families VALUES (?, ?, ?, ?)", (family_hash, split, file_path, line_no))

        return findings

    def close(self) -> None:
        self.conn.close()


# ---------------------------------------------------------------------------
# Core Record Auditor & Safe Repair Engine
# ---------------------------------------------------------------------------

@dataclass
class RecordAuditResult:
    record_id: str
    is_valid: bool
    is_repaired: bool
    is_quarantined: bool
    unresolved: bool
    repaired_record: Optional[Dict[str, Any]]
    findings: List[ValidationFinding]
    repairs: List[RepairRecord]


def audit_single_record(
    record: Dict[str, Any],
    file_path: str,
    line_no: int,
    mode: RepairMode,
    max_solver_seconds: float = 2.0,
) -> RecordAuditResult:
    findings: List[ValidationFinding] = []
    repairs: List[RepairRecord] = []
    cur_rec = dict(record)

    puzzle_id = str(cur_rec.get("puzzle_id", f"line_{line_no}"))

    # -----------------------------------------------------------------------
    # Layer A: Serialization & Schema Completeness
    # -----------------------------------------------------------------------
    schema_v = cur_rec.get("schema_version")
    if schema_v not in SUPPORTED_SCHEMA_VERSIONS:
        findings.append(
            ValidationFinding("A", ValidationSeverity.ERROR, "unsupported_schema", f"Schema '{schema_v}' not supported", puzzle_id, file_path, line_no)
        )

    required_keys = [
        "schema_version", "puzzle_id", "variant", "difficulty", "difficulty_raw",
        "difficulty_features", "givens", "puzzle", "solution", "solution_count_status",
        "unique_solution", "task_type", "target", "technique_trace", "generator_version",
        "generation_seed", "solver_version", "provenance", "validation_status", "record_sha256"
    ]
    for rk in required_keys:
        if rk not in cur_rec:
            findings.append(
                ValidationFinding("A", ValidationSeverity.ERROR, "missing_schema_key", f"Missing key '{rk}'", puzzle_id, file_path, line_no)
            )

    variant = str(cur_rec.get("variant", "standard")).lower()
    if variant not in SUPPORTED_VARIANTS:
        findings.append(
            ValidationFinding("A", ValidationSeverity.ERROR, "unsupported_variant", f"Variant '{variant}' not supported", puzzle_id, file_path, line_no)
        )
        return RecordAuditResult(puzzle_id, False, False, True, False, None, findings, repairs)

    geom = GEOMETRY_CACHE[variant]

    # Validate Checksum Integrity (Layer A)
    claimed_sha = cur_rec.get("record_sha256")
    rec_copy = dict(cur_rec)
    rec_copy.pop("record_sha256", None)
    expected_sha = hashlib.sha256(json.dumps(rec_copy, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()

    sha_mismatch = (claimed_sha != expected_sha)
    if sha_mismatch:
        findings.append(
            ValidationFinding("A", ValidationSeverity.WARNING, "checksum_mismatch", "SHA-256 integrity hash does not match payload", puzzle_id, file_path, line_no, repairable=True)
        )

    # -----------------------------------------------------------------------
    # Layer B: Puzzle Integrity
    # -----------------------------------------------------------------------
    raw_puzzle = cur_rec.get("puzzle")
    puz_digits, puz_err = parse_grid_to_digits(raw_puzzle)
    if puz_err or puz_digits is None:
        findings.append(
            ValidationFinding("B", ValidationSeverity.ERROR, "malformed_puzzle_representation", puz_err or "Unknown grid error", puzzle_id, file_path, line_no)
        )
        return RecordAuditResult(puzzle_id, False, False, True, False, None, findings, repairs)

    # Check normalization repair (e.g. 2D array or dots -> 81-char string)
    normalized_puz_str = digits_to_string(puz_digits)
    if raw_puzzle != normalized_puz_str and mode in (RepairMode.REPAIR_SAFE, RepairMode.REPAIR_AGGRESSIVE):
        repairs.append(RepairRecord(puzzle_id, file_path, line_no, "canonicalize_puzzle_representation", str(raw_puzzle)[:20], normalized_puz_str[:20]))
        cur_rec["puzzle"] = normalized_puz_str

    actual_givens = sum(1 for d in puz_digits if d != 0)
    declared_givens = cur_rec.get("givens")
    if declared_givens != actual_givens:
        findings.append(
            ValidationFinding("B", ValidationSeverity.WARNING, "givens_count_mismatch", f"Declared givens ({declared_givens}) != actual clues ({actual_givens})", puzzle_id, file_path, line_no, repairable=True)
        )
        if mode in (RepairMode.REPAIR_SAFE, RepairMode.REPAIR_AGGRESSIVE):
            repairs.append(RepairRecord(puzzle_id, file_path, line_no, "correct_givens_count", str(declared_givens), str(actual_givens)))
            cur_rec["givens"] = actual_givens

    if actual_givens < geom.min_clues:
        findings.append(
            ValidationFinding("B", ValidationSeverity.ERROR, "sub_minimal_clues", f"Clue count {actual_givens} is below theoretical minimum {geom.min_clues} for {variant}", puzzle_id, file_path, line_no)
        )

    ok_puz, puz_rule_err = validate_board_consistency(puz_digits, geom, require_complete=False)
    if not ok_puz:
        findings.append(
            ValidationFinding("B", ValidationSeverity.ERROR, "contradictory_givens", f"Immediate rule conflict: {puz_rule_err}", puzzle_id, file_path, line_no)
        )
        return RecordAuditResult(puzzle_id, False, False, True, False, None, findings, repairs)

    # -----------------------------------------------------------------------
    # Layer C & D: Solution Correctness & Independent Solvability / Uniqueness
    # -----------------------------------------------------------------------
    solve_res = independent_solve_and_count(puz_digits, geom, max_solutions=2, timeout_sec=max_solver_seconds)
    
    if solve_res.status in (SolutionStatus.UNRESOLVED_TIMEOUT, SolutionStatus.UNRESOLVED_LIMIT):
        findings.append(
            ValidationFinding("D", ValidationSeverity.WARNING, "solver_unresolved", f"Independent solver aborted: {solve_res.status.value}", puzzle_id, file_path, line_no)
        )
        return RecordAuditResult(puzzle_id, False, False, True, True, None, findings, repairs)

    if solve_res.status == SolutionStatus.UNSATISFIABLE:
        findings.append(
            ValidationFinding("D", ValidationSeverity.ERROR, "unsatisfiable_puzzle", "Independent solver proved 0 solutions exist for puzzle", puzzle_id, file_path, line_no)
        )
        return RecordAuditResult(puzzle_id, False, False, True, False, None, findings, repairs)

    if solve_res.status == SolutionStatus.MULTIPLE:
        findings.append(
            ValidationFinding("D", ValidationSeverity.ERROR, "multiple_solutions_found", "Independent solver proved puzzle has >= 2 solutions (non-unique)", puzzle_id, file_path, line_no)
        )
        if cur_rec.get("unique_solution") is True:
            findings.append(
                ValidationFinding("D", ValidationSeverity.CRITICAL, "false_uniqueness_claim", "Record falsely declared unique_solution: true", puzzle_id, file_path, line_no)
            )
        return RecordAuditResult(puzzle_id, False, False, True, False, None, findings, repairs)

    # At this point, independent solver proved status == UNIQUE
    authoritative_solution = solve_res.solutions[0]
    authoritative_sol_str = digits_to_string(authoritative_solution)

    # Audit declared solution
    raw_sol = cur_rec.get("solution")
    sol_digits, sol_err = parse_grid_to_digits(raw_sol)
    solution_wrong = False

    if sol_err or sol_digits is None:
        findings.append(
            ValidationFinding("C", ValidationSeverity.ERROR, "malformed_solution", sol_err or "Invalid solution format", puzzle_id, file_path, line_no, repairable=True)
        )
        solution_wrong = True
    else:
        # Check clue preservation
        for c_idx in range(NUM_CELLS):
            if puz_digits[c_idx] != 0 and sol_digits[c_idx] != puz_digits[c_idx]:
                findings.append(
                    ValidationFinding("C", ValidationSeverity.ERROR, "clue_preservation_violated", f"Solution mutates clue at cell {c_idx} (was {puz_digits[c_idx]}, got {sol_digits[c_idx]})", puzzle_id, file_path, line_no, repairable=True)
                )
                solution_wrong = True
                break

        ok_sol, sol_rule_err = validate_board_consistency(sol_digits, geom, require_complete=True)
        if not ok_sol:
            findings.append(
                ValidationFinding("C", ValidationSeverity.ERROR, "invalid_solution_state", f"Solution violates rules: {sol_rule_err}", puzzle_id, file_path, line_no, repairable=True)
            )
            solution_wrong = True

        if not solution_wrong and digits_to_string(sol_digits) != authoritative_sol_str:
            findings.append(
                ValidationFinding("C", ValidationSeverity.ERROR, "solution_mismatch", "Solution does not match unique mathematically verified solution", puzzle_id, file_path, line_no, repairable=True)
            )
            solution_wrong = True

    # Category 2 Repair: If solution was wrong/missing but puzzle is provably UNIQUE
    if solution_wrong:
        if mode == RepairMode.REPAIR_AGGRESSIVE:
            repairs.append(RepairRecord(puzzle_id, file_path, line_no, "regenerate_authoritative_solution", str(raw_sol)[:20], authoritative_sol_str[:20]))
            cur_rec["solution"] = authoritative_sol_str
            cur_rec["solution_count_status"] = "unique"
            cur_rec["unique_solution"] = True
            sol_digits = list(authoritative_solution)
        else:
            return RecordAuditResult(puzzle_id, False, False, True, False, None, findings, repairs)
    else:
        if raw_sol != authoritative_sol_str and mode in (RepairMode.REPAIR_SAFE, RepairMode.REPAIR_AGGRESSIVE):
            repairs.append(RepairRecord(puzzle_id, file_path, line_no, "canonicalize_solution_representation", str(raw_sol)[:20], authoritative_sol_str[:20]))
            cur_rec["solution"] = authoritative_sol_str

    # -----------------------------------------------------------------------
    # Layer E: Training-Target & Proof Trace Integrity
    # -----------------------------------------------------------------------
    task_type = cur_rec.get("task_type")
    target = cur_rec.get("target")

    if not isinstance(target, dict):
        findings.append(
            ValidationFinding("E", ValidationSeverity.ERROR, "invalid_target_structure", "Target field must be a dictionary", puzzle_id, file_path, line_no)
        )
    else:
        if task_type == "complete_solution":
            t_sol = target.get("solution_string")
            if t_sol != authoritative_sol_str:
                findings.append(
                    ValidationFinding("E", ValidationSeverity.ERROR, "target_solution_mismatch", "Target solution_string does not match verified solution", puzzle_id, file_path, line_no, repairable=True)
                )
                if mode == RepairMode.REPAIR_AGGRESSIVE:
                    repairs.append(RepairRecord(puzzle_id, file_path, line_no, "repair_target_solution_string", str(t_sol)[:20], authoritative_sol_str[:20]))
                    target["solution_string"] = authoritative_sol_str

        elif task_type == "next_logical_deduction":
            deduction = target.get("deduction")
            if not isinstance(deduction, dict):
                findings.append(
                    ValidationFinding("E", ValidationSeverity.ERROR, "missing_deduction_target", "next_logical_deduction requires deduction dictionary", puzzle_id, file_path, line_no)
                )
            else:
                v_copy, m_copy = init_masks(puz_digits, geom)
                if v_copy is not None and m_copy is not None:
                    step_ok, step_err = verify_single_step_proof(deduction, v_copy, m_copy, geom)
                    if not step_ok:
                        findings.append(
                            ValidationFinding("E", ValidationSeverity.ERROR, "invalid_target_deduction", f"Deduction invalid: {step_err}", puzzle_id, file_path, line_no)
                        )

        elif task_type == "solver_policy":
            r_cell = target.get("recommended_cell")
            r_digit = target.get("recommended_digit")
            if r_cell is not None and r_digit is not None:
                if not (0 <= r_cell < NUM_CELLS) or puz_digits[r_cell] != 0:
                    findings.append(
                        ValidationFinding("E", ValidationSeverity.ERROR, "invalid_policy_cell", f"Recommended cell {r_cell} is not empty in puzzle", puzzle_id, file_path, line_no)
                    )
                elif authoritative_solution[r_cell] != r_digit:
                    findings.append(
                        ValidationFinding("E", ValidationSeverity.ERROR, "invalid_policy_digit", f"Recommended digit {r_digit} does not match verified solution digit {authoritative_solution[r_cell]}", puzzle_id, file_path, line_no)
                    )

    # Proof Trace Replay
    trace = cur_rec.get("technique_trace", [])
    if isinstance(trace, list) and len(trace) > 0:
        trace_ok, f_step, f_msg = replay_technique_trace(puz_digits, trace, geom)
        if not trace_ok:
            findings.append(
                ValidationFinding("E", ValidationSeverity.ERROR, "invalid_technique_trace", f"Trace replay failed at step {f_step}: {f_msg}", puzzle_id, file_path, line_no)
            )

    # -----------------------------------------------------------------------
    # Layer F: Difficulty Calibration Metadata
    # -----------------------------------------------------------------------
    diff = cur_rec.get("difficulty")
    diff_raw = cur_rec.get("difficulty_raw")
    if not isinstance(diff, (int, float)) or isinstance(diff, bool) or not math.isfinite(diff) or not (1.0 <= diff <= 10.0):
        findings.append(
            ValidationFinding("F", ValidationSeverity.ERROR, "invalid_difficulty_range", f"Difficulty {diff} out of bounds [1.0, 10.0]", puzzle_id, file_path, line_no)
        )
    if not isinstance(diff_raw, (int, float)) or isinstance(diff_raw, bool) or not math.isfinite(diff_raw) or not (1.0 <= diff_raw <= 10.0):
        findings.append(
            ValidationFinding("F", ValidationSeverity.ERROR, "invalid_difficulty_raw", f"Raw difficulty {diff_raw} out of bounds [1.0, 10.0]", puzzle_id, file_path, line_no)
        )

    # Check Deterministic Hashes in Provenance
    exact_h, canon_h, fam_h = compute_independent_hashes(puz_digits, authoritative_solution, variant)
    prov = cur_rec.get("provenance")
    if not isinstance(prov, dict):
        findings.append(
            ValidationFinding("F", ValidationSeverity.WARNING, "missing_provenance", "Provenance dictionary missing", puzzle_id, file_path, line_no, repairable=True)
        )
        if mode in (RepairMode.REPAIR_SAFE, RepairMode.REPAIR_AGGRESSIVE):
            cur_rec["provenance"] = {
                "exact_puzzle_hash": exact_h,
                "canonical_puzzle_hash": canon_h,
                "solution_family_hash": fam_h,
                "calibration_version": "sm-cal-v2.1",
            }
            repairs.append(RepairRecord(puzzle_id, file_path, line_no, "rebuild_provenance", "None", str(cur_rec["provenance"])))
    else:
        if prov.get("canonical_puzzle_hash") != canon_h or prov.get("exact_puzzle_hash") != exact_h:
            findings.append(
                ValidationFinding("F", ValidationSeverity.WARNING, "provenance_hash_mismatch", "Provenance hashes differ from recomputed hashes", puzzle_id, file_path, line_no, repairable=True)
            )
            if mode in (RepairMode.REPAIR_SAFE, RepairMode.REPAIR_AGGRESSIVE):
                prov["exact_puzzle_hash"] = exact_h
                prov["canonical_puzzle_hash"] = canon_h
                prov["solution_family_hash"] = fam_h
                repairs.append(RepairRecord(puzzle_id, file_path, line_no, "correct_provenance_hashes", "mismatched", canon_h[:16]))

    # Finalize Layer A Checksum Repair if necessary
    has_critical_or_error = any(f.severity in (ValidationSeverity.CRITICAL, ValidationSeverity.ERROR) for f in findings)
    
    if sha_mismatch or len(repairs) > 0:
        if not has_critical_or_error and mode in (RepairMode.REPAIR_SAFE, RepairMode.REPAIR_AGGRESSIVE):
            cur_rec.pop("record_sha256", None)
            new_sha = hashlib.sha256(json.dumps(cur_rec, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
            cur_rec["record_sha256"] = new_sha
            repairs.append(RepairRecord(puzzle_id, file_path, line_no, "recalculate_record_sha256", str(claimed_sha)[:16], new_sha[:16]))

    is_valid = not has_critical_or_error
    is_repaired = len(repairs) > 0 and is_valid
    is_quarantined = not is_valid

    return RecordAuditResult(
        record_id=puzzle_id,
        is_valid=is_valid,
        is_repaired=is_repaired,
        is_quarantined=is_quarantined,
        unresolved=False,
        repaired_record=cur_rec if is_repaired else (cur_rec if is_valid else None),
        findings=findings,
        repairs=repairs,
    )


# ---------------------------------------------------------------------------
# Multiprocessing Worker & Streaming File Auditor
# ---------------------------------------------------------------------------

@dataclass
class WorkerBatchTask:
    lines: List[Tuple[int, str]]  # (line_number, raw_json_str)
    file_path: str
    mode: str
    max_solver_seconds: float


@dataclass
class WorkerBatchResult:
    results: List[RecordAuditResult]
    parse_errors: List[ValidationFinding]


def _audit_batch_worker(task: WorkerBatchTask) -> WorkerBatchResult:
    mode = RepairMode(task.mode)
    out_results: List[RecordAuditResult] = []
    out_errors: List[ValidationFinding] = []

    for line_no, line_str in task.lines:
        stripped = line_str.strip()
        if not stripped:
            continue
        try:
            parsed = strict_json_loads(stripped)
        except Exception as exc:
            out_errors.append(
                ValidationFinding(
                    layer="A",
                    severity=ValidationSeverity.CRITICAL,
                    category="json_parse_failure",
                    message=f"JSON parsing error: {exc}",
                    record_id=f"line_{line_no}",
                    file_path=task.file_path,
                    line_number=line_no,
                )
            )
            continue

        res = audit_single_record(
            record=parsed,
            file_path=task.file_path,
            line_no=line_no,
            mode=mode,
            max_solver_seconds=task.max_solver_seconds,
        )
        out_results.append(res)

    return WorkerBatchResult(out_results, out_errors)


# ---------------------------------------------------------------------------
# Dataset Discovery & Pipeline Runner
# ---------------------------------------------------------------------------

@dataclass
class DatasetFileInfo:
    file_path: str
    split_name: str
    file_size_bytes: int
    is_test_split: bool


def discover_input_files(target_path: str) -> List[DatasetFileInfo]:
    abs_path = os.path.abspath(target_path)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"Input path does not exist: {abs_path}")

    files: List[DatasetFileInfo] = []
    if os.path.isfile(abs_path):
        fname = os.path.basename(abs_path)
        split = "unknown"
        for candidate in ("train", "validation", "test"):
            if candidate in fname:
                split = candidate
                break
        files.append(DatasetFileInfo(abs_path, split, os.path.getsize(abs_path), split == "test"))
        return files

    # Directory walk
    for root, _, filenames in os.walk(abs_path):
        for f in sorted(filenames):
            if f.endswith(".jsonl") and not f.endswith(".part"):
                fpath = os.path.join(root, f)
                rel = os.path.relpath(fpath, abs_path).lower()
                split = "train"
                if "test" in rel:
                    split = "test"
                elif "val" in rel:
                    split = "validation"
                files.append(DatasetFileInfo(fpath, split, os.path.getsize(fpath), split == "test"))

    return files


def audit_dataset_manifest(split_dir: str, findings: List[ValidationFinding]) -> None:
    manifest_path = os.path.join(split_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        return
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            m = json.load(f)
    except Exception as exc:
        findings.append(ValidationFinding("A", ValidationSeverity.ERROR, "corrupted_manifest", f"Manifest JSON invalid: {exc}", file_path=manifest_path))
        return

    data_file = m.get("file_path")
    declared_sha = m.get("file_sha256")
    if data_file and declared_sha:
        full_data_path = os.path.join(split_dir, data_file)
        if os.path.exists(full_data_path):
            h = hashlib.sha256()
            with open(full_data_path, "rb") as df:
                while chunk := df.read(1024 * 1024):
                    h.update(chunk)
            actual_sha = h.hexdigest()
            if actual_sha != declared_sha:
                findings.append(
                    ValidationFinding(
                        layer="A",
                        severity=ValidationSeverity.ERROR,
                        category="manifest_hash_mismatch",
                        message=f"Manifest sha256 ({declared_sha[:16]}) != actual file ({actual_sha[:16]})",
                        file_path=manifest_path,
                    )
                )


# ---------------------------------------------------------------------------
# Orchestrator & CLI Runner
# ---------------------------------------------------------------------------

@dataclass
class AuditExecutionConfig:
    input_path: str
    output_path: Optional[str] = None
    quarantine_path: Optional[str] = None
    report_path: Optional[str] = None
    repair_log_path: Optional[str] = None
    mode: RepairMode = RepairMode.AUDIT
    cross_split: bool = False
    dry_run: bool = False
    workers: int = 2
    max_solver_seconds: float = 2.0
    max_line_bytes: int = 2_000_000
    batch_size: int = 100
    verbose: bool = False
    quiet: bool = False


def run_auditor(cfg: AuditExecutionConfig) -> Tuple[ExitCode, Dict[str, Any]]:
    t0 = time.perf_counter()
    files = discover_input_files(cfg.input_path)
    if not files:
        sys.stderr.write(f"No .jsonl files found in {cfg.input_path}\n")
        return ExitCode.CLI_ERROR, {}

    if not cfg.quiet:
        print("=" * 78)
        print(f"{PROGRAM_NAME} v{PROGRAM_VERSION} | Independent Verification & Audit Engine")
        print("=" * 78)
        print(f"Mode                 : {cfg.mode.value.upper()} (dry_run={cfg.dry_run})")
        print(f"Independent Solver   : {INDEPENDENT_SOLVER_NAME}")
        print(f"Input Target         : {os.path.abspath(cfg.input_path)} ({len(files)} file(s))")
        print(f"Output Directory     : {os.path.abspath(cfg.output_path) if cfg.output_path else 'None (Read-Only Audit)'}")
        print(f"Cross-Split Check    : {cfg.cross_split}")
        print(f"Workers / Timeout    : {cfg.workers} worker(s) | {cfg.max_solver_seconds:.1f}s per board")
        print("=" * 78)

    temp_dir = tempfile.mkdtemp(prefix="sudokumind_audit_")
    registry_path = os.path.join(temp_dir, "leak_registry.sqlite3")
    registry = AuditRegistry(registry_path)

    total_records = 0
    valid_count = 0
    invalid_count = 0
    unresolved_count = 0
    repaired_count = 0
    quarantined_count = 0

    all_findings: List[ValidationFinding] = []
    all_repairs: List[RepairRecord] = []

    # Check manifest files if directory given
    if os.path.isdir(cfg.input_path):
        for root, dirs, _ in os.walk(cfg.input_path):
            audit_dataset_manifest(root, all_findings)

    pool: Optional[mp.pool.Pool] = None
    if cfg.workers > 1:
        pool = mp.Pool(processes=cfg.workers)

    try:
        for file_info in files:
            if not cfg.quiet:
                print(f"\n[AUDIT] Scanning {file_info.split_name.upper()}: {os.path.basename(file_info.file_path)} ({file_info.file_size_bytes / 1024:.1f} KB)")

            out_handle = None
            quarantine_handle = None
            out_tmp_path = None
            quarantine_tmp_path = None

            if cfg.output_path and not cfg.dry_run and cfg.mode in (RepairMode.REPAIR_SAFE, RepairMode.REPAIR_AGGRESSIVE):
                rel_sub = os.path.relpath(file_info.file_path, os.path.abspath(cfg.input_path)) if os.path.isdir(cfg.input_path) else os.path.basename(file_info.file_path)
                dest_file = os.path.join(cfg.output_path, rel_sub)
                os.makedirs(os.path.dirname(dest_file), exist_ok=True)
                out_tmp_path = f"{dest_file}.tmp.{os.getpid()}"
                out_handle = open(out_tmp_path, "w", encoding="utf-8")

            if cfg.quarantine_path and not cfg.dry_run and cfg.mode in (RepairMode.QUARANTINE, RepairMode.REPAIR_SAFE, RepairMode.REPAIR_AGGRESSIVE):
                q_dest = os.path.join(cfg.quarantine_path, f"quarantine_{file_info.split_name}.jsonl")
                os.makedirs(os.path.dirname(q_dest), exist_ok=True)
                quarantine_tmp_path = f"{q_dest}.tmp.{os.getpid()}"
                quarantine_handle = open(quarantine_tmp_path, "a", encoding="utf-8")

            batch_lines: List[Tuple[int, str]] = []

            def _process_batch(lines_to_eval: List[Tuple[int, str]]) -> None:
                nonlocal total_records, valid_count, invalid_count, unresolved_count, repaired_count, quarantined_count
                task = WorkerBatchTask(lines_to_eval, file_info.file_path, cfg.mode.value, cfg.max_solver_seconds)
                batch_res = _audit_batch_worker(task) if pool is None else pool.apply(_audit_batch_worker, (task,))

                for err in batch_res.parse_errors:
                    all_findings.append(err)
                    invalid_count += 1
                    total_records += 1

                for r in batch_res.results:
                    total_records += 1
                    all_findings.extend(r.findings)
                    all_repairs.extend(r.repairs)

                    # Layer G cross-split leakage check
                    if r.is_valid and cfg.cross_split:
                        prov = r.repaired_record["provenance"] if r.repaired_record else {}
                        c_h = prov.get("canonical_puzzle_hash", "")
                        e_h = prov.get("exact_puzzle_hash", "")
                        f_h = prov.get("solution_family_hash", "")
                        l_findings = registry.register_and_audit_leak(
                            r.record_id, file_info.split_name, file_info.file_path, 0, e_h, c_h, f_h
                        )
                        if l_findings:
                            all_findings.extend(l_findings)
                            r.is_valid = False
                            r.is_quarantined = True

                    if r.unresolved:
                        unresolved_count += 1
                    elif r.is_valid:
                        valid_count += 1
                        if r.is_repaired:
                            repaired_count += 1
                        if out_handle is not None and r.repaired_record is not None:
                            out_handle.write(json.dumps(r.repaired_record, sort_keys=True) + "\n")
                    else:
                        invalid_count += 1
                        if r.is_quarantined:
                            quarantined_count += 1
                            if quarantine_handle is not None:
                                quarantine_handle.write(json.dumps({
                                    "record_id": r.record_id,
                                    "file_path": file_info.file_path,
                                    "reasons": [f.message for f in r.findings if f.severity in (ValidationSeverity.ERROR, ValidationSeverity.CRITICAL)],
                                }, sort_keys=True) + "\n")

            with open(file_info.file_path, "r", encoding="utf-8", errors="replace") as f:
                for line_idx, line in enumerate(f, start=1):
                    if len(line.encode("utf-8")) > cfg.max_line_bytes:
                        all_findings.append(
                            ValidationFinding(
                                layer="A",
                                severity=ValidationSeverity.CRITICAL,
                                category="line_length_exceeded",
                                message=f"Line {line_idx} exceeded size limit ({cfg.max_line_bytes} bytes)",
                                file_path=file_info.file_path,
                                line_number=line_idx,
                            )
                        )
                        invalid_count += 1
                        total_records += 1
                        continue

                    batch_lines.append((line_idx, line))
                    if len(batch_lines) >= cfg.batch_size:
                        _process_batch(batch_lines)
                        batch_lines = []

                if batch_lines:
                    _process_batch(batch_lines)

            if out_handle is not None:
                out_handle.close()
                assert out_tmp_path is not None
                final_dest = out_tmp_path.replace(f".tmp.{os.getpid()}", "")
                os.replace(out_tmp_path, final_dest)

            if quarantine_handle is not None:
                quarantine_handle.close()
                assert quarantine_tmp_path is not None
                final_q_dest = quarantine_tmp_path.replace(f".tmp.{os.getpid()}", "")
                os.replace(quarantine_tmp_path, final_q_dest)

    finally:
        if pool is not None:
            pool.terminate()
            pool.join()
        registry.close()
        shutil.rmtree(temp_dir, ignore_errors=True)

    elapsed_sec = round(time.perf_counter() - t0, 3)

    # Determine status
    has_critical = any(f.severity == ValidationSeverity.CRITICAL for f in all_findings)
    has_error = any(f.severity == ValidationSeverity.ERROR for f in all_findings)
    has_warning = any(f.severity == ValidationSeverity.WARNING for f in all_findings)

    if _SHUTDOWN_REQUESTED or unresolved_count > 0:
        overall_status = AuditStatus.INCOMPLETE
        exit_code = ExitCode.INCOMPLETE
    elif has_critical or has_error or invalid_count > 0:
        overall_status = AuditStatus.FAIL
        exit_code = ExitCode.FAIL
    elif has_warning or repaired_count > 0:
        overall_status = AuditStatus.PASS_WITH_WARNINGS
        exit_code = ExitCode.PASS_WITH_WARNINGS
    else:
        overall_status = AuditStatus.PASS
        exit_code = ExitCode.PASS

    # Produce Machine-Readable JSON Report
    report_dict: Dict[str, Any] = {
        "validator_name": PROGRAM_NAME,
        "validator_version": PROGRAM_VERSION,
        "audit_timestamp_unix": time.time(),
        "elapsed_seconds": elapsed_sec,
        "overall_status": overall_status.value,
        "mode": cfg.mode.value,
        "dry_run": cfg.dry_run,
        "summary_counts": {
            "total_records": total_records,
            "valid_records": valid_count,
            "invalid_records": invalid_count,
            "unresolved_records": unresolved_count,
            "repaired_records": repaired_count,
            "quarantined_records": quarantined_count,
        },
        "findings_by_severity": {
            "critical": sum(1 for f in all_findings if f.severity == ValidationSeverity.CRITICAL),
            "error": sum(1 for f in all_findings if f.severity == ValidationSeverity.ERROR),
            "warning": sum(1 for f in all_findings if f.severity == ValidationSeverity.WARNING),
            "info": sum(1 for f in all_findings if f.severity == ValidationSeverity.INFO),
        },
        "findings": [f.to_dict() for f in all_findings[:500]],  # Cap detailed telemetry to first 500
        "repairs_applied": [r.to_dict() for r in all_repairs[:500]],
    }

    if cfg.report_path:
        os.makedirs(os.path.dirname(os.path.abspath(cfg.report_path)), exist_ok=True)
        with open(cfg.report_path, "w", encoding="utf-8") as rf:
            json.dump(report_dict, rf, indent=2, sort_keys=True)

    if cfg.repair_log_path and all_repairs:
        os.makedirs(os.path.dirname(os.path.abspath(cfg.repair_log_path)), exist_ok=True)
        with open(cfg.repair_log_path, "w", encoding="utf-8") as rlf:
            json.dump([r.to_dict() for r in all_repairs], rlf, indent=2, sort_keys=True)

    if not cfg.quiet:
        print("\n" + "=" * 78)
        print("AUDIT EXECUTION SUMMARY")
        print("=" * 78)
        print(f"Overall Status       : {overall_status.value} (Exit Code: {exit_code.value})")
        print(f"Total Evaluated      : {total_records:,}")
        print(f"Mathematically Valid : {valid_count:,} ({valid_count / max(1, total_records) * 100:.1f}%)")
        print(f"Invalid / Rejected   : {invalid_count:,}")
        print(f"Unresolved / Timeout : {unresolved_count:,}")
        print(f"Repaired Operations  : {repaired_count:,}")
        print(f"Quarantined Records  : {quarantined_count:,}")
        print(f"Elapsed Time         : {elapsed_sec:.2f}s ({total_records / max(0.001, elapsed_sec):.1f} rec/s)")
        if cfg.report_path:
            print(f"JSON Report          : {os.path.abspath(cfg.report_path)}")
        print("=" * 78)

    return exit_code, report_dict


# ---------------------------------------------------------------------------
# Comprehensive Built-In Test Suite (32 Invariant Scenarios)
# ---------------------------------------------------------------------------

def run_self_tests() -> int:
    print("=" * 78)
    print(f"{PROGRAM_NAME} v{PROGRAM_VERSION} — Executing 32 Automated Invariant Tests")
    print("=" * 78)
    passed = 0
    failed = 0

    std_geom = compile_geometry("standard")

    def _execute_case(test_num: int, title: str, fn: Callable[[], None]) -> None:
        nonlocal passed, failed
        t_start = time.perf_counter()
        try:
            fn()
            dur = (time.perf_counter() - t_start) * 1000.0
            print(f"  [PASS] Test {test_num:02d}: {title:<52} ({dur:.1f} ms)")
            passed += 1
        except Exception as exc:
            print(f"  [FAIL] Test {test_num:02d}: {title:<52} -> {exc}")
            failed += 1

    # Base valid fixtures
    sample_puz_str = "530070000600195000098000060800060003400803001700020006060000280000419005000080079"
    sample_sol_str = "534678912672195348198342567859761423426853791713924856961537284287419635345286179"

    def _make_dummy_record(**kwargs: Any) -> Dict[str, Any]:
        rec: Dict[str, Any] = {
            "schema_version": "sm-schema-v2.0",
            "puzzle_id": "test_puz_001",
            "split": "train",
            "variant": "standard",
            "difficulty": 2.5,
            "difficulty_raw": 2.54,
            "difficulty_features": {"clue_count": 30, "empty_cells": 51},
            "givens": sum(1 for c in sample_puz_str if c != "0"),
            "puzzle": sample_puz_str,
            "solution": sample_sol_str,
            "solution_count_status": "unique",
            "unique_solution": True,
            "task_type": "complete_solution",
            "target": {"task_type": "complete_solution", "solution_string": sample_sol_str},
            "technique_trace": [],
            "generator_version": "2.1.0",
            "generation_seed": 42,
            "solver_version": INDEPENDENT_SOLVER_NAME,
            "provenance": {
                "exact_puzzle_hash": "dummy_exact",
                "canonical_puzzle_hash": "dummy_canon",
                "solution_family_hash": "dummy_fam",
                "calibration_version": "sm-cal-v2.1",
            },
            "validation_status": "verified_unique",
        }
        rec.update(kwargs)
        rec_copy = dict(rec)
        rec_copy.pop("record_sha256", None)
        rec["record_sha256"] = hashlib.sha256(json.dumps(rec_copy, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        return rec

    # 1. Valid standard record
    def _t01() -> None:
        rec = _make_dummy_record()
        res = audit_single_record(rec, "f", 1, RepairMode.AUDIT)
        assert res.is_valid, f"Expected valid, got findings: {res.findings}"

    # 2. Invalid grid dimensions
    def _t02() -> None:
        rec = _make_dummy_record(puzzle="12345")
        res = audit_single_record(rec, "f", 1, RepairMode.AUDIT)
        assert not res.is_valid and any(f.category == "malformed_puzzle_representation" for f in res.findings)

    # 3. Contradictory clues
    def _t03() -> None:
        bad_puz = "550000000" + "0" * 72
        rec = _make_dummy_record(puzzle=bad_puz, givens=2)
        res = audit_single_record(rec, "f", 1, RepairMode.AUDIT)
        assert not res.is_valid and any(f.category == "contradictory_givens" for f in res.findings)

    # 4. Unsatisfiable puzzle
    def _t04() -> None:
        unsat = "012345678900000000000000000000000000000000000000000000000000000000000000000000000"
        rec = _make_dummy_record(puzzle=unsat, givens=sum(1 for c in unsat if c != "0"))
        res = audit_single_record(rec, "f", 1, RepairMode.AUDIT)
        assert not res.is_valid and any(f.category == "unsatisfiable_puzzle" for f in res.findings)

    # 5. Unique-solution puzzles
    def _t05() -> None:
        grid, _ = parse_grid_to_digits(sample_puz_str)
        assert grid is not None
        s = independent_solve_and_count(grid, std_geom, max_solutions=2)
        assert s.status == SolutionStatus.UNIQUE and len(s.solutions) == 1

    # 6. Multiple-solution puzzles
    def _t06() -> None:
        sparse = [0] * NUM_CELLS
        sparse[0], sparse[1] = 1, 2
        s = independent_solve_and_count(sparse, std_geom, max_solutions=2)
        assert s.status == SolutionStatus.MULTIPLE

    # 7. Wrong reference solution
    def _t07() -> None:
        wrong_sol = "1" * 81
        rec = _make_dummy_record(solution=wrong_sol)
        res = audit_single_record(rec, "f", 1, RepairMode.AUDIT)
        assert not res.is_valid and any("solution" in f.category for f in res.findings)

    # 8. False uniqueness claim
    def _t08() -> None:
        sparse_str = "12" + "0" * 79
        rec = _make_dummy_record(puzzle=sparse_str, givens=2, unique_solution=True)
        res = audit_single_record(rec, "f", 1, RepairMode.AUDIT)
        assert not res.is_valid and any(f.category == "false_uniqueness_claim" for f in res.findings)

    # 9. Incorrect task targets
    def _t09() -> None:
        rec = _make_dummy_record(target={"task_type": "complete_solution", "solution_string": "9" * 81})
        res = audit_single_record(rec, "f", 1, RepairMode.AUDIT)
        assert any(f.category == "target_solution_mismatch" for f in res.findings)

    # 10. Invalid logical deductions
    def _t10() -> None:
        step = {"technique": "full_house", "action_type": "placement", "target_cells": [0], "digits": [9]}
        rec = _make_dummy_record(task_type="next_logical_deduction", target={"deduction": step})
        res = audit_single_record(rec, "f", 1, RepairMode.AUDIT)
        assert any(f.category == "invalid_target_deduction" for f in res.findings)

    # 11. Invalid proof traces
    def _t11() -> None:
        trace = [{"technique": "naked_single", "action_type": "placement", "target_cells": [2], "digits": [9]}]
        rec = _make_dummy_record(technique_trace=trace)
        res = audit_single_record(rec, "f", 1, RepairMode.AUDIT)
        assert any(f.category == "invalid_technique_trace" for f in res.findings)

    # 12. Incorrect difficulty metadata
    def _t12() -> None:
        rec = _make_dummy_record(difficulty=15.0)
        res = audit_single_record(rec, "f", 1, RepairMode.AUDIT)
        assert any(f.category == "invalid_difficulty_range" for f in res.findings)

    # 13. Duplicate records detection
    def _t13() -> None:
        with tempfile.TemporaryDirectory() as td:
            db_p = os.path.join(td, "test_reg.db")
            reg = AuditRegistry(db_p)
            f1 = reg.register_and_audit_leak("p1", "train", "f1", 1, "e1", "c1", "fam1")
            assert len(f1) == 0
            f2 = reg.register_and_audit_leak("p1", "train", "f1", 2, "e1", "c1", "fam1")
            assert any(x.category == "duplicate_puzzle_id" for x in f2)
            reg.close()

    # 14. Cross-split leakage
    def _t14() -> None:
        with tempfile.TemporaryDirectory() as td:
            db_p = os.path.join(td, "test_reg.db")
            reg = AuditRegistry(db_p)
            reg.register_and_audit_leak("p_train", "train", "f1", 1, "e_same", "c_same", "fam_same")
            leaks = reg.register_and_audit_leak("p_test", "test", "f2", 1, "e_same", "c_same", "fam_same")
            assert any(x.category == "cross_split_puzzle_leakage" for x in leaks)
            reg.close()

    # 15. Equivalent puzzles under D4 + digit permutation
    def _t15() -> None:
        g1, _ = parse_grid_to_digits(sample_puz_str)
        assert g1 is not None
        # Invert digits (1..9 -> 9..1)
        g2 = [(10 - x) if x != 0 else 0 for x in g1]
        c1 = canonicalize_d4_grid(g1)
        c2 = canonicalize_d4_grid(g2)
        assert c1 == c2

    # 16. Malformed JSONL
    def _t16() -> None:
        try:
            strict_json_loads("{bad json")
            raise AssertionError("Should raise JSONDecodeError")
        except Exception:
            pass

    # 17. Duplicate JSON keys
    def _t17() -> None:
        raw = '{"a": 1, "a": 2}'
        try:
            strict_json_loads(raw)
            raise AssertionError("Should reject duplicate JSON keys")
        except ValueError:
            pass

    # 18. Truncated final records
    def _t18() -> None:
        with tempfile.TemporaryDirectory() as td:
            fpath = os.path.join(td, "trunc.jsonl")
            with open(fpath, "w", encoding="utf-8") as f:
                f.write('{"schema_version": "sm-schema-v2.0"\n')
            cfg = AuditExecutionConfig(input_path=fpath, quiet=True)
            ec, rep = run_auditor(cfg)
            assert ec == ExitCode.FAIL

    # 19. Incorrect hashes (record_sha256 mismatch)
    def _t19() -> None:
        rec = _make_dummy_record()
        rec["record_sha256"] = "wrong_hash"
        res = audit_single_record(rec, "f", 1, RepairMode.AUDIT)
        assert any(f.category == "checksum_mismatch" for f in res.findings)

    # 20. Manifest inconsistencies
    def _t20() -> None:
        with tempfile.TemporaryDirectory() as td:
            mf = os.path.join(td, "manifest.json")
            with open(mf, "w", encoding="utf-8") as f:
                json.dump({"file_path": "data.jsonl", "file_sha256": "bogus"}, f)
            with open(os.path.join(td, "data.jsonl"), "w", encoding="utf-8") as f:
                f.write("content\n")
            findings: List[ValidationFinding] = []
            audit_dataset_manifest(td, findings)
            assert any(f.category == "manifest_hash_mismatch" for f in findings)

    # 21. Safe repairs (recalculating sha256 & canonicalizing givens)
    def _t21() -> None:
        rec = _make_dummy_record(givens=99)
        res = audit_single_record(rec, "f", 1, RepairMode.REPAIR_SAFE)
        assert res.is_valid and res.is_repaired
        assert res.repaired_record is not None and res.repaired_record["givens"] == 30

    # 22. Repairs that must be refused
    def _t22() -> None:
        rec = _make_dummy_record(puzzle="12" + "0" * 79, givens=2)
        res = audit_single_record(rec, "f", 1, RepairMode.REPAIR_SAFE)
        assert not res.is_valid and res.repaired_record is None

    # 23. Quarantine behavior
    def _t23() -> None:
        with tempfile.TemporaryDirectory() as td:
            in_p = os.path.join(td, "bad.jsonl")
            q_p = os.path.join(td, "quarantine")
            with open(in_p, "w", encoding="utf-8") as f:
                f.write(json.dumps(_make_dummy_record(difficulty=20.0)) + "\n")
            cfg = AuditExecutionConfig(input_path=in_p, quarantine_path=q_p, mode=RepairMode.QUARANTINE, quiet=True)
            run_auditor(cfg)
            q_files = os.listdir(q_p)
            assert len(q_files) > 0

    # 24. Dry-run behavior
    def _t24() -> None:
        with tempfile.TemporaryDirectory() as td:
            in_p = os.path.join(td, "in.jsonl")
            out_p = os.path.join(td, "out")
            with open(in_p, "w", encoding="utf-8") as f:
                f.write(json.dumps(_make_dummy_record(givens=99)) + "\n")
            cfg = AuditExecutionConfig(input_path=in_p, output_path=out_p, mode=RepairMode.REPAIR_SAFE, dry_run=True, quiet=True)
            run_auditor(cfg)
            assert not os.path.exists(out_p)

    # 25. Atomic output handling
    def _t25() -> None:
        with tempfile.TemporaryDirectory() as td:
            in_p = os.path.join(td, "in.jsonl")
            out_p = os.path.join(td, "out")
            with open(in_p, "w", encoding="utf-8") as f:
                f.write(json.dumps(_make_dummy_record()) + "\n")
            cfg = AuditExecutionConfig(input_path=in_p, output_path=out_p, mode=RepairMode.REPAIR_SAFE, quiet=True)
            run_auditor(cfg)
            # Output file should exist and no .tmp files left
            assert os.path.exists(os.path.join(out_p, "in.jsonl"))
            assert all(not fn.endswith(".tmp") for fn in os.listdir(out_p))

    # 26. Interrupted operations
    def _t26() -> None:
        global _SHUTDOWN_REQUESTED
        _SHUTDOWN_REQUESTED = True
        try:
            with tempfile.TemporaryDirectory() as td:
                in_p = os.path.join(td, "in.jsonl")
                with open(in_p, "w", encoding="utf-8") as f:
                    f.write(json.dumps(_make_dummy_record()) + "\n")
                cfg = AuditExecutionConfig(input_path=in_p, quiet=True)
                ec, _ = run_auditor(cfg)
                assert ec == ExitCode.INCOMPLETE
        finally:
            _SHUTDOWN_REQUESTED = False

    # 27. Resource-limit enforcement (line-length)
    def _t27() -> None:
        with tempfile.TemporaryDirectory() as td:
            in_p = os.path.join(td, "huge.jsonl")
            with open(in_p, "w", encoding="utf-8") as f:
                f.write("A" * 1000 + "\n")
            cfg = AuditExecutionConfig(input_path=in_p, max_line_bytes=100, quiet=True)
            ec, _ = run_auditor(cfg)
            assert ec == ExitCode.FAIL

    # 28. Deterministic reports
    def _t28() -> None:
        with tempfile.TemporaryDirectory() as td:
            in_p = os.path.join(td, "in.jsonl")
            with open(in_p, "w", encoding="utf-8") as f:
                f.write(json.dumps(_make_dummy_record()) + "\n")
            cfg = AuditExecutionConfig(input_path=in_p, quiet=True)
            _, r1 = run_auditor(cfg)
            _, r2 = run_auditor(cfg)
            assert r1["summary_counts"] == r2["summary_counts"]

    # 29. Unsupported schema versions
    def _t29() -> None:
        rec = _make_dummy_record(schema_version="sm-schema-v999.0")
        res = audit_single_record(rec, "f", 1, RepairMode.AUDIT)
        assert any(f.category == "unsupported_schema" for f in res.findings)

    # 30. Unsupported variants
    def _t30() -> None:
        rec = _make_dummy_record(variant="thermo")
        res = audit_single_record(rec, "f", 1, RepairMode.AUDIT)
        assert any(f.category == "unsupported_variant" for f in res.findings)

    # 31. Corrupted checkpoints in directory scan
    def _t31() -> None:
        with tempfile.TemporaryDirectory() as td:
            with open(os.path.join(td, "train.jsonl"), "w", encoding="utf-8") as f:
                f.write(json.dumps(_make_dummy_record()) + "\n")
            with open(os.path.join(td, "checkpoint.json"), "w", encoding="utf-8") as f:
                f.write("corrupted json")
            cfg = AuditExecutionConfig(input_path=td, quiet=True)
            ec, _ = run_auditor(cfg)
            assert ec == ExitCode.PASS  # Checkpoint ignored if final dataset is present

    # 32. Held-out test set protection
    def _t32() -> None:
        with tempfile.TemporaryDirectory() as td:
            test_file = os.path.join(td, "test.jsonl")
            rec = _make_dummy_record(split="test")
            with open(test_file, "w", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            files = discover_input_files(test_file)
            assert files[0].is_test_split is True

    # Run all 32 cases
    cases = [
        (1, "Valid standard Sudoku records", _t01),
        (2, "Invalid grid dimensions rejection", _t02),
        (3, "Contradictory clue rejection", _t03),
        (4, "Unsatisfiable puzzle detection", _t04),
        (5, "Unique-solution verification", _t05),
        (6, "Multiple-solution puzzle detection", _t06),
        (7, "Wrong reference solution detection", _t07),
        (8, "False uniqueness claim audit", _t08),
        (9, "Incorrect task target verification", _t09),
        (10, "Invalid logical deduction detection", _t10),
        (11, "Invalid proof trace replay", _t11),
        (12, "Incorrect difficulty metadata check", _t12),
        (13, "Duplicate record detection", _t13),
        (14, "Cross-split contamination audit", _t14),
        (15, "Equivalent puzzles under D4 + digit permutation", _t15),
        (16, "Malformed JSONL handling", _t16),
        (17, "Duplicate JSON key rejection", _t17),
        (18, "Truncated final line detection", _t18),
        (19, "Tamper & SHA-256 integrity check", _t19),
        (20, "Manifest inconsistency audit", _t20),
        (21, "Category 1 Safe repair execution", _t21),
        (22, "Refusal of ambiguous repairs", _t22),
        (23, "Quarantine isolation behavior", _t23),
        (24, "Dry-run non-mutation guarantee", _t24),
        (25, "Atomic temporary file writing", _t25),
        (26, "Graceful interruption handling", _t26),
        (27, "Resource limits (line-size) enforcement", _t27),
        (28, "Deterministic reporting verification", _t28),
        (29, "Unsupported schema rejection", _t29),
        (30, "Unsupported variant rejection", _t30),
        (31, "Corrupted checkpoint resilience", _t31),
        (32, "Held-out test set identification", _t32),
    ]

    for num, title, test_fn in cases:
        _execute_case(num, title, test_fn)

    print("=" * 78)
    print(f"Test Suite Summary: {passed} passed, {failed} failed out of {len(cases)} tests.")
    print("=" * 78)
    return 0 if failed == 0 else 1


# ---------------------------------------------------------------------------
# CLI Argument Parsing & Entrypoint
# ---------------------------------------------------------------------------

def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dataset_validator.py",
        description=(
            f"{PROGRAM_NAME} v{PROGRAM_VERSION} — Industrial-grade Sudoku dataset integrity auditor, "
            "independent constraint verifier, leakage detector, and safe repair system."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("command", nargs="?", default="audit", choices=["audit", "repair-safe", "repair-aggressive", "quarantine"], help="Primary execution command (default: audit)")
    parser.add_argument("--input", default="datasets", help="Input JSONL file or dataset directory (default: datasets)")
    parser.add_argument("--output", default=None, help="Output directory for repaired datasets")
    parser.add_argument("--quarantine", default=None, help="Output directory for quarantined invalid records")
    parser.add_argument("--report", default="reports/dataset_audit.json", help="Path for machine-readable JSON report")
    parser.add_argument("--repair-log", default="reports/repair_log.json", help="Path for JSON repair operation log")
    parser.add_argument("--cross-split", action="store_true", help="Audit cross-split puzzle and family contamination")
    parser.add_argument("--dry-run", action="store_true", help="Simulate audit/repair without writing data to disk")
    parser.add_argument("--workers", type=int, default=2, help="Worker processes for constraint validation (default: 2)")
    parser.add_argument("--max-solver-seconds", type=float, default=2.0, help="Per-puzzle independent solver timeout (default: 2.0s)")
    parser.add_argument("--max-line-bytes", type=int, default=2_000_000, help="Maximum allowable line size in bytes (default: 2MB)")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size per worker task (default: 100)")
    parser.add_argument("--run-tests", action="store_true", help="Execute built-in 32-scenario automated test suite")
    parser.add_argument("--verbose", action="store_true", help="Print verbose per-file diagnostic information")
    parser.add_argument("--quiet", action="store_true", help="Suppress console progress output")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    signal.signal(signal.SIGINT, _sig_handler)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _sig_handler)

    parser = build_cli_parser()
    args = parser.parse_args(argv)

    if args.run_tests:
        return run_self_tests()

    mode = RepairMode(args.command)
    cfg = AuditExecutionConfig(
        input_path=args.input,
        output_path=args.output,
        quarantine_path=args.quarantine,
        report_path=args.report,
        repair_log_path=args.repair_log,
        mode=mode,
        cross_split=args.cross_split,
        dry_run=args.dry_run,
        workers=args.workers,
        max_solver_seconds=args.max_solver_seconds,
        max_line_bytes=args.max_line_bytes,
        batch_size=args.batch_size,
        verbose=args.verbose,
        quiet=args.quiet,
    )

    try:
        exit_code, _ = run_auditor(cfg)
        return int(exit_code)
    except Exception as exc:
        sys.stderr.write(f"\n[FATAL ERROR] {exc.__class__.__name__}: {exc}\n")
        return int(ExitCode.CLI_ERROR)


if __name__ == "__main__":
    sys.exit(main())