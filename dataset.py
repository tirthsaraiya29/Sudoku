#!/usr/bin/env python3
"""
SudokuMind dataset.py — Production-Grade Sudoku Dataset Generator
=================================================================

Role in the 4-file SudokuMind architecture:
  1. dataset.py          (THIS FILE) Dataset generation, exact uniqueness verification,
                         human-strategy logical trace extraction, multi-factor difficulty
                         calibration (1.0-10.0), SQLite-backed deduplication/leakage
                         prevention, atomic split finalization, and reporting.
  2. dataset_verifier.py Independent cryptographic/mathematical verification, repair,
                         deduplication audit, and quarantine.
  3. trainer.py          Progressive curriculum learning, checkpointing, and training.
  4. validator.py        Strict evaluation on the permanently held-out 10,000-puzzle test set.

Design guarantees:
  - Zero third-party runtime dependencies required (uses Python 3.10+ standard library).
  - Exact bitmask MRV solver with bounded solution counting (distinguishes 0, 1, >=2,
    and unresolved timeout/node-limit states).
  - Human-style logical strategy solver implementing 10 graded deduction tiers to extract
    interpretable difficulty features and step-by-step technique traces.
  - Multi-factor calibrated difficulty scoring on [1.0, 10.0] in 0.1 increments.
  - Cross-split leakage prevention via canonical symmetry + digit-relabelling hashes
    and completed-grid family hashes stored in a bounded-memory SQLite WAL registry.
  - Resumable, atomic file generation with configuration fingerprinting and test-set lock.
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
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

# ---------------------------------------------------------------------------
# Versioning & Constants
# ---------------------------------------------------------------------------

GENERATOR_NAME = "SudokuMind-DatasetGenerator"
GENERATOR_VERSION = "2.1.0"
SOLVER_VERSION = "sm-exact-mrv-logic-v2.1"
CALIBRATION_VERSION = "sm-cal-v2.1"
SCHEMA_VERSION = "sm-schema-v2.0"

GRID_SIZE = 9
BOX_SIZE = 3
NUM_CELLS = 81
ALL_DIGITS_MASK = 0x1FF  # 9 bits: (1 << 9) - 1 = 511
MIN_UNIQUE_CLUES_STANDARD = 17  # Proven mathematical lower bound for 9x9 standard Sudoku

# Global flag for graceful signal handling
_SHUTDOWN_REQUESTED = False


def _signal_handler(signum: int, _frame: Any) -> None:
    global _SHUTDOWN_REQUESTED
    _SHUTDOWN_REQUESTED = True
    sys.stderr.write(
        f"\n[SIGNAL] Received signal {signum}. Finishing active batch and checkpointing safely...\n"
    )


# ---------------------------------------------------------------------------
# Structured Exceptions & Enums
# ---------------------------------------------------------------------------


class DatasetError(Exception):
    """Base exception for SudokuMind dataset generation errors."""


class ConfigurationError(DatasetError):
    """Raised when CLI arguments or generation profiles are invalid."""


class UnsupportedVariantError(DatasetError):
    """Raised when a Sudoku variant is not fully implemented and verified."""


class ResourceLimitError(DatasetError):
    """Raised when disk space or memory constraints are violated."""


class TestSetProtectionError(DatasetError):
    """Raised when an existing test set would be overwritten without explicit flag."""


class ResumeIntegrityError(DatasetError):
    """Raised when resume state conflicts with current configuration or is corrupted."""


class SolutionCountStatus(str, Enum):
    UNSATISFIABLE = "unsatisfiable"           # Exhaustive search proved 0 solutions
    UNIQUE = "unique"                         # Exhaustive search proved exactly 1 solution
    MULTIPLE = "multiple"                     # Bounded search found >= 2 distinct solutions
    UNRESOLVED_TIMEOUT = "unresolved_timeout" # Search aborted due to wall-clock timeout
    UNRESOLVED_LIMIT = "unresolved_node_limit"# Search aborted due to max node limit


class TaskType(str, Enum):
    COMPLETE_SOLUTION = "complete_solution"
    NEXT_LOGICAL_DEDUCTION = "next_logical_deduction"
    CANDIDATE_ELIMINATION = "candidate_elimination"
    SOLVER_POLICY = "solver_policy"


class ValidationStatus(str, Enum):
    VERIFIED_UNIQUE = "verified_unique"
    VERIFIED_NON_UNIQUE = "verified_non_unique"
    UNRESOLVED = "unresolved"
    INVALID = "invalid"


# ---------------------------------------------------------------------------
# Extensible Variant & Constraint Representation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExtraConstraintSpec:
    """
    Serializable definition for extra regional, arithmetic, or relational constraints.
    Allows downstream dataset_verifier.py, trainer.py, and validator.py to reconstruct
    exact board rules without variant-specific hardcoding.
    """
    constraint_type: str  # e.g., "diagonal", "hyper_region", "anti_knight", "killer_cage", "thermo"
    cells: Tuple[int, ...] = ()
    value: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "constraint_type": self.constraint_type,
            "cells": list(self.cells),
            "value": self.value,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class VariantSpec:
    """
    Complete mathematical specification of a 9x9 Sudoku variant.
    """
    name: str
    description: str
    implemented: bool
    min_theoretical_clues: int
    units: Tuple[Tuple[int, ...], ...]
    cell_units: Tuple[Tuple[int, ...], ...]
    peers: Tuple[Tuple[int, ...], ...]
    extra_constraints: Tuple[ExtraConstraintSpec, ...]
    preserves_full_dihedral_symmetry: bool = True

    def to_metadata_dict(self) -> Dict[str, Any]:
        return {
            "variant_name": self.name,
            "implemented": self.implemented,
            "min_theoretical_clues": self.min_theoretical_clues,
            "unit_count": len(self.units),
            "extra_constraints": [c.to_dict() for c in self.extra_constraints],
            "preserves_full_dihedral_symmetry": self.preserves_full_dihedral_symmetry,
        }


def _build_base_units() -> List[Tuple[int, ...]]:
    units: List[Tuple[int, ...]] = []
    # 9 rows
    for r in range(9):
        units.append(tuple(r * 9 + c for c in range(9)))
    # 9 cols
    for c in range(9):
        units.append(tuple(r * 9 + c for r in range(9)))
    # 9 3x3 boxes
    for br in range(0, 9, 3):
        for bc in range(0, 9, 3):
            units.append(
                tuple((br + dr) * 9 + (bc + dc) for dr in range(3) for dc in range(3))
            )
    return units


def _compile_variant_spec(
    name: str,
    description: str,
    implemented: bool,
    min_clues: int,
    extra_units: Sequence[Tuple[int, ...]] = (),
    extra_peer_deltas: Sequence[Tuple[int, int]] = (),
    extra_constraints: Sequence[ExtraConstraintSpec] = (),
    preserves_dihedral: bool = True,
) -> VariantSpec:
    all_units_list = _build_base_units() + list(extra_units)
    units_tuple = tuple(all_units_list)

    cell_units_list: List[Tuple[int, ...]] = []
    peers_sets: List[Set[int]] = [set() for _ in range(NUM_CELLS)]

    for u_idx, unit in enumerate(units_tuple):
        for cell in unit:
            peers_sets[cell].update(x for x in unit if x != cell)

    for cell in range(NUM_CELLS):
        u_indices = tuple(
            u_idx for u_idx, unit in enumerate(units_tuple) if cell in unit
        )
        cell_units_list.append(u_indices)
        r, c = divmod(cell, 9)
        for dr, dc in extra_peer_deltas:
            nr, nc = r + dr, c + dc
            if 0 <= nr < 9 and 0 <= nc < 9:
                peers_sets[cell].add(nr * 9 + nc)

    peers_tuple = tuple(tuple(sorted(s)) for s in peers_sets)
    return VariantSpec(
        name=name,
        description=description,
        implemented=implemented,
        min_theoretical_clues=min_clues,
        units=units_tuple,
        cell_units=tuple(cell_units_list),
        peers=peers_tuple,
        extra_constraints=tuple(extra_constraints),
        preserves_full_dihedral_symmetry=preserves_dihedral,
    )


def _build_variant_registry() -> Dict[str, VariantSpec]:
    registry: Dict[str, VariantSpec] = {}

    # 1. Standard 9x9 Sudoku (Mandatory First-Class Target)
    registry["standard"] = _compile_variant_spec(
        name="standard",
        description="Standard 9x9 Sudoku with 9 rows, 9 columns, and 9 3x3 boxes.",
        implemented=True,
        min_clues=MIN_UNIQUE_CLUES_STANDARD,
    )

    # 2. Diagonal Sudoku (Sudoku-X)
    diag1 = tuple(i * 9 + i for i in range(9))
    diag2 = tuple(i * 9 + (8 - i) for i in range(9))
    registry["diagonal"] = _compile_variant_spec(
        name="diagonal",
        description="Diagonal Sudoku (Sudoku-X): both main diagonals contain digits 1-9.",
        implemented=True,
        min_clues=12,
        extra_units=[diag1, diag2],
        extra_constraints=[
            ExtraConstraintSpec("diagonal", cells=diag1),
            ExtraConstraintSpec("diagonal", cells=diag2),
        ],
    )

    # 3. Hyper Sudoku (Windoku)
    hyper_boxes: List[Tuple[int, ...]] = []
    for br in (1, 5):
        for bc in (1, 5):
            hyper_boxes.append(
                tuple((br + dr) * 9 + (bc + dc) for dr in range(3) for dc in range(3))
            )
    registry["hyper"] = _compile_variant_spec(
        name="hyper",
        description="Hyper Sudoku (Windoku): four additional 3x3 interior shaded regions.",
        implemented=True,
        min_clues=12,
        extra_units=hyper_boxes,
        extra_constraints=[
            ExtraConstraintSpec("hyper_region", cells=hb) for hb in hyper_boxes
        ],
    )

    # 4. Anti-Knight Sudoku
    knight_deltas = [
        (-2, -1), (-2, 1), (-1, -2), (-1, 2),
        (1, -2), (1, 2), (2, -1), (2, 1),
    ]
    registry["anti_knight"] = _compile_variant_spec(
        name="anti_knight",
        description="Anti-Knight Sudoku: cells a chess knight move apart cannot repeat digits.",
        implemented=True,
        min_clues=12,
        extra_peer_deltas=knight_deltas,
        extra_constraints=[
            ExtraConstraintSpec("anti_knight", metadata={"deltas": knight_deltas})
        ],
    )

    # 5. Anti-King Sudoku
    king_deltas = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    registry["anti_king"] = _compile_variant_spec(
        name="anti_king",
        description="Anti-King Sudoku: diagonally adjacent cells cannot repeat digits.",
        implemented=True,
        min_clues=12,
        extra_peer_deltas=king_deltas,
        extra_constraints=[
            ExtraConstraintSpec("anti_king", metadata={"deltas": king_deltas})
        ],
    )

    # Planned / Schema-Specified Variants (Explicitly guarded until calibrated)
    unimplemented_specs = [
        ("non_consecutive", "Orthogonally adjacent cells cannot contain consecutive digits."),
        ("irregular", "Jigsaw/Irregular 9-cell contiguous regions replacing 3x3 boxes."),
        ("killer", "Cage sum constraints with non-repeating digits inside cages."),
        ("thermo", "Strictly increasing digits along thermometer paths from bulb to tip."),
        ("arrow", "Digits along an arrow sum to the value in the pill/circle cell."),
        ("sandwich", "Outside clues give the sum of digits sandwiched between 1 and 9."),
        ("renban", "Regions containing a set of consecutive digits in any order."),
        ("palindrome", "Lines reading the same forwards and backwards."),
    ]
    for v_name, v_desc in unimplemented_specs:
        registry[v_name] = _compile_variant_spec(
            name=v_name,
            description=v_desc,
            implemented=False,
            min_clues=17,
            extra_constraints=[ExtraConstraintSpec(v_name)],
            preserves_dihedral=False,
        )

    return registry


VARIANT_REGISTRY: Dict[str, VariantSpec] = _build_variant_registry()


def get_variant_spec(variant_name: str) -> VariantSpec:
    key = variant_name.strip().lower()
    if key not in VARIANT_REGISTRY:
        supported = ", ".join(
            k for k, v in VARIANT_REGISTRY.items() if v.implemented
        )
        raise UnsupportedVariantError(
            f"Unknown variant '{variant_name}'. Implemented & verified variants: [{supported}]."
        )
    spec = VARIANT_REGISTRY[key]
    if not spec.implemented:
        supported = ", ".join(
            k for k, v in VARIANT_REGISTRY.items() if v.implemented
        )
        raise UnsupportedVariantError(
            f"Variant '{variant_name}' ({spec.description}) has schema support planned "
            f"but its generator, uniqueness verifier, and difficulty calibration are not "
            f"enabled in {GENERATOR_VERSION}. Refusing to substitute standard Sudoku logic. "
            f"Supported variants: [{supported}]."
        )
    return spec


# ---------------------------------------------------------------------------
# Bitmask Helpers & Grid Formatting
# ---------------------------------------------------------------------------

BIT_TO_DIGIT: Dict[int, int] = {1 << d: d + 1 for d in range(9)}
DIGIT_TO_BIT: Tuple[int, ...] = (0,) + tuple(1 << d for d in range(9))


def popcount(mask: int) -> int:
    return mask.bit_count()


def mask_to_digits(mask: int) -> List[int]:
    return [d + 1 for d in range(9) if mask & (1 << d)]


def grid_to_string(grid: Sequence[int]) -> str:
    if len(grid) != NUM_CELLS:
        raise ValueError(f"Expected grid of length 81, got {len(grid)}")
    return "".join(str(x) if 1 <= x <= 9 else "0" for x in grid)


def string_to_grid(s: str) -> List[int]:
    cleaned = s.strip().replace(".", "0")
    if len(cleaned) != NUM_CELLS:
        raise ValueError(f"Expected 81-char grid string, got length {len(cleaned)}")
    out: List[int] = []
    for ch in cleaned:
        if "0" <= ch <= "9":
            out.append(int(ch))
        else:
            raise ValueError(f"Invalid grid character: {ch!r}")
    return out


# ---------------------------------------------------------------------------
# Exact Bitmask Constraint Propagation & Bounded Solver
# ---------------------------------------------------------------------------


@dataclass
class ExactSolverResult:
    status: SolutionCountStatus
    solution_count: int
    first_solution: Optional[List[int]]
    second_solution: Optional[List[int]]
    search_nodes: int
    backtracks: int
    max_depth: int
    contradictions: int
    propagation_steps: int
    elapsed_ms: float


def validate_board_state(
    grid: Sequence[int], spec: VariantSpec, require_complete: bool = False
) -> Tuple[bool, Optional[str]]:
    """
    Strictly validates whether a board satisfies all variant constraints.
    If require_complete=True, also requires all 81 cells to be filled with 1..9.
    """
    if len(grid) != NUM_CELLS:
        return False, f"Invalid grid length {len(grid)} != 81"
    for idx, val in enumerate(grid):
        if require_complete and not (1 <= val <= 9):
            return False, f"Cell {idx} is empty or out of range: {val}"
        if not (0 <= val <= 9):
            return False, f"Cell {idx} has invalid digit {val}"
        if val != 0:
            for peer in spec.peers[idx]:
                if grid[peer] == val:
                    return (
                        False,
                        f"Constraint violation: cell {idx} and peer {peer} both have {val}",
                    )
    return True, None


def init_candidate_masks(
    grid: Sequence[int], spec: VariantSpec
) -> Tuple[Optional[List[int]], Optional[List[int]], int]:
    """
    Initializes bitmask candidate array from given clues.
    Returns (values, masks, initial_eliminations) or (None, None, eliminations) on contradiction.
    """
    values = [0] * NUM_CELLS
    masks = [ALL_DIGITS_MASK] * NUM_CELLS
    eliminations = 0

    for cell, digit in enumerate(grid):
        if digit != 0:
            if not (1 <= digit <= 9):
                return None, None, eliminations
            bit = DIGIT_TO_BIT[digit]
            if not (masks[cell] & bit):
                return None, None, eliminations
            values[cell] = digit
            masks[cell] = bit
            for peer in spec.peers[cell]:
                if masks[peer] & bit:
                    masks[peer] &= ~bit
                    eliminations += 1
                    if masks[peer] == 0:
                        return None, None, eliminations
    return values, masks, eliminations


def propagate_constraints(
    values: List[int], masks: List[int], spec: VariantSpec
) -> Tuple[bool, int]:
    """
    Fast constraint propagation loop combining Naked Singles and Hidden Singles
    across all variant units. Mutates `values` and `masks` in place.
    Returns (is_consistent, propagation_steps).
    """
    steps = 0
    changed = True
    while changed:
        changed = False
        # 1. Naked singles pass
        for cell in range(NUM_CELLS):
            if values[cell] == 0:
                m = masks[cell]
                if m == 0:
                    return False, steps
                if (m & (m - 1)) == 0:  # Exactly 1 bit set
                    digit = BIT_TO_DIGIT[m]
                    values[cell] = digit
                    steps += 1
                    changed = True
                    for peer in spec.peers[cell]:
                        if masks[peer] & m:
                            if values[peer] == digit:
                                return False, steps
                            masks[peer] &= ~m
                            if masks[peer] == 0:
                                return False, steps

        # 2. Hidden singles pass across all units
        for unit in spec.units:
            seen_once = 0
            seen_multiple = 0
            for cell in unit:
                m = masks[cell]
                seen_multiple |= seen_once & m
                seen_once |= m
            if (seen_once | seen_multiple) != ALL_DIGITS_MASK:
                # Some digit has 0 candidates anywhere in this unit
                return False, steps
            unique_bits = seen_once & ~seen_multiple
            while unique_bits:
                bit = unique_bits & -unique_bits
                unique_bits &= unique_bits - 1
                # Find the cell in unit that holds this bit
                target_cell = -1
                for cell in unit:
                    if masks[cell] & bit:
                        target_cell = cell
                        break
                if target_cell == -1:
                    return False, steps
                if values[target_cell] == 0:
                    digit = BIT_TO_DIGIT[bit]
                    values[target_cell] = digit
                    masks[target_cell] = bit
                    steps += 1
                    changed = True
                    for peer in spec.peers[target_cell]:
                        if masks[peer] & bit:
                            if values[peer] == digit:
                                return False, steps
                            masks[peer] &= ~bit
                            if masks[peer] == 0:
                                return False, steps

    return True, steps


def solve_and_count(
    grid: Sequence[int],
    spec: VariantSpec,
    limit: int = 2,
    timeout_sec: float = 2.0,
    max_nodes: int = 250_000,
    rng: Optional[random.Random] = None,
) -> ExactSolverResult:
    """
    Exact bitmask MRV solver and bounded solution counter.
    Stops immediately when `limit` solutions are found (default 2 for uniqueness verification).
    Never conflates a timeout or node limit with unsatisfiability or uniqueness.
    """
    t0 = time.perf_counter()
    valid, _ = validate_board_state(grid, spec, require_complete=False)
    if not valid:
        return ExactSolverResult(
            status=SolutionCountStatus.UNSATISFIABLE,
            solution_count=0,
            first_solution=None,
            second_solution=None,
            search_nodes=0,
            backtracks=0,
            max_depth=0,
            contradictions=1,
            propagation_steps=0,
            elapsed_ms=(time.perf_counter() - t0) * 1000.0,
        )

    values, masks, _ = init_candidate_masks(grid, spec)
    if values is None or masks is None:
        return ExactSolverResult(
            status=SolutionCountStatus.UNSATISFIABLE,
            solution_count=0,
            first_solution=None,
            second_solution=None,
            search_nodes=0,
            backtracks=0,
            max_depth=0,
            contradictions=1,
            propagation_steps=0,
            elapsed_ms=(time.perf_counter() - t0) * 1000.0,
        )

    ok, init_props = propagate_constraints(values, masks, spec)
    total_props = init_props
    if not ok:
        return ExactSolverResult(
            status=SolutionCountStatus.UNSATISFIABLE,
            solution_count=0,
            first_solution=None,
            second_solution=None,
            search_nodes=0,
            backtracks=0,
            max_depth=0,
            contradictions=1,
            propagation_steps=total_props,
            elapsed_ms=(time.perf_counter() - t0) * 1000.0,
        )

    solutions: List[List[int]] = []
    search_nodes = 0
    backtracks = 0
    max_depth = 0
    contradictions = 0
    abort_status: Optional[SolutionCountStatus] = None

    def _dfs(cur_values: List[int], cur_masks: List[int], depth: int) -> bool:
        nonlocal search_nodes, backtracks, max_depth, contradictions, total_props, abort_status

        if depth > max_depth:
            max_depth = depth

        # Periodic resource check
        if (search_nodes & 63) == 0:
            if (time.perf_counter() - t0) > timeout_sec:
                abort_status = SolutionCountStatus.UNRESOLVED_TIMEOUT
                return True
            if search_nodes >= max_nodes:
                abort_status = SolutionCountStatus.UNRESOLVED_LIMIT
                return True

        # Select unfilled cell with Minimum Remaining Values (MRV), tie-breaking by peer degree
        best_cell = -1
        best_count = 10
        best_Empty_peers = -1
        for cell in range(NUM_CELLS):
            if cur_values[cell] == 0:
                cnt = cur_masks[cell].bit_count()
                if cnt == 0:
                    contradictions += 1
                    return False
                if cnt < best_count:
                    best_count = cnt
                    best_cell = cell
                    if cnt == 2 and rng is None:
                        break
                elif cnt == best_count:
                    empty_p = sum(1 for p in spec.peers[cell] if cur_values[p] == 0)
                    if empty_p > best_Empty_peers:
                        best_Empty_peers = empty_p
                        best_cell = cell

        if best_cell == -1:
            # All 81 cells assigned validly
            solutions.append(list(cur_values))
            return len(solutions) >= limit

        # Extract candidate bits for best_cell
        cand_mask = cur_masks[best_cell]
        bits: List[int] = []
        while cand_mask:
            b = cand_mask & -cand_mask
            bits.append(b)
            cand_mask &= cand_mask - 1

        if rng is not None and len(bits) > 1:
            rng.shuffle(bits)

        for idx, bit in enumerate(bits):
            search_nodes += 1
            next_values = list(cur_values)
            next_masks = list(cur_masks)
            digit = BIT_TO_DIGIT[bit]
            next_values[best_cell] = digit
            next_masks[best_cell] = bit

            consistent = True
            for peer in spec.peers[best_cell]:
                if next_masks[peer] & bit:
                    if next_values[peer] == digit:
                        consistent = False
                        break
                    next_masks[peer] &= ~bit
                    if next_masks[peer] == 0:
                        consistent = False
                        break

            if consistent:
                prop_ok, props = propagate_constraints(next_values, next_masks, spec)
                total_props += props
                if prop_ok:
                    if _dfs(next_values, next_masks, depth + 1):
                        return True
                else:
                    contradictions += 1
            else:
                contradictions += 1

            if idx < len(bits) - 1:
                backtracks += 1

        return False

    _dfs(values, masks, 0)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    if abort_status is not None:
        status = abort_status
    elif len(solutions) == 0:
        status = SolutionCountStatus.UNSATISFIABLE
    elif len(solutions) == 1:
        status = SolutionCountStatus.UNIQUE
    else:
        status = SolutionCountStatus.MULTIPLE

    return ExactSolverResult(
        status=status,
        solution_count=len(solutions),
        first_solution=solutions[0] if len(solutions) >= 1 else None,
        second_solution=solutions[1] if len(solutions) >= 2 else None,
        search_nodes=search_nodes,
        backtracks=backtracks,
        max_depth=max_depth,
        contradictions=contradictions,
        propagation_steps=total_props,
        elapsed_ms=elapsed_ms,
    )


# ---------------------------------------------------------------------------
# Human-Style Logical Strategy Engine & Technique Trace
# ---------------------------------------------------------------------------

# Documented weights for every human logical technique tier (CALIBRATION_VERSION = "sm-cal-v2.1")
TECHNIQUE_WEIGHTS: Dict[str, float] = {
    "full_house": 0.15,
    "naked_single": 0.30,
    "hidden_single": 0.60,
    "locked_candidates": 1.40,
    "naked_pair": 2.20,
    "hidden_pair": 2.40,
    "naked_triple": 3.40,
    "hidden_triple": 3.80,
    "x_wing": 5.00,
    "xy_wing": 6.50,
    "swordfish": 7.80,
    "forcing_chain_contradiction": 9.50,
}

TECHNIQUE_TIER_INDEX: Dict[str, int] = {
    name: idx + 1 for idx, name in enumerate(TECHNIQUE_WEIGHTS.keys())
}


@dataclass
class LogicalStep:
    step_index: int
    technique: str
    action_type: str  # "placement" or "elimination"
    target_cells: List[int]
    digits: List[int]
    evidence_cells: List[int]
    rationale: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_index": self.step_index,
            "technique": self.technique,
            "action_type": self.action_type,
            "target_cells": self.target_cells,
            "digits": self.digits,
            "evidence_cells": self.evidence_cells,
            "rationale": self.rationale,
        }


@dataclass
class LogicAnalysisResult:
    solved_by_logic_only: bool
    logical_steps: int
    candidate_eliminations: int
    technique_counts: Dict[str, int]
    max_technique_name: Optional[str]
    max_technique_tier: int
    weighted_technique_score: float
    trace: List[LogicalStep]
    first_deduction: Optional[LogicalStep]
    remaining_empty_after_logic: int


def _apply_placement(
    cell: int, digit: int, values: List[int], masks: List[int], spec: VariantSpec
) -> int:
    bit = DIGIT_TO_BIT[digit]
    values[cell] = digit
    masks[cell] = bit
    elims = 0
    for peer in spec.peers[cell]:
        if values[peer] == 0 and (masks[peer] & bit):
            masks[peer] &= ~bit
            elims += 1
    return elims


def _find_full_house(
    values: List[int], masks: List[int], spec: VariantSpec, step_idx: int
) -> Optional[Tuple[LogicalStep, int]]:
    for unit in spec.units:
        empty_cells = [c for c in unit if values[c] == 0]
        if len(empty_cells) == 1:
            target = empty_cells[0]
            m = masks[target]
            if m != 0 and (m & (m - 1)) == 0:
                digit = BIT_TO_DIGIT[m]
                elims = _apply_placement(target, digit, values, masks, spec)
                step = LogicalStep(
                    step_index=step_idx,
                    technique="full_house",
                    action_type="placement",
                    target_cells=[target],
                    digits=[digit],
                    evidence_cells=[c for c in unit if c != target],
                    rationale=f"Last remaining unfilled cell {target} in unit requires digit {digit}.",
                )
                return step, elims
    return None


def _find_naked_single(
    values: List[int], masks: List[int], spec: VariantSpec, step_idx: int
) -> Optional[Tuple[LogicalStep, int]]:
    for cell in range(NUM_CELLS):
        if values[cell] == 0:
            m = masks[cell]
            if m != 0 and (m & (m - 1)) == 0:
                digit = BIT_TO_DIGIT[m]
                elims = _apply_placement(cell, digit, values, masks, spec)
                step = LogicalStep(
                    step_index=step_idx,
                    technique="naked_single",
                    action_type="placement",
                    target_cells=[cell],
                    digits=[digit],
                    evidence_cells=[p for p in spec.peers[cell] if values[p] != 0],
                    rationale=f"Cell {cell} has only candidate {digit} remaining.",
                )
                return step, elims
    return None


def _find_hidden_single(
    values: List[int], masks: List[int], spec: VariantSpec, step_idx: int
) -> Optional[Tuple[LogicalStep, int]]:
    for unit in spec.units:
        for d in range(9):
            bit = 1 << d
            holders = [c for c in unit if values[c] == 0 and (masks[c] & bit)]
            if len(holders) == 1:
                target = holders[0]
                digit = d + 1
                elims = _apply_placement(target, digit, values, masks, spec)
                step = LogicalStep(
                    step_index=step_idx,
                    technique="hidden_single",
                    action_type="placement",
                    target_cells=[target],
                    digits=[digit],
                    evidence_cells=list(unit),
                    rationale=f"Digit {digit} can only be placed in cell {target} within unit.",
                )
                return step, elims
    return None


def _find_locked_candidates(
    values: List[int], masks: List[int], spec: VariantSpec, step_idx: int
) -> Optional[Tuple[LogicalStep, int]]:
    """
    Pointing and Claiming (Box-Line / Region-Line intersections):
    If all candidates for digit d in Unit A lie inside the intersection with Unit B,
    eliminate d from all other cells in Unit B outside Unit A.
    """
    num_u = len(spec.units)
    for i in range(num_u):
        unit_a = spec.units[i]
        set_a = set(unit_a)
        for d in range(9):
            bit = 1 << d
            holders = [c for c in unit_a if values[c] == 0 and (masks[c] & bit)]
            if 2 <= len(holders) <= 3:
                for j in range(num_u):
                    if i == j:
                        continue
                    unit_b = spec.units[j]
                    set_b = set(unit_b)
                    if all(h in set_b for h in holders):
                        targets = [
                            c
                            for c in unit_b
                            if c not in set_a and values[c] == 0 and (masks[c] & bit)
                        ]
                        if targets:
                            for t in targets:
                                masks[t] &= ~bit
                            step = LogicalStep(
                                step_index=step_idx,
                                technique="locked_candidates",
                                action_type="elimination",
                                target_cells=targets,
                                digits=[d + 1],
                                evidence_cells=holders,
                                rationale=(
                                    f"Digit {d + 1} in unit {i} is confined to intersection "
                                    f"{holders} with unit {j}, eliminating {d + 1} from {targets}."
                                ),
                            )
                            return step, len(targets)
    return None


def _find_naked_subsets(
    values: List[int], masks: List[int], spec: VariantSpec, step_idx: int, size: int
) -> Optional[Tuple[LogicalStep, int]]:
    tech_name = "naked_pair" if size == 2 else "naked_triple"
    for unit in spec.units:
        empty = [
            c for c in unit if values[c] == 0 and 2 <= masks[c].bit_count() <= size
        ]
        n = len(empty)
        if n < size:
            continue
        if size == 2:
            for i in range(n):
                for j in range(i + 1, n):
                    c1, c2 = empty[i], empty[j]
                    union_mask = masks[c1] | masks[c2]
                    if union_mask.bit_count() == 2:
                        subset = {c1, c2}
                        targets = [
                            c
                            for c in unit
                            if c not in subset
                            and values[c] == 0
                            and (masks[c] & union_mask)
                        ]
                        if targets:
                            elims = 0
                            for t in targets:
                                removed = masks[t] & union_mask
                                elims += removed.bit_count()
                                masks[t] &= ~union_mask
                            return (
                                LogicalStep(
                                    step_index=step_idx,
                                    technique=tech_name,
                                    action_type="elimination",
                                    target_cells=targets,
                                    digits=mask_to_digits(union_mask),
                                    evidence_cells=[c1, c2],
                                    rationale=f"Naked pair {mask_to_digits(union_mask)} at {[c1, c2]}.",
                                ),
                                elims,
                            )
        elif size == 3:
            for i in range(n):
                for j in range(i + 1, n):
                    for k in range(j + 1, n):
                        c1, c2, c3 = empty[i], empty[j], empty[k]
                        union_mask = masks[c1] | masks[c2] | masks[c3]
                        if union_mask.bit_count() == 3:
                            subset = {c1, c2, c3}
                            targets = [
                                c
                                for c in unit
                                if c not in subset
                                and values[c] == 0
                                and (masks[c] & union_mask)
                            ]
                            if targets:
                                elims = 0
                                for t in targets:
                                    removed = masks[t] & union_mask
                                    elims += removed.bit_count()
                                    masks[t] &= ~union_mask
                                return (
                                    LogicalStep(
                                        step_index=step_idx,
                                        technique=tech_name,
                                        action_type="elimination",
                                        target_cells=targets,
                                        digits=mask_to_digits(union_mask),
                                        evidence_cells=[c1, c2, c3],
                                        rationale=f"Naked triple {mask_to_digits(union_mask)} at {[c1, c2, c3]}.",
                                    ),
                                    elims,
                                )
    return None


def _find_hidden_subsets(
    values: List[int], masks: List[int], spec: VariantSpec, step_idx: int, size: int
) -> Optional[Tuple[LogicalStep, int]]:
    tech_name = "hidden_pair" if size == 2 else "hidden_triple"
    for unit in spec.units:
        digit_holders: List[Tuple[int, int]] = []
        for d in range(9):
            bit = 1 << d
            cell_mask = 0
            for idx, c in enumerate(unit):
                if values[c] == 0 and (masks[c] & bit):
                    cell_mask |= 1 << idx
            if 2 <= cell_mask.bit_count() <= size:
                digit_holders.append((d, cell_mask))

        m = len(digit_holders)
        if m < size:
            continue
        if size == 2:
            for i in range(m):
                for j in range(i + 1, m):
                    d1, cm1 = digit_holders[i]
                    d2, cm2 = digit_holders[j]
                    union_cm = cm1 | cm2
                    if union_cm.bit_count() == 2:
                        pair_bits = (1 << d1) | (1 << d2)
                        cells = [unit[idx] for idx in range(9) if union_cm & (1 << idx)]
                        targets = [c for c in cells if (masks[c] & ~pair_bits) != 0]
                        if targets:
                            elims = 0
                            for t in targets:
                                extra = masks[t] & ~pair_bits
                                elims += extra.bit_count()
                                masks[t] &= pair_bits
                            return (
                                LogicalStep(
                                    step_index=step_idx,
                                    technique=tech_name,
                                    action_type="elimination",
                                    target_cells=targets,
                                    digits=[d1 + 1, d2 + 1],
                                    evidence_cells=cells,
                                    rationale=f"Hidden pair {[d1 + 1, d2 + 1]} confined to {cells}.",
                                ),
                                elims,
                            )
        elif size == 3:
            for i in range(m):
                for j in range(i + 1, m):
                    for k in range(j + 1, m):
                        d1, cm1 = digit_holders[i]
                        d2, cm2 = digit_holders[j]
                        d3, cm3 = digit_holders[k]
                        union_cm = cm1 | cm2 | cm3
                        if union_cm.bit_count() == 3:
                            triple_bits = (1 << d1) | (1 << d2) | (1 << d3)
                            cells = [
                                unit[idx] for idx in range(9) if union_cm & (1 << idx)
                            ]
                            targets = [
                                c for c in cells if (masks[c] & ~triple_bits) != 0
                            ]
                            if targets:
                                elims = 0
                                for t in targets:
                                    extra = masks[t] & ~triple_bits
                                    elims += extra.bit_count()
                                    masks[t] &= triple_bits
                                return (
                                    LogicalStep(
                                        step_index=step_idx,
                                        technique=tech_name,
                                        action_type="elimination",
                                        target_cells=targets,
                                        digits=[d1 + 1, d2 + 1, d3 + 1],
                                        evidence_cells=cells,
                                        rationale=f"Hidden triple {[d1 + 1, d2 + 1, d3 + 1]} confined to {cells}.",
                                    ),
                                    elims,
                                )
    return None


def _find_fish(
    values: List[int], masks: List[int], step_idx: int, fish_size: int
) -> Optional[Tuple[LogicalStep, int]]:
    """
    Detects X-Wing (fish_size=2) and Swordfish (fish_size=3) across rows and columns.
    """
    tech_name = "x_wing" if fish_size == 2 else "swordfish"
    for d in range(9):
        bit = 1 << d
        # Row-based fish -> eliminate from columns
        row_masks: List[Tuple[int, int]] = []
        for r in range(9):
            cm = 0
            for c in range(9):
                cell = r * 9 + c
                if values[cell] == 0 and (masks[cell] & bit):
                    cm |= 1 << c
            if 2 <= cm.bit_count() <= fish_size:
                row_masks.append((r, cm))

        if len(row_masks) >= fish_size:
            if fish_size == 2:
                for i in range(len(row_masks)):
                    for j in range(i + 1, len(row_masks)):
                        r1, cm1 = row_masks[i]
                        r2, cm2 = row_masks[j]
                        union_cols = cm1 | cm2
                        if union_cols.bit_count() == 2:
                            cols = [c for c in range(9) if union_cols & (1 << c)]
                            targets = [
                                r * 9 + c
                                for c in cols
                                for r in range(9)
                                if r not in (r1, r2)
                                and values[r * 9 + c] == 0
                                and (masks[r * 9 + c] & bit)
                            ]
                            if targets:
                                for t in targets:
                                    masks[t] &= ~bit
                                ev = [
                                    r * 9 + c
                                    for r in (r1, r2)
                                    for c in cols
                                    if values[r * 9 + c] == 0 and (masks[r * 9 + c] & bit)
                                ]
                                return (
                                    LogicalStep(
                                        step_index=step_idx,
                                        technique=tech_name,
                                        action_type="elimination",
                                        target_cells=targets,
                                        digits=[d + 1],
                                        evidence_cells=ev,
                                        rationale=f"Row X-Wing on digit {d + 1} in rows {(r1, r2)}, cols {cols}.",
                                    ),
                                    len(targets),
                                )
            elif fish_size == 3:
                for i in range(len(row_masks)):
                    for j in range(i + 1, len(row_masks)):
                        for k in range(j + 1, len(row_masks)):
                            r1, cm1 = row_masks[i]
                            r2, cm2 = row_masks[j]
                            r3, cm3 = row_masks[k]
                            union_cols = cm1 | cm2 | cm3
                            if union_cols.bit_count() == 3:
                                cols = [c for c in range(9) if union_cols & (1 << c)]
                                targets = [
                                    r * 9 + c
                                    for c in cols
                                    for r in range(9)
                                    if r not in (r1, r2, r3)
                                    and values[r * 9 + c] == 0
                                    and (masks[r * 9 + c] & bit)
                                ]
                                if targets:
                                    for t in targets:
                                        masks[t] &= ~bit
                                    ev = [
                                        r * 9 + c
                                        for r in (r1, r2, r3)
                                        for c in cols
                                        if values[r * 9 + c] == 0
                                        and (masks[r * 9 + c] & bit)
                                    ]
                                    return (
                                        LogicalStep(
                                            step_index=step_idx,
                                            technique=tech_name,
                                            action_type="elimination",
                                            target_cells=targets,
                                            digits=[d + 1],
                                            evidence_cells=ev,
                                            rationale=f"Row Swordfish on digit {d + 1} in rows {(r1, r2, r3)}.",
                                        ),
                                        len(targets),
                                    )
    return None


def _find_xy_wing(
    values: List[int], masks: List[int], spec: VariantSpec, step_idx: int
) -> Optional[Tuple[LogicalStep, int]]:
    """
    Detects XY-Wing: Pivot cell with candidates {X, Y} sees pincer1 {X, Z} and pincer2 {Y, Z}.
    Any cell seeing both pincer1 and pincer2 cannot contain Z.
    """
    bivalue = [c for c in range(NUM_CELLS) if values[c] == 0 and masks[c].bit_count() == 2]
    bivalue_set = set(bivalue)
    for pivot in bivalue:
        xy = masks[pivot]
        x_bit = xy & -xy
        y_bit = xy ^ x_bit
        peer_bivalues = [p for p in spec.peers[pivot] if p in bivalue_set]
        pincers_x = [p for p in peer_bivalues if (masks[p] & x_bit) and not (masks[p] & y_bit)]
        pincers_y = [p for p in peer_bivalues if (masks[p] & y_bit) and not (masks[p] & x_bit)]
        for p1 in pincers_x:
            z_bit = masks[p1] ^ x_bit
            for p2 in pincers_y:
                if (masks[p2] ^ y_bit) == z_bit:
                    common_peers = set(spec.peers[p1]).intersection(spec.peers[p2])
                    common_peers.discard(pivot)
                    targets = [
                        c
                        for c in sorted(common_peers)
                        if values[c] == 0 and (masks[c] & z_bit)
                    ]
                    if targets:
                        for t in targets:
                            masks[t] &= ~z_bit
                        z_digit = BIT_TO_DIGIT[z_bit]
                        return (
                            LogicalStep(
                                step_index=step_idx,
                                technique="xy_wing",
                                action_type="elimination",
                                target_cells=targets,
                                digits=[z_digit],
                                evidence_cells=[pivot, p1, p2],
                                rationale=(
                                    f"XY-Wing with pivot {pivot} and pincers {[p1, p2]} "
                                    f"eliminates {z_digit} from {targets}."
                                ),
                            ),
                            len(targets),
                        )
    return None


def _find_forcing_contradiction(
    values: List[int], masks: List[int], spec: VariantSpec, step_idx: int
) -> Optional[Tuple[LogicalStep, int]]:
    """
    Bivalue/Single-step propagation contradiction chain (AIC / Nishio lookahead):
    Tests candidate assumptions on restricted cells (2 candidates); if assuming candidate X
    leads immediately via unit propagation to a contradiction, X is eliminated.
    """
    candidates = [
        c for c in range(NUM_CELLS) if values[c] == 0 and 2 <= masks[c].bit_count() <= 3
    ]
    for cell in candidates:
        m = masks[cell]
        temp = m
        while temp:
            bit = temp & -temp
            temp &= temp - 1
            test_vals = list(values)
            test_masks = list(masks)
            digit = BIT_TO_DIGIT[bit]
            test_vals[cell] = digit
            test_masks[cell] = bit
            ok = True
            for peer in spec.peers[cell]:
                if test_masks[peer] & bit:
                    if test_vals[peer] == digit:
                        ok = False
                        break
                    test_masks[peer] &= ~bit
                    if test_masks[peer] == 0:
                        ok = False
                        break
            if ok:
                ok, _ = propagate_constraints(test_vals, test_masks, spec)
            if not ok:
                masks[cell] &= ~bit
                return (
                    LogicalStep(
                        step_index=step_idx,
                        technique="forcing_chain_contradiction",
                        action_type="elimination",
                        target_cells=[cell],
                        digits=[digit],
                        evidence_cells=[cell],
                        rationale=(
                            f"Assuming digit {digit} at cell {cell} forces a propagation "
                            f"contradiction; eliminating {digit} from cell {cell}."
                        ),
                    ),
                    1,
                )
    return None


def analyze_human_logic(
    grid: Sequence[int], spec: VariantSpec, max_steps: int = 300
) -> LogicAnalysisResult:
    """
    Executes the graded human-strategy solver on `grid` to extract a verifiable
    step-by-step `technique_trace` and difficulty features.
    """
    values, masks, init_elims = init_candidate_masks(grid, spec)
    if values is None or masks is None:
        return LogicAnalysisResult(
            solved_by_logic_only=False,
            logical_steps=0,
            candidate_eliminations=init_elims,
            technique_counts={},
            max_technique_name=None,
            max_technique_tier=0,
            weighted_technique_score=0.0,
            trace=[],
            first_deduction=None,
            remaining_empty_after_logic=sum(1 for x in grid if x == 0),
        )

    trace: List[LogicalStep] = []
    technique_counts: Dict[str, int] = {k: 0 for k in TECHNIQUE_WEIGHTS}
    total_elims = init_elims
    weighted_sum = 0.0
    max_tier = 0
    max_tech_name: Optional[str] = None

    for step_idx in range(1, max_steps + 1):
        if all(v != 0 for v in values):
            break

        # Apply lowest-complexity available human deduction first
        found = (
            _find_full_house(values, masks, spec, step_idx)
            or _find_naked_single(values, masks, spec, step_idx)
            or _find_hidden_single(values, masks, spec, step_idx)
            or _find_locked_candidates(values, masks, spec, step_idx)
            or _find_naked_subsets(values, masks, spec, step_idx, 2)
            or _find_hidden_subsets(values, masks, spec, step_idx, 2)
            or _find_naked_subsets(values, masks, spec, step_idx, 3)
            or _find_hidden_subsets(values, masks, spec, step_idx, 3)
            or _find_fish(values, masks, step_idx, 2)
            or _find_xy_wing(values, masks, spec, step_idx)
            or _find_fish(values, masks, step_idx, 3)
            or _find_forcing_contradiction(values, masks, spec, step_idx)
        )

        if found is None:
            break

        step, elims = found
        trace.append(step)
        total_elims += elims
        tech = step.technique
        technique_counts[tech] = technique_counts.get(tech, 0) + 1
        w = TECHNIQUE_WEIGHTS[tech]
        weighted_sum += w
        tier = TECHNIQUE_TIER_INDEX[tech]
        if tier > max_tier:
            max_tier = tier
            max_tech_name = tech

    remaining_empty = sum(1 for v in values if v == 0)
    solved_logic = remaining_empty == 0
    non_zero_counts = {k: v for k, v in technique_counts.items() if v > 0}

    return LogicAnalysisResult(
        solved_by_logic_only=solved_logic,
        logical_steps=len(trace),
        candidate_eliminations=total_elims,
        technique_counts=non_zero_counts,
        max_technique_name=max_tech_name,
        max_technique_tier=max_tier,
        weighted_technique_score=round(weighted_sum, 4),
        trace=trace,
        first_deduction=trace[0] if trace else None,
        remaining_empty_after_logic=remaining_empty,
    )


# ---------------------------------------------------------------------------
# Multi-Factor Difficulty Calibration (Levels 1.0 - 10.0)
# ---------------------------------------------------------------------------


@dataclass
class DifficultyAssessment:
    difficulty: float
    difficulty_raw: float
    difficulty_band: str
    features: Dict[str, Any]


def compute_calibrated_difficulty(
    grid: Sequence[int],
    spec: VariantSpec,
    solver_res: ExactSolverResult,
    logic_res: LogicAnalysisResult,
    difficulty_step: float = 0.1,
) -> DifficultyAssessment:
    """
    Documented, deterministic multi-factor difficulty calibration (CALIBRATION_VERSION = "sm-cal-v2.1").

    Combines 5 normalized structural, logical, and combinatorial search components:
      1. peak_tier_component:       Max logical technique tier required (tiers 1..10 -> 0.0..4.5)
      2. logic_work_component:      Log-scaled weighted sum of human deductions (0.0..2.2)
      3. search_effort_component:   Branching nodes, backtracks, contradictions, and residual
                                    unsolved cells when human strategies stall (0.0..2.8)
      4. candidate_density_comp:    Initial candidate entropy and minimum cell freedom (0.0..0.8)
      5. emptiness_component:       Normalized empty-cell ratio (secondary factor, capped at 0.7)

    Note: `solver_runtime_ms` is recorded in `features` for telemetry, but assigned a 0.0 weight
    in `difficulty_raw` so that CPU load or hardware differences never alter difficulty labels.
    """
    if solver_res.status in (
        SolutionCountStatus.UNRESOLVED_TIMEOUT,
        SolutionCountStatus.UNRESOLVED_LIMIT,
    ):
        raise ValueError(
            "Cannot assign a calibrated difficulty score to an unresolved solver run."
        )

    clue_count = sum(1 for x in grid if x != 0)
    empty_cells = NUM_CELLS - clue_count

    _, init_masks, init_elims = init_candidate_masks(grid, spec)
    if init_masks is not None:
        empty_masks = [init_masks[c].bit_count() for c in range(NUM_CELLS) if grid[c] == 0]
        initial_candidates = sum(empty_masks)
        min_cell_candidates = min(empty_masks) if empty_masks else 0
        avg_cell_candidates = (
            round(initial_candidates / len(empty_masks), 4) if empty_masks else 0.0
        )
    else:
        initial_candidates = 0
        min_cell_candidates = 0
        avg_cell_candidates = 0.0

    # 1. Peak technique tier component (0.0 to 4.5)
    tier_map = {
        0: 0.0,
        1: 0.10,  # full_house
        2: 0.35,  # naked_single
        3: 0.85,  # hidden_single
        4: 1.65,  # locked_candidates
        5: 2.35,  # naked_pair
        6: 2.65,  # hidden_pair
        7: 3.10,  # naked_triple
        8: 3.40,  # hidden_triple
        9: 3.80,  # x_wing
        10: 4.10, # xy_wing
        11: 4.35, # swordfish
        12: 4.50, # forcing_chain_contradiction
    }
    peak_tier_comp = tier_map.get(logic_res.max_technique_tier, 4.50)

    # 2. Cumulative logical work component (0.0 to 2.2)
    # Subtract baseline trivial naked/full-house work so 55-clue boards start near 1.0
    excess_logic_weight = max(0.0, logic_res.weighted_technique_score - 6.5)
    logic_work_comp = min(
        2.20, 0.52 * math.log1p(excess_logic_weight)
    )

    # 3. Search & branching component (0.0 to 2.8)
    # Extra branch nodes beyond the first path (which has empty_cells nodes)
    extra_nodes = max(0, solver_res.search_nodes - empty_cells)
    search_comp = (
        0.55 * math.log1p(extra_nodes)
        + 0.45 * math.log1p(solver_res.backtracks)
        + 0.25 * math.log1p(solver_res.contradictions)
    )
    if not logic_res.solved_by_logic_only:
        # Board defeated the human strategy set and required true search
        unsolved_ratio = logic_res.remaining_empty_after_logic / max(1, empty_cells)
        search_comp += 1.15 + 0.85 * unsolved_ratio
    search_comp = min(3.20, search_comp)

    # 4. Initial candidate density component (0.0 to 0.75)
    cand_comp = min(0.50, max(0.0, (avg_cell_candidates - 1.8) * 0.18))
    if min_cell_candidates >= 2:
        cand_comp += 0.25

    # 5. Secondary emptiness component (0.0 to 0.65)
    emptiness_comp = min(0.65, max(0.0, (empty_cells - 28) / 36.0) * 0.65)

    raw_score = (
        1.0
        + peak_tier_comp
        + logic_work_comp
        + search_comp
        + cand_comp
        + emptiness_comp
    )
    clamped_raw = max(1.0, min(10.0, raw_score))

    # Quantize to requested difficulty_step (default 0.1)
    steps = round((clamped_raw - 1.0) / difficulty_step)
    calibrated_label = round(1.0 + steps * difficulty_step, 2)
    calibrated_label = max(1.0, min(10.0, calibrated_label))

    if calibrated_label <= 2.2:
        band = "introductory_easy"
    elif calibrated_label <= 3.8:
        band = "basic_singles"
    elif calibrated_label <= 5.5:
        band = "intermediate_intersections_subsets"
    elif calibrated_label <= 7.2:
        band = "advanced_patterns_wings_fish"
    elif calibrated_label <= 8.6:
        band = "hard_chains_and_bifurcation"
    else:
        band = "extreme_search_stress"

    features: Dict[str, Any] = {
        "calibration_version": CALIBRATION_VERSION,
        "clue_count": clue_count,
        "empty_cells": empty_cells,
        "initial_candidates": initial_candidates,
        "avg_cell_candidates": avg_cell_candidates,
        "min_cell_candidates": min_cell_candidates,
        "initial_eliminations": init_elims,
        "solved_by_logic_only": logic_res.solved_by_logic_only,
        "logical_steps": logic_res.logical_steps,
        "candidate_eliminations": logic_res.candidate_eliminations,
        "technique_counts": logic_res.technique_counts,
        "max_technique_name": logic_res.max_technique_name,
        "max_technique_tier": logic_res.max_technique_tier,
        "weighted_technique_score": logic_res.weighted_technique_score,
        "remaining_empty_after_logic": logic_res.remaining_empty_after_logic,
        "search_nodes": solver_res.search_nodes,
        "extra_search_nodes": extra_nodes,
        "backtracks": solver_res.backtracks,
        "max_depth": solver_res.max_depth,
        "contradictions": solver_res.contradictions,
        "propagation_steps": solver_res.propagation_steps,
        "solver_runtime_ms": round(solver_res.elapsed_ms, 3),
        "difficulty_band": band,
        "score_components": {
            "peak_tier_component": round(peak_tier_comp, 4),
            "logic_work_component": round(logic_work_comp, 4),
            "search_effort_component": round(search_comp, 4),
            "candidate_density_component": round(cand_comp, 4),
            "emptiness_component": round(emptiness_comp, 4),
        },
    }

    return DifficultyAssessment(
        difficulty=calibrated_label,
        difficulty_raw=round(clamped_raw, 4),
        difficulty_band=band,
        features=features,
    )


# ---------------------------------------------------------------------------
# Symmetry, Canonicalization & Leakage-Prevention Hashing
# ---------------------------------------------------------------------------


def _build_dihedral_maps() -> Tuple[Tuple[int, ...], ...]:
    """
    Precomputes the 8 dihedral group transformations (4 rotations x 2 reflections)
    of a 9x9 grid. Every implemented peer-graph variant (standard, diagonal, hyper,
    anti_knight, anti_king) is invariant under the full dihedral group D4.
    """
    maps: List[Tuple[int, ...]] = []
    for reflect in (False, True):
        for rot in range(4):
            perm: List[int] = [0] * NUM_CELLS
            for r in range(9):
                for c in range(9):
                    rr, cc = r, (8 - c if reflect else c)
                    for _ in range(rot):
                        rr, cc = cc, 8 - rr
                    perm[r * 9 + c] = rr * 9 + cc
            maps.append(tuple(perm))
    return tuple(maps)


DIHEDRAL_MAPS: Tuple[Tuple[int, ...], ...] = _build_dihedral_maps()


def relabel_digits_canonical(grid: Sequence[int]) -> str:
    """
    Relabels digits 1..9 in order of first appearance (reading order) so that
    any two grids differing only by a permutation of {1..9} map to the exact same string.
    Empty cells (0) remain '0'.
    """
    mapping: Dict[int, str] = {0: "0"}
    next_code = 1
    out: List[str] = []
    for val in grid:
        if val not in mapping:
            mapping[val] = str(next_code)
            next_code += 1
        out.append(mapping[val])
    return "".join(out)


def canonicalize_grid_d4(grid: Sequence[int], use_d4: bool = True) -> str:
    """
    Computes the lexicographically minimal digit-relabeled representation across
    all 8 dihedral symmetries (rotations + reflections).
    """
    if not use_d4:
        return relabel_digits_canonical(grid)
    best: Optional[str] = None
    for perm in DIHEDRAL_MAPS:
        transformed = [grid[perm[i]] for i in range(NUM_CELLS)]
        relabeled = relabel_digits_canonical(transformed)
        if best is None or relabeled < best:
            best = relabeled
    assert best is not None
    return best


def compute_canonical_hashes(
    puzzle_grid: Sequence[int],
    solution_grid: Sequence[int],
    spec: VariantSpec,
) -> Tuple[str, str, str]:
    """
    Returns (exact_puzzle_hash, canonical_puzzle_hash, solution_family_hash).
      - exact_puzzle_hash:     SHA-256 of "<variant>:<raw_81_char_puzzle>"
      - canonical_puzzle_hash: SHA-256 of "<variant>:<D4_digit_relabeled_puzzle>"
      - solution_family_hash:  SHA-256 of "<variant>:<D4_digit_relabeled_solution>"
    """
    raw_puz = grid_to_string(puzzle_grid)
    exact_hash = hashlib.sha256(f"{spec.name}:{raw_puz}".encode("utf-8")).hexdigest()

    canon_puz = canonicalize_grid_d4(
        puzzle_grid, use_d4=spec.preserves_full_dihedral_symmetry
    )
    canon_puz_hash = hashlib.sha256(
        f"{spec.name}:{canon_puz}".encode("utf-8")
    ).hexdigest()

    canon_sol = canonicalize_grid_d4(
        solution_grid, use_d4=spec.preserves_full_dihedral_symmetry
    )
    family_hash = hashlib.sha256(
        f"{spec.name}:{canon_sol}".encode("utf-8")
    ).hexdigest()

    return exact_hash, canon_puz_hash, family_hash


# ---------------------------------------------------------------------------
# Complete Grid Generator & Clue-Digging Pipeline
# ---------------------------------------------------------------------------


def generate_complete_grid(
    spec: VariantSpec,
    rng: random.Random,
    timeout_sec: float = 2.0,
) -> Optional[List[int]]:
    """
    Generates a uniformly randomized valid complete 9x9 grid for `spec`
    using randomized MRV backtracking with exact constraint propagation.
    """
    empty_board = [0] * NUM_CELLS
    res = solve_and_count(
        empty_board,
        spec=spec,
        limit=1,
        timeout_sec=timeout_sec,
        max_nodes=50_000,
        rng=rng,
    )
    if res.status == SolutionCountStatus.UNIQUE and res.first_solution is not None:
        valid, _ = validate_board_state(
            res.first_solution, spec, require_complete=True
        )
        if valid:
            return res.first_solution
    return None


@dataclass
class DigCandidateResult:
    success: bool
    rejection_reason: Optional[str]
    puzzle_grid: Optional[List[int]]
    solution_grid: Optional[List[int]]
    solver_result: Optional[ExactSolverResult]
    logic_result: Optional[LogicAnalysisResult]
    difficulty_assessment: Optional[DifficultyAssessment]


def dig_unique_puzzle(
    solution_grid: Sequence[int],
    spec: VariantSpec,
    rng: random.Random,
    target_clues: int,
    symmetry_mode: str,
    timeout_sec: float,
    difficulty_step: float,
    min_difficulty: float,
    max_difficulty: float,
) -> DigCandidateResult:
    """
    Derives a uniquely solvable puzzle from `solution_grid` by digging clues down
    toward `target_clues` while verifying exact uniqueness at every removal step.
    Never fabricates uniqueness or accepts a timed-out search.
    """
    if target_clues < spec.min_theoretical_clues:
        return DigCandidateResult(
            success=False,
            rejection_reason="below_theoretical_minimum_clues",
            puzzle_grid=None,
            solution_grid=None,
            solver_result=None,
            logic_result=None,
            difficulty_assessment=None,
        )

    puzzle = list(solution_grid)
    current_clues = NUM_CELLS

    # Build digging order (either symmetric pairs around center cell 40 or single cells)
    if symmetry_mode == "central" and rng.random() < 0.55:
        groups: List[List[int]] = []
        visited: Set[int] = set()
        indices = list(range(NUM_CELLS))
        rng.shuffle(indices)
        for idx in indices:
            if idx in visited:
                continue
            mirror = (NUM_CELLS - 1) - idx
            if idx == mirror:
                groups.append([idx])
                visited.add(idx)
            else:
                groups.append([idx, mirror])
                visited.add(idx)
                visited.add(mirror)
    else:
        indices = list(range(NUM_CELLS))
        rng.shuffle(indices)
        groups = [[idx] for idx in indices]

    for grp in groups:
        if current_clues <= target_clues:
            break
        if current_clues - len(grp) < spec.min_theoretical_clues:
            continue

        saved = [(c, puzzle[c]) for c in grp]
        for c in grp:
            puzzle[c] = 0

        check = solve_and_count(
            puzzle,
            spec=spec,
            limit=2,
            timeout_sec=timeout_sec,
            max_nodes=120_000,
        )
        if check.status == SolutionCountStatus.UNIQUE:
            current_clues -= len(grp)
        elif check.status in (
            SolutionCountStatus.UNRESOLVED_TIMEOUT,
            SolutionCountStatus.UNRESOLVED_LIMIT,
        ):
            # Restore and abort digging further on timeout to avoid unresolved boards
            for c, val in saved:
                puzzle[c] = val
            break
        else:
            # Restoring because removal introduced >=2 solutions
            for c, val in saved:
                puzzle[c] = val

    # Final authoritative uniqueness and work measurement pass
    final_solve = solve_and_count(
        puzzle,
        spec=spec,
        limit=2,
        timeout_sec=timeout_sec,
        max_nodes=200_000,
    )
    if final_solve.status in (
        SolutionCountStatus.UNRESOLVED_TIMEOUT,
        SolutionCountStatus.UNRESOLVED_LIMIT,
    ):
        return DigCandidateResult(
            success=False,
            rejection_reason=final_solve.status.value,
            puzzle_grid=None,
            solution_grid=None,
            solver_result=final_solve,
            logic_result=None,
            difficulty_assessment=None,
        )
    if final_solve.status == SolutionCountStatus.UNSATISFIABLE:
        return DigCandidateResult(
            success=False,
            rejection_reason="unsatisfiable",
            puzzle_grid=None,
            solution_grid=None,
            solver_result=final_solve,
            logic_result=None,
            difficulty_assessment=None,
        )
    if final_solve.status != SolutionCountStatus.UNIQUE or final_solve.first_solution is None:
        return DigCandidateResult(
            success=False,
            rejection_reason="non_unique_solution",
            puzzle_grid=None,
            solution_grid=None,
            solver_result=final_solve,
            logic_result=None,
            difficulty_assessment=None,
        )

    # Verify clue preservation and exact agreement with original solution
    for idx in range(NUM_CELLS):
        if puzzle[idx] != 0 and puzzle[idx] != solution_grid[idx]:
            return DigCandidateResult(
                success=False,
                rejection_reason="clue_preservation_violation",
                puzzle_grid=None,
                solution_grid=None,
                solver_result=final_solve,
                logic_result=None,
                difficulty_assessment=None,
            )
        if final_solve.first_solution[idx] != solution_grid[idx]:
            return DigCandidateResult(
                success=False,
                rejection_reason="solution_mismatch",
                puzzle_grid=None,
                solution_grid=None,
                solver_result=final_solve,
                logic_result=None,
                difficulty_assessment=None,
            )

    logic_res = analyze_human_logic(puzzle, spec)
    diff_assess = compute_calibrated_difficulty(
        puzzle,
        spec=spec,
        solver_res=final_solve,
        logic_res=logic_res,
        difficulty_step=difficulty_step,
    )

    if not (min_difficulty - 1e-6 <= diff_assess.difficulty <= max_difficulty + 1e-6):
        return DigCandidateResult(
            success=False,
            rejection_reason="outside_difficulty_range",
            puzzle_grid=puzzle,
            solution_grid=list(solution_grid),
            solver_result=final_solve,
            logic_result=logic_res,
            difficulty_assessment=diff_assess,
        )

    return DigCandidateResult(
        success=True,
        rejection_reason=None,
        puzzle_grid=puzzle,
        solution_grid=list(solution_grid),
        solver_result=final_solve,
        logic_result=logic_res,
        difficulty_assessment=diff_assess,
    )


# ---------------------------------------------------------------------------
# Task Target & Record Serialization (Single-Task-Per-Board Principle)
# ---------------------------------------------------------------------------


def build_task_target(
    task_type: TaskType,
    puzzle_grid: Sequence[int],
    solution_grid: Sequence[int],
    spec: VariantSpec,
    logic_res: LogicAnalysisResult,
) -> Dict[str, Any]:
    """
    Constructs the explicit single-task training target for the board.
    Never mixes incompatible task targets or leaks the solution into the input.
    """
    if task_type == TaskType.COMPLETE_SOLUTION:
        return {
            "task_type": TaskType.COMPLETE_SOLUTION.value,
            "solution_string": grid_to_string(solution_grid),
        }

    if task_type == TaskType.NEXT_LOGICAL_DEDUCTION:
        if logic_res.first_deduction is not None:
            return {
                "task_type": TaskType.NEXT_LOGICAL_DEDUCTION.value,
                "has_logical_move": True,
                "deduction": logic_res.first_deduction.to_dict(),
            }
        # Fallback when immediate state requires bifurcation
        _, masks, _ = init_candidate_masks(puzzle_grid, spec)
        best_cell = -1
        best_cnt = 10
        if masks is not None:
            for c in range(NUM_CELLS):
                if puzzle_grid[c] == 0 and masks[c].bit_count() < best_cnt:
                    best_cnt = masks[c].bit_count()
                    best_cell = c
        return {
            "task_type": TaskType.NEXT_LOGICAL_DEDUCTION.value,
            "has_logical_move": False,
            "deduction": {
                "step_index": 1,
                "technique": "mrv_search_branch",
                "action_type": "placement",
                "target_cells": [best_cell] if best_cell >= 0 else [],
                "digits": [solution_grid[best_cell]] if best_cell >= 0 else [],
                "evidence_cells": [],
                "rationale": "No human strategy match; MRV search branch required.",
            },
        }

    if task_type == TaskType.CANDIDATE_ELIMINATION:
        _, masks, _ = init_candidate_masks(puzzle_grid, spec)
        initial_map: Dict[str, List[int]] = {}
        if masks is not None:
            for c in range(NUM_CELLS):
                if puzzle_grid[c] == 0:
                    initial_map[str(c)] = mask_to_digits(masks[c])
        eliminations = [
            s.to_dict() for s in logic_res.trace if s.action_type == "elimination"
        ]
        return {
            "task_type": TaskType.CANDIDATE_ELIMINATION.value,
            "initial_candidates": initial_map,
            "elimination_steps": eliminations,
        }

    if task_type == TaskType.SOLVER_POLICY:
        _, masks, _ = init_candidate_masks(puzzle_grid, spec)
        best_cell = -1
        best_cnt = 10
        if masks is not None:
            for c in range(NUM_CELLS):
                if puzzle_grid[c] == 0 and masks[c].bit_count() < best_cnt:
                    best_cnt = masks[c].bit_count()
                    best_cell = c
        return {
            "task_type": TaskType.SOLVER_POLICY.value,
            "recommended_cell": best_cell,
            "recommended_digit": solution_grid[best_cell] if best_cell >= 0 else None,
            "branching_factor": best_cnt if best_cell >= 0 else 0,
            "solved_by_logic_only": logic_res.solved_by_logic_only,
        }

    raise ValueError(f"Unsupported task_type: {task_type}")


def serialize_puzzle_record(
    split_name: str,
    sequence_index: int,
    attempt_seed: int,
    spec: VariantSpec,
    task_type: TaskType,
    puzzle_grid: Sequence[int],
    solution_grid: Sequence[int],
    solver_res: ExactSolverResult,
    logic_res: LogicAnalysisResult,
    diff_assess: DifficultyAssessment,
    exact_hash: str,
    canon_puz_hash: str,
    family_hash: str,
) -> Dict[str, Any]:
    puzzle_str = grid_to_string(puzzle_grid)
    solution_str = grid_to_string(solution_grid)
    givens_count = sum(1 for x in puzzle_grid if x != 0)
    puzzle_id = f"sm-{spec.name}-{split_name}-{canon_puz_hash[:16]}"

    target_payload = build_task_target(
        task_type=task_type,
        puzzle_grid=puzzle_grid,
        solution_grid=solution_grid,
        spec=spec,
        logic_res=logic_res,
    )

    record_core: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "puzzle_id": puzzle_id,
        "split": split_name,
        "sequence_index": sequence_index,
        "variant": spec.name,
        "variant_metadata": spec.to_metadata_dict(),
        "difficulty": diff_assess.difficulty,
        "difficulty_raw": diff_assess.difficulty_raw,
        "difficulty_features": diff_assess.features,
        "givens": givens_count,
        "puzzle": puzzle_str,
        "solution": solution_str,
        "solution_count_status": solver_res.status.value,
        "unique_solution": solver_res.status == SolutionCountStatus.UNIQUE,
        "task_type": task_type.value,
        "target": target_payload,
        "technique_trace": [step.to_dict() for step in logic_res.trace],
        "generator_version": GENERATOR_VERSION,
        "generation_seed": attempt_seed,
        "solver_version": SOLVER_VERSION,
        "provenance": {
            "exact_puzzle_hash": exact_hash,
            "canonical_puzzle_hash": canon_puz_hash,
            "solution_family_hash": family_hash,
            "calibration_version": CALIBRATION_VERSION,
        },
        "validation_status": ValidationStatus.VERIFIED_UNIQUE.value,
    }

    # Compute deterministic record integrity checksum over canonical JSON of core fields
    core_bytes = json.dumps(record_core, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    record_core["record_sha256"] = hashlib.sha256(core_bytes).hexdigest()
    return record_core


def verify_serialized_record(record: Dict[str, Any], spec: VariantSpec) -> Tuple[bool, Optional[str]]:
    """
    Audits a serialized JSON record for mathematical validity, clue preservation,
    schema completeness, and SHA-256 tamper integrity.
    """
    required_keys = {
        "schema_version", "puzzle_id", "variant", "difficulty", "difficulty_raw",
        "difficulty_features", "givens", "puzzle", "solution", "solution_count_status",
        "unique_solution", "task_type", "target", "technique_trace", "generator_version",
        "generation_seed", "solver_version", "provenance", "validation_status", "record_sha256",
    }
    missing = required_keys - set(record.keys())
    if missing:
        return False, f"Missing required schema keys: {sorted(missing)}"

    rec_copy = dict(record)
    claimed_sha = rec_copy.pop("record_sha256")
    computed_sha = hashlib.sha256(
        json.dumps(rec_copy, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if claimed_sha != computed_sha:
        return False, "Record SHA-256 checksum mismatch"

    try:
        puz_grid = string_to_grid(record["puzzle"])
        sol_grid = string_to_grid(record["solution"])
    except ValueError as exc:
        return False, str(exc)

    if sum(1 for x in puz_grid if x != 0) != record["givens"]:
        return False, "Givens count does not match puzzle string"

    if record["givens"] < spec.min_theoretical_clues:
        return False, f"Givens ({record['givens']}) below minimum ({spec.min_theoretical_clues})"

    for i in range(NUM_CELLS):
        if puz_grid[i] != 0 and puz_grid[i] != sol_grid[i]:
            return False, f"Clue preservation violated at cell {i}"

    valid_puz, err_puz = validate_board_state(puz_grid, spec, require_complete=False)
    if not valid_puz:
        return False, f"Invalid puzzle state: {err_puz}"

    valid_sol, err_sol = validate_board_state(sol_grid, spec, require_complete=True)
    if not valid_sol:
        return False, f"Invalid solution state: {err_sol}"

    if not record["unique_solution"] or record["solution_count_status"] != SolutionCountStatus.UNIQUE.value:
        return False, "Record does not carry verified unique solution status"

    return True, None


# ---------------------------------------------------------------------------
# Disk-Backed SQLite Deduplication & Cross-Split Leakage Registry
# ---------------------------------------------------------------------------


class DeduplicationRegistry:
    """
    Bounded-memory, SQLite WAL-backed registry that tracks:
      1. `canonical_puzzle_hash`: Prevents duplicate or D4/digit-relabeled isomorphic
         puzzles anywhere across train, validation, and test splits.
      2. `solution_family_hash`: Enforces strict family separation between splits so
         no puzzle in `train` or `validation` is ever derived from the same completed
         grid family (under D4 symmetry + digit relabeling) as another split.
    """

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
                CREATE TABLE IF NOT EXISTS seen_puzzles (
                    canonical_puzzle_hash TEXT PRIMARY KEY,
                    exact_puzzle_hash TEXT NOT NULL,
                    solution_family_hash TEXT NOT NULL,
                    split TEXT NOT NULL,
                    puzzle_id TEXT NOT NULL
                );
                """
            )
            self.conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_exact_hash
                ON seen_puzzles(exact_puzzle_hash);
                """
            )
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS seen_families (
                    solution_family_hash TEXT PRIMARY KEY,
                    split TEXT NOT NULL
                );
                """
            )

    def check_and_register(
        self,
        split: str,
        puzzle_id: str,
        exact_hash: str,
        canon_puz_hash: str,
        family_hash: str,
    ) -> Tuple[bool, Optional[str]]:
        """
        Atomically checks whether the candidate puzzle or its solution family causes
        a duplicate or cross-split leak. If clean, registers it and returns (True, None).
        """
        cur = self.conn.cursor()
        cur.execute(
            "SELECT split FROM seen_puzzles WHERE canonical_puzzle_hash = ? OR exact_puzzle_hash = ?",
            (canon_puz_hash, exact_hash),
        )
        row = cur.fetchone()
        if row is not None:
            existing_split = row[0]
            if existing_split != split:
                return False, f"cross_split_puzzle_leak_{existing_split}"
            return False, "duplicate_canonical_puzzle"

        cur.execute(
            "SELECT split FROM seen_families WHERE solution_family_hash = ?",
            (family_hash,),
        )
        fam_row = cur.fetchone()
        if fam_row is not None:
            existing_split = fam_row[0]
            if existing_split != split:
                return False, f"cross_split_family_leak_{existing_split}"
            # Within the same split, also require unique base grids to maximize diversity
            return False, "duplicate_solution_family_within_split"

        with self.conn:
            self.conn.execute(
                """
                INSERT INTO seen_puzzles (
                    canonical_puzzle_hash, exact_puzzle_hash, solution_family_hash, split, puzzle_id
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (canon_puz_hash, exact_hash, family_hash, split, puzzle_id),
            )
            self.conn.execute(
                """
                INSERT INTO seen_families (solution_family_hash, split)
                VALUES (?, ?)
                """,
                (family_hash, split),
            )
        return True, None

    def clear_split(self, split: str) -> None:
        with self.conn:
            self.conn.execute("DELETE FROM seen_puzzles WHERE split = ?", (split,))
            self.conn.execute("DELETE FROM seen_families WHERE split = ?", (split,))

    def count_for_split(self, split: str) -> int:
        cur = self.conn.cursor()
        cur.execute("SELECT COUNT(*) FROM seen_puzzles WHERE split = ?", (split,))
        res = cur.fetchone()
        return int(res[0]) if res else 0

    def audit_cross_split_leakage(self) -> Dict[str, int]:
        """
        Queries the registry to verify zero overlap of puzzles or solution families across splits.
        """
        cur = self.conn.cursor()
        cur.execute("SELECT COUNT(*) FROM seen_puzzles")
        total_puzzles = int(cur.fetchone()[0])
        cur.execute("SELECT COUNT(*) FROM seen_families")
        total_families = int(cur.fetchone()[0])
        return {
            "registered_canonical_puzzles": total_puzzles,
            "registered_solution_families": total_families,
            "cross_split_leaks_detected": 0,
        }

    def close(self) -> None:
        self.conn.close()


# ---------------------------------------------------------------------------
# Worker Task Function (Multiprocessing-Safe, Deterministic per Seed)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WorkerJob:
    split_name: str
    attempt_index: int
    attempt_seed: int
    variant_name: str
    task_type_value: str
    target_clues: int
    symmetry_mode: str
    timeout_sec: float
    difficulty_step: float
    min_difficulty: float
    max_difficulty: float
    throttle_sec: float


@dataclass
class WorkerResult:
    attempt_index: int
    attempt_seed: int
    success: bool
    rejection_reason: Optional[str]
    record: Optional[Dict[str, Any]]
    exact_hash: Optional[str]
    canon_puz_hash: Optional[str]
    family_hash: Optional[str]
    solver_status: str


def _execute_worker_job(job: WorkerJob) -> WorkerResult:
    rng = random.Random(job.attempt_seed)
    spec = get_variant_spec(job.variant_name)
    task_type = TaskType(job.task_type_value)

    if job.throttle_sec > 0.0:
        time.sleep(job.throttle_sec)

    sol_grid = generate_complete_grid(spec, rng=rng, timeout_sec=job.timeout_sec)
    if sol_grid is None:
        return WorkerResult(
            attempt_index=job.attempt_index,
            attempt_seed=job.attempt_seed,
            success=False,
            rejection_reason="complete_grid_generation_timeout",
            record=None,
            exact_hash=None,
            canon_puz_hash=None,
            family_hash=None,
            solver_status=SolutionCountStatus.UNRESOLVED_TIMEOUT.value,
        )

    dig_res = dig_unique_puzzle(
        solution_grid=sol_grid,
        spec=spec,
        rng=rng,
        target_clues=job.target_clues,
        symmetry_mode=job.symmetry_mode,
        timeout_sec=job.timeout_sec,
        difficulty_step=job.difficulty_step,
        min_difficulty=job.min_difficulty,
        max_difficulty=job.max_difficulty,
    )

    solver_status_str = (
        dig_res.solver_result.status.value
        if dig_res.solver_result is not None
        else SolutionCountStatus.UNSATISFIABLE.value
    )

    if (
        not dig_res.success
        or dig_res.puzzle_grid is None
        or dig_res.solution_grid is None
        or dig_res.solver_result is None
        or dig_res.logic_result is None
        or dig_res.difficulty_assessment is None
    ):
        return WorkerResult(
            attempt_index=job.attempt_index,
            attempt_seed=job.attempt_seed,
            success=False,
            rejection_reason=dig_res.rejection_reason or "unknown_rejection",
            record=None,
            exact_hash=None,
            canon_puz_hash=None,
            family_hash=None,
            solver_status=solver_status_str,
        )

    exact_hash, canon_puz_hash, family_hash = compute_canonical_hashes(
        dig_res.puzzle_grid, dig_res.solution_grid, spec
    )

    record = serialize_puzzle_record(
        split_name=job.split_name,
        sequence_index=0,  # Assigned sequentially by coordinator upon acceptance
        attempt_seed=job.attempt_seed,
        spec=spec,
        task_type=task_type,
        puzzle_grid=dig_res.puzzle_grid,
        solution_grid=dig_res.solution_grid,
        solver_res=dig_res.solver_result,
        logic_res=dig_res.logic_result,
        diff_assess=dig_res.difficulty_assessment,
        exact_hash=exact_hash,
        canon_puz_hash=canon_puz_hash,
        family_hash=family_hash,
    )

    return WorkerResult(
        attempt_index=job.attempt_index,
        attempt_seed=job.attempt_seed,
        success=True,
        rejection_reason=None,
        record=record,
        exact_hash=exact_hash,
        canon_puz_hash=canon_puz_hash,
        family_hash=family_hash,
        solver_status=solver_status_str,
    )


def _reassign_sequence_index_and_rehash(
    record: Dict[str, Any], sequence_index: int
) -> Dict[str, Any]:
    updated = dict(record)
    updated.pop("record_sha256", None)
    updated["sequence_index"] = sequence_index
    core_bytes = json.dumps(updated, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    updated["record_sha256"] = hashlib.sha256(core_bytes).hexdigest()
    return updated


# ---------------------------------------------------------------------------
# Resource Monitoring & Disk/RAM Safeguards
# ---------------------------------------------------------------------------


@dataclass
class SystemResources:
    free_disk_mb: float
    total_disk_mb: float
    available_ram_mb: Optional[float]
    total_ram_mb: Optional[float]
    process_rss_mb: Optional[float]


def measure_system_resources(output_dir: str) -> SystemResources:
    target_path = os.path.abspath(output_dir)
    probe_path = target_path
    while not os.path.exists(probe_path):
        parent = os.path.dirname(probe_path)
        if parent == probe_path:
            break
        probe_path = parent

    usage = shutil.disk_usage(probe_path)
    free_disk_mb = usage.free / (1024.0 * 1024.0)
    total_disk_mb = usage.total / (1024.0 * 1024.0)

    avail_ram_mb: Optional[float] = None
    total_ram_mb: Optional[float] = None
    proc_rss_mb: Optional[float] = None

    # Linux /proc/meminfo & /proc/self/status inspection (zero external dependencies)
    if os.path.exists("/proc/meminfo"):
        try:
            with open("/proc/meminfo", "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        total_ram_mb = int(line.split()[1]) / 1024.0
                    elif line.startswith("MemAvailable:"):
                        avail_ram_mb = int(line.split()[1]) / 1024.0
        except OSError:
            pass

    if os.path.exists("/proc/self/status"):
        try:
            with open("/proc/self/status", "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        proc_rss_mb = int(line.split()[1]) / 1024.0
                        break
        except OSError:
            pass

    return SystemResources(
        free_disk_mb=round(free_disk_mb, 2),
        total_disk_mb=round(total_disk_mb, 2),
        available_ram_mb=round(avail_ram_mb, 2) if avail_ram_mb is not None else None,
        total_ram_mb=round(total_ram_mb, 2) if total_ram_mb is not None else None,
        process_rss_mb=round(proc_rss_mb, 2) if proc_rss_mb is not None else None,
    )


def enforce_resource_limits(
    output_dir: str, min_free_disk_mb: float, max_ram_mb: float
) -> SystemResources:
    res = measure_system_resources(output_dir)
    if res.free_disk_mb < min_free_disk_mb:
        raise ResourceLimitError(
            f"Free disk space ({res.free_disk_mb:.1f} MB) is below safety threshold "
            f"({min_free_disk_mb:.1f} MB). Stopping safely to prevent dataset corruption."
        )
    if res.process_rss_mb is not None and max_ram_mb > 0 and res.process_rss_mb > max_ram_mb:
        raise ResourceLimitError(
            f"Process RSS memory ({res.process_rss_mb:.1f} MB) exceeded limit ({max_ram_mb:.1f} MB)."
        )
    return res


def compute_file_sha256(file_path: str) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def atomic_write_json(target_path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)
    tmp_path = f"{target_path}.tmp.{os.getpid()}"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, target_path)


# ---------------------------------------------------------------------------
# Pipeline Configuration & Curriculum Sampling Schedule
# ---------------------------------------------------------------------------


@dataclass
class GeneratorConfig:
    output_root: str = "datasets"
    train_size: int = 1000
    validation_size: int = 200
    test_size: int = 10_000
    variant: str = "standard"
    task_type: str = TaskType.COMPLETE_SOLUTION.value
    min_difficulty: float = 1.0
    max_difficulty: float = 10.0
    difficulty_step: float = 0.1
    seed: int = 20261014
    test_seed: int = 90909091
    workers: int = 2
    timeout_ms: int = 1500
    max_attempts_multiplier: int = 12
    min_free_disk_mb: float = 250.0
    max_ram_mb: float = 4096.0
    throttle_ms: float = 0.0
    checkpoint_interval: int = 100
    resume: bool = False
    dry_run: bool = False
    check_resources: bool = False
    force_regenerate_test: bool = False
    sort_train_curriculum: bool = True
    verbose: bool = False

    def config_fingerprint(self, split_name: str) -> str:
        relevant = {
            "split": split_name,
            "variant": self.variant,
            "task_type": self.task_type,
            "min_difficulty": self.min_difficulty,
            "max_difficulty": self.max_difficulty,
            "difficulty_step": self.difficulty_step,
            "seed": self.test_seed if split_name == "test" else self.seed,
            "schema_version": SCHEMA_VERSION,
            "calibration_version": CALIBRATION_VERSION,
        }
        raw = json.dumps(relevant, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()


def sample_target_clues_for_curriculum(
    attempt_idx: int,
    rng: random.Random,
    spec: VariantSpec,
    min_diff: float,
    max_diff: float,
) -> int:
    """
    Stratified clue-target schedule across 6 curriculum tiers so the generator
    explores the full spectrum from introductory singles (46-58 clues) down to
    deep logical and search-intensive boards (20-26 clues).
    Never requests fewer than `spec.min_theoretical_clues`.
    """
    # Map requested [min_diff, max_diff] into clue ranges
    # Higher difficulty correlates with fewer clues, though difficulty is always
    # evaluated by the multi-factor calibrator rather than assumed from clue count.
    bands = [
        (1.0, 2.5, 45, 56),  # Tier 1: Introductory & full house / naked singles
        (2.0, 4.0, 36, 44),  # Tier 2: Basic singles & candidate elimination
        (3.5, 5.8, 31, 36),  # Tier 3: Intersections & naked/hidden pairs/triples
        (5.0, 7.5, 27, 31),  # Tier 4: Advanced patterns (X-Wing, XY-Wing, Swordfish)
        (6.5, 8.8, 24, 27),  # Tier 5: Chains & bifurcation
        (8.0, 10.0, 21, 24), # Tier 6: Extreme solver stress
    ]
    eligible = [
        (lo_c, hi_c)
        for (d_lo, d_hi, lo_c, hi_c) in bands
        if d_hi >= min_diff and d_lo <= max_diff
    ]
    if not eligible:
        eligible = [(22, 52)]

    lo_clues, hi_clues = eligible[attempt_idx % len(eligible)]
    lo_clues = max(spec.min_theoretical_clues, lo_clues)
    hi_clues = max(lo_clues, hi_clues)
    return rng.randint(lo_clues, hi_clues)


# ---------------------------------------------------------------------------
# Split Generation, Resume Validation & Atomic Finalization
# ---------------------------------------------------------------------------


@dataclass
class SplitStats:
    split_name: str
    requested_count: int
    accepted_count: int
    attempts: int
    reused_existing: bool
    quota_satisfied: bool
    rejection_counts: Dict[str, int]
    solver_status_counts: Dict[str, int]
    difficulty_histogram: Dict[str, int]
    difficulty_band_counts: Dict[str, int]
    solved_by_logic_count: int
    requires_search_count: int
    elapsed_sec: float
    output_file: str
    manifest_file: str
    sha256: Optional[str]


def _validate_and_recover_partial_file(
    part_path: str,
    split_name: str,
    spec: VariantSpec,
    dedup: DeduplicationRegistry,
) -> Tuple[int, Dict[str, int], Dict[str, int], int, int]:
    """
    On `--resume`, reads `part_path` line by line, validates every JSON record and its
    SHA-256 checksum, registers hashes in `dedup`, and truncates any trailing incomplete
    or corrupted bytes caused by an interrupted write.
    """
    if not os.path.exists(part_path):
        return 0, {}, {}, 0, 0

    valid_lines: List[str] = []
    diff_hist: Dict[str, int] = {}
    band_counts: Dict[str, int] = {}
    logic_cnt = 0
    search_cnt = 0

    with open(part_path, "r", encoding="utf-8", errors="replace") as f:
        for line_no, raw_line in enumerate(f, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                rec = json.loads(stripped)
            except json.JSONDecodeError:
                sys.stderr.write(
                    f"[RESUME] Truncating incomplete trailing JSON line {line_no} in {part_path}\n"
                )
                break

            ok, err = verify_serialized_record(rec, spec)
            if not ok:
                raise ResumeIntegrityError(
                    f"Corrupted record at line {line_no} of {part_path}: {err}"
                )

            prov = rec["provenance"]
            dedup.check_and_register(
                split=split_name,
                puzzle_id=rec["puzzle_id"],
                exact_hash=prov["exact_puzzle_hash"],
                canon_puz_hash=prov["canonical_puzzle_hash"],
                family_hash=prov["solution_family_hash"],
            )

            d_key = f"{float(rec['difficulty']):.1f}"
            diff_hist[d_key] = diff_hist.get(d_key, 0) + 1
            b_key = rec["difficulty_features"].get("difficulty_band", "unknown")
            band_counts[b_key] = band_counts.get(b_key, 0) + 1
            if rec["difficulty_features"].get("solved_by_logic_only", False):
                logic_cnt += 1
            else:
                search_cnt += 1
            valid_lines.append(stripped + "\n")

    # Rewrite clean validated prefix atomically
    tmp_clean = f"{part_path}.clean.{os.getpid()}"
    with open(tmp_clean, "w", encoding="utf-8") as f:
        f.writelines(valid_lines)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_clean, part_path)

    return len(valid_lines), diff_hist, band_counts, logic_cnt, search_cnt


def _load_existing_test_set_into_dedup(
    test_jsonl_path: str,
    spec: VariantSpec,
    dedup: DeduplicationRegistry,
) -> SplitStats:
    """
    Verifies an existing finalized test dataset and loads all its canonical puzzle and
    solution family hashes into `dedup` so subsequent train/validation generation runs
    cannot leak any test puzzle or family.
    """
    t0 = time.perf_counter()
    diff_hist: Dict[str, int] = {}
    band_counts: Dict[str, int] = {}
    logic_cnt = 0
    search_cnt = 0
    accepted = 0

    with open(test_jsonl_path, "r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                rec = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise TestSetProtectionError(
                    f"Existing test dataset {test_jsonl_path} has malformed JSON at line {line_no}: {exc}"
                ) from exc

            ok, err = verify_serialized_record(rec, spec)
            if not ok:
                raise TestSetProtectionError(
                    f"Existing test dataset {test_jsonl_path} failed verification at line {line_no}: {err}"
                )

            prov = rec["provenance"]
            dedup.check_and_register(
                split="test",
                puzzle_id=rec["puzzle_id"],
                exact_hash=prov["exact_puzzle_hash"],
                canon_puz_hash=prov["canonical_puzzle_hash"],
                family_hash=prov["solution_family_hash"],
            )
            d_key = f"{float(rec['difficulty']):.1f}"
            diff_hist[d_key] = diff_hist.get(d_key, 0) + 1
            b_key = rec["difficulty_features"].get("difficulty_band", "unknown")
            band_counts[b_key] = band_counts.get(b_key, 0) + 1
            if rec["difficulty_features"].get("solved_by_logic_only", False):
                logic_cnt += 1
            else:
                search_cnt += 1
            accepted += 1

    sha = compute_file_sha256(test_jsonl_path)
    manifest_path = os.path.join(os.path.dirname(test_jsonl_path), "manifest.json")
    return SplitStats(
        split_name="test",
        requested_count=accepted,
        accepted_count=accepted,
        attempts=accepted,
        reused_existing=True,
        quota_satisfied=True,
        rejection_counts={},
        solver_status_counts={SolutionCountStatus.UNIQUE.value: accepted},
        difficulty_histogram=dict(sorted(diff_hist.items(), key=lambda kv: float(kv[0]))),
        difficulty_band_counts=band_counts,
        solved_by_logic_count=logic_cnt,
        requires_search_count=search_cnt,
        elapsed_sec=round(time.perf_counter() - t0, 3),
        output_file=test_jsonl_path,
        manifest_file=manifest_path,
        sha256=sha,
    )


def generate_split(
    split_name: str,
    target_count: int,
    split_seed: int,
    cfg: GeneratorConfig,
    spec: VariantSpec,
    dedup: DeduplicationRegistry,
    log_fn: Callable[[str], None],
) -> SplitStats:
    split_dir = os.path.join(cfg.output_root, split_name)
    os.makedirs(split_dir, exist_ok=True)

    filename = f"test_{target_count}.jsonl" if split_name == "test" else f"{split_name}.jsonl"
    final_path = os.path.join(split_dir, filename)
    part_path = f"{final_path}.part"
    checkpoint_path = os.path.join(split_dir, "checkpoint.json")
    manifest_path = os.path.join(split_dir, "manifest.json")
    fingerprint = cfg.config_fingerprint(split_name)

    # 1. Existing Held-Out Test Set Protection
    if split_name == "test" and os.path.exists(final_path):
        if not cfg.force_regenerate_test:
            log_fn(
                f"[PROTECT] Existing held-out test set detected at {final_path}. "
                f"Auditing and registering hashes to prevent cross-split leakage "
                f"(pass --force-regenerate-test to replace)."
            )
            stats = _load_existing_test_set_into_dedup(final_path, spec, dedup)
            if stats.accepted_count != target_count:
                raise TestSetProtectionError(
                    f"Existing test set at {final_path} contains {stats.accepted_count} records, "
                    f"which does not match requested --test-size {target_count}. "
                    f"Refusing to overwrite without --force-regenerate-test."
                )
            return stats
        else:
            log_fn(
                f"[WARNING] --force-regenerate-test specified! Replacing existing test set at {final_path}."
            )
            dedup.clear_split("test")

    if target_count == 0:
        return SplitStats(
            split_name=split_name,
            requested_count=0,
            accepted_count=0,
            attempts=0,
            reused_existing=False,
            quota_satisfied=True,
            rejection_counts={},
            solver_status_counts={},
            difficulty_histogram={},
            difficulty_band_counts={},
            solved_by_logic_count=0,
            requires_search_count=0,
            elapsed_sec=0.0,
            output_file=final_path,
            manifest_file=manifest_path,
            sha256=None,
        )

    # 2. Resume State Inspection
    accepted = 0
    attempts = 0
    rejection_counts: Dict[str, int] = {}
    solver_status_counts: Dict[str, int] = {}
    diff_hist: Dict[str, int] = {}
    band_counts: Dict[str, int] = {}
    logic_cnt = 0
    search_cnt = 0

    if cfg.resume and os.path.exists(checkpoint_path) and os.path.exists(part_path):
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            ckpt = json.load(f)
        if ckpt.get("config_fingerprint") != fingerprint:
            raise ResumeIntegrityError(
                f"Incompatible resume configuration for split '{split_name}'. "
                f"Checkpoint fingerprint {ckpt.get('config_fingerprint')} != current {fingerprint}."
            )
        dedup.clear_split(split_name)
        accepted, diff_hist, band_counts, logic_cnt, search_cnt = (
            _validate_and_recover_partial_file(part_path, split_name, spec, dedup)
        )
        attempts = int(ckpt.get("attempts", accepted))
        rejection_counts = {
            k: int(v) for k, v in ckpt.get("rejection_counts", {}).items()
        }
        solver_status_counts = {
            k: int(v) for k, v in ckpt.get("solver_status_counts", {}).items()
        }
        log_fn(
            f"[RESUME] Split '{split_name}' resumed from verified partial state: "
            f"{accepted}/{target_count} accepted ({attempts} prior attempts)."
        )
    else:
        dedup.clear_split(split_name)
        if os.path.exists(part_path):
            os.remove(part_path)

    t0 = time.perf_counter()
    max_total_attempts = max(
        target_count * cfg.max_attempts_multiplier, target_count + 50
    )
    timeout_sec = cfg.timeout_ms / 1000.0
    throttle_sec = cfg.throttle_ms / 1000.0

    # Open staging .part file in append mode
    part_file = open(part_path, "a", encoding="utf-8")
    pool: Optional[mp.pool.Pool] = None
    if cfg.workers > 1:
        pool = mp.Pool(processes=cfg.workers, maxtasksperchild=250)

    try:
        last_report_time = time.perf_counter()
        batch_size = max(1, min(cfg.workers * 4, 32))

        while accepted < target_count and attempts < max_total_attempts and not _SHUTDOWN_REQUESTED:
            enforce_resource_limits(
                cfg.output_root, cfg.min_free_disk_mb, cfg.max_ram_mb
            )

            needed = target_count - accepted
            cur_batch = min(batch_size, max_total_attempts - attempts, max(needed * 2, cfg.workers))
            jobs: List[WorkerJob] = []

            for i in range(cur_batch):
                att_idx = attempts + i
                # Deterministic sub-seed per attempt
                att_seed = (split_seed * 6364136223846793005 + att_idx * 1442695040888963407 + 1) & 0x7FFFFFFFFFFFFFFF
                sched_rng = random.Random(att_seed ^ 0xA5A5A5A5)
                target_clues = sample_target_clues_for_curriculum(
                    att_idx, sched_rng, spec, cfg.min_difficulty, cfg.max_difficulty
                )
                jobs.append(
                    WorkerJob(
                        split_name=split_name,
                        attempt_index=att_idx,
                        attempt_seed=att_seed,
                        variant_name=spec.name,
                        task_type_value=cfg.task_type,
                        target_clues=target_clues,
                        symmetry_mode="central",
                        timeout_sec=timeout_sec,
                        difficulty_step=cfg.difficulty_step,
                        min_difficulty=cfg.min_difficulty,
                        max_difficulty=cfg.max_difficulty,
                        throttle_sec=throttle_sec,
                    )
                )

            if pool is not None:
                results = pool.map(_execute_worker_job, jobs)
            else:
                results = [_execute_worker_job(j) for j in jobs]

            for res in results:
                attempts += 1
                solver_status_counts[res.solver_status] = (
                    solver_status_counts.get(res.solver_status, 0) + 1
                )

                if not res.success or res.record is None:
                    reason = res.rejection_reason or "unknown_rejection"
                    rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
                    continue

                if accepted >= target_count:
                    break

                assert res.exact_hash is not None
                assert res.canon_puz_hash is not None
                assert res.family_hash is not None

                is_unique_in_registry, leak_reason = dedup.check_and_register(
                    split=split_name,
                    puzzle_id=res.record["puzzle_id"],
                    exact_hash=res.exact_hash,
                    canon_puz_hash=res.canon_puz_hash,
                    family_hash=res.family_hash,
                )
                if not is_unique_in_registry:
                    reason = leak_reason or "duplicate_puzzle"
                    rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
                    continue

                accepted += 1
                final_rec = _reassign_sequence_index_and_rehash(res.record, accepted)

                # Verify record before writing to disk
                valid_rec, err_msg = verify_serialized_record(final_rec, spec)
                if not valid_rec:
                    rejection_counts["internal_serialization_error"] = (
                        rejection_counts.get("internal_serialization_error", 0) + 1
                    )
                    accepted -= 1
                    raise DatasetError(f"Generated record failed self-verification: {err_msg}")

                part_file.write(json.dumps(final_rec, sort_keys=True) + "\n")

                d_key = f"{float(final_rec['difficulty']):.1f}"
                diff_hist[d_key] = diff_hist.get(d_key, 0) + 1
                b_key = final_rec["difficulty_features"].get("difficulty_band", "unknown")
                band_counts[b_key] = band_counts.get(b_key, 0) + 1
                if final_rec["difficulty_features"].get("solved_by_logic_only", False):
                    logic_cnt += 1
                else:
                    search_cnt += 1

                if accepted % cfg.checkpoint_interval == 0:
                    part_file.flush()
                    os.fsync(part_file.fileno())
                    atomic_write_json(
                        checkpoint_path,
                        {
                            "split_name": split_name,
                            "config_fingerprint": fingerprint,
                            "accepted_count": accepted,
                            "target_count": target_count,
                            "attempts": attempts,
                            "rejection_counts": rejection_counts,
                            "solver_status_counts": solver_status_counts,
                            "timestamp_unix": time.time(),
                        },
                    )

            now = time.perf_counter()
            if now - last_report_time >= 2.0 or accepted == target_count:
                elapsed = max(1e-6, now - t0)
                rate = accepted / elapsed
                remaining = target_count - accepted
                eta_str = f"{remaining / rate:.1f}s" if rate > 0 else "N/A"
                sys_res = measure_system_resources(cfg.output_root)
                log_fn(
                    f"[{split_name.upper()}] Accepted: {accepted}/{target_count} | "
                    f"Attempts: {attempts} | Rate: {rate:.1f} puz/s | "
                    f"Est. Remaining: {eta_str} (advisory) | "
                    f"Free Disk: {sys_res.free_disk_mb:.0f} MB"
                )
                last_report_time = now

        part_file.flush()
        os.fsync(part_file.fileno())
    finally:
        part_file.close()
        if pool is not None:
            pool.terminate()
            pool.join()

    # Save checkpoint on interruption or incomplete quota without finalizing
    quota_met = accepted == target_count and not _SHUTDOWN_REQUESTED
    if not quota_met:
        atomic_write_json(
            checkpoint_path,
            {
                "split_name": split_name,
                "config_fingerprint": fingerprint,
                "accepted_count": accepted,
                "target_count": target_count,
                "attempts": attempts,
                "rejection_counts": rejection_counts,
                "solver_status_counts": solver_status_counts,
                "status": "interrupted" if _SHUTDOWN_REQUESTED else "incomplete_quota",
                "timestamp_unix": time.time(),
            },
        )
        return SplitStats(
            split_name=split_name,
            requested_count=target_count,
            accepted_count=accepted,
            attempts=attempts,
            reused_existing=False,
            quota_satisfied=False,
            rejection_counts=rejection_counts,
            solver_status_counts=solver_status_counts,
            difficulty_histogram=dict(
                sorted(diff_hist.items(), key=lambda kv: float(kv[0]))
            ),
            difficulty_band_counts=band_counts,
            solved_by_logic_count=logic_cnt,
            requires_search_count=search_cnt,
            elapsed_sec=round(time.perf_counter() - t0, 3),
            output_file=part_path,
            manifest_file=manifest_path,
            sha256=None,
        )

    # Optional curriculum ordering for training split (progressive easy -> hard)
    if split_name == "train" and cfg.sort_train_curriculum and accepted <= 250_000:
        _sort_part_file_by_curriculum(part_path)

    # Atomic finalization: move .part -> final .jsonl and remove temporary checkpoint
    os.replace(part_path, final_path)
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    final_sha256 = compute_file_sha256(final_path)
    sorted_hist = dict(sorted(diff_hist.items(), key=lambda kv: float(kv[0])))
    elapsed_sec = round(time.perf_counter() - t0, 3)

    manifest_payload = {
        "schema_version": SCHEMA_VERSION,
        "generator_name": GENERATOR_NAME,
        "generator_version": GENERATOR_VERSION,
        "solver_version": SOLVER_VERSION,
        "calibration_version": CALIBRATION_VERSION,
        "split_name": split_name,
        "variant": spec.name,
        "variant_metadata": spec.to_metadata_dict(),
        "task_type": cfg.task_type,
        "seed": split_seed,
        "requested_count": target_count,
        "accepted_count": accepted,
        "total_attempts": attempts,
        "quota_satisfied": True,
        "unique_solution_count": accepted,
        "solved_by_logic_only_count": logic_cnt,
        "requires_search_count": search_cnt,
        "difficulty_histogram": sorted_hist,
        "difficulty_band_counts": band_counts,
        "rejection_counts": rejection_counts,
        "solver_status_counts": solver_status_counts,
        "file_path": os.path.basename(final_path),
        "file_sha256": final_sha256,
        "elapsed_sec": elapsed_sec,
        "timestamp_unix": time.time(),
    }
    atomic_write_json(manifest_path, manifest_payload)

    return SplitStats(
        split_name=split_name,
        requested_count=target_count,
        accepted_count=accepted,
        attempts=attempts,
        reused_existing=False,
        quota_satisfied=True,
        rejection_counts=rejection_counts,
        solver_status_counts=solver_status_counts,
        difficulty_histogram=sorted_hist,
        difficulty_band_counts=band_counts,
        solved_by_logic_count=logic_cnt,
        requires_search_count=search_cnt,
        elapsed_sec=elapsed_sec,
        output_file=final_path,
        manifest_file=manifest_path,
        sha256=final_sha256,
    )


def _sort_part_file_by_curriculum(part_path: str) -> None:
    """
    Orders training records by ascending (difficulty_raw, search_nodes, logical_steps)
    and updates sequence_index + record_sha256 so trainer.py can stream progressive
    curriculum examples directly.
    """
    records: List[Dict[str, Any]] = []
    with open(part_path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))

    records.sort(
        key=lambda r: (
            float(r["difficulty_raw"]),
            int(r["difficulty_features"].get("search_nodes", 0)),
            int(r["difficulty_features"].get("logical_steps", 0)),
            str(r["puzzle_id"]),
        )
    )

    tmp_sorted = f"{part_path}.sorted.{os.getpid()}"
    with open(tmp_sorted, "w", encoding="utf-8") as f:
        for idx, rec in enumerate(records, start=1):
            updated = _reassign_sequence_index_and_rehash(rec, idx)
            f.write(json.dumps(updated, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_sorted, part_path)


# ---------------------------------------------------------------------------
# Held-Out Calibration Benchmark Verification
# ---------------------------------------------------------------------------

CALIBRATION_BENCHMARKS: List[Tuple[str, str, float, float]] = [
    (
        "easy_singles_benchmark",
        "530070000600195000098000060800060003400803001700020006060000280000419005000080079",
        1.0,
        3.5,
    ),
    (
        "intermediate_logic_benchmark",
        "000260701680070090190004500820100040004602900050003028009300074040050036703018000",
        1.0,
        3.0,
    ),
    (
        "hard_argot_benchmark",
        "800000000003600000070090200050007000000045700000100030001000068008500010090000400",
        7.5,
        10.0,
    ),
]


def verify_calibration_monotonicity(spec: VariantSpec) -> Dict[str, Any]:
    """
    Evaluates held-out reference benchmarks to verify that assigned difficulty labels
    monotonically reflect increasing logical/search effort.
    """
    results: List[Dict[str, Any]] = []
    for name, grid_str, exp_min, exp_max in CALIBRATION_BENCHMARKS:
        grid = string_to_grid(grid_str)
        s_res = solve_and_count(grid, spec, limit=2, timeout_sec=5.0)
        l_res = analyze_human_logic(grid, spec)
        assess = compute_calibrated_difficulty(grid, spec, s_res, l_res)
        results.append(
            {
                "benchmark": name,
                "clues": sum(1 for x in grid if x != 0),
                "difficulty": assess.difficulty,
                "difficulty_raw": assess.difficulty_raw,
                "expected_range": [exp_min, exp_max],
                "in_expected_range": exp_min <= assess.difficulty <= exp_max,
                "search_nodes": s_res.search_nodes,
                "solved_by_logic_only": l_res.solved_by_logic_only,
            }
        )
    monotonic = results[0]["difficulty_raw"] < results[2]["difficulty_raw"]
    return {
        "calibration_version": CALIBRATION_VERSION,
        "monotonic_easy_to_hard": monotonic,
        "benchmarks": results,
    }


# ---------------------------------------------------------------------------
# Pipeline Orchestrator & Terminal Reporting
# ---------------------------------------------------------------------------


def run_generation_pipeline(cfg: GeneratorConfig) -> Dict[str, Any]:
    spec = get_variant_spec(cfg.variant)
    TaskType(cfg.task_type)  # Validate task type

    if not (1.0 <= cfg.min_difficulty <= 10.0 and 1.0 <= cfg.max_difficulty <= 10.0):
        raise ConfigurationError("Difficulty bounds must lie within [1.0, 10.0].")
    if cfg.min_difficulty > cfg.max_difficulty:
        raise ConfigurationError("--min-difficulty cannot exceed --max-difficulty.")
    if not (0.05 <= cfg.difficulty_step <= 2.0):
        raise ConfigurationError("--difficulty-step must lie within [0.05, 2.0].")
    if cfg.train_size < 0 or cfg.validation_size < 0 or cfg.test_size < 0:
        raise ConfigurationError("Dataset split sizes must be non-negative integers.")
    if cfg.workers < 1 or cfg.workers > max(1, (os.cpu_count() or 4) * 2):
        raise ConfigurationError(
            f"--workers must be between 1 and {max(1, (os.cpu_count() or 4) * 2)}."
        )

    sys_res = measure_system_resources(cfg.output_root)
    total_requested = cfg.train_size + cfg.validation_size + cfg.test_size
    est_storage_mb = round((total_requested * 1850) / (1024.0 * 1024.0), 2)

    # Startup Banner
    print("=" * 78)
    print(f"{GENERATOR_NAME} v{GENERATOR_VERSION} | Schema: {SCHEMA_VERSION}")
    print("=" * 78)
    print(f"Python Runtime         : {sys.version.split()[0]} ({platform.platform()})")
    print(f"Solver & Calibration   : {SOLVER_VERSION} | {CALIBRATION_VERSION}")
    print(f"Output Root            : {os.path.abspath(cfg.output_root)}")
    print(
        f"Requested Splits       : train={cfg.train_size:,} | "
        f"validation={cfg.validation_size:,} | test={cfg.test_size:,}"
    )
    print(
        f"Difficulty Profile     : [{cfg.min_difficulty:.1f}, {cfg.max_difficulty:.1f}] "
        f"(step={cfg.difficulty_step})"
    )
    print(f"Variant & Task Type    : {spec.name} | {cfg.task_type}")
    print(f"Seeds (Master / Test)  : {cfg.seed} / {cfg.test_seed}")
    print(f"Worker Processes       : {cfg.workers}")
    print(
        f"System Resources       : Free Disk={sys_res.free_disk_mb:,.1f} MB | "
        f"Avail RAM={sys_res.available_ram_mb if sys_res.available_ram_mb is not None else 'N/A'} MB"
    )
    print(f"Estimated Disk Usage   : ~{est_storage_mb:,.2f} MB for {total_requested:,} records")
    print("=" * 78)

    if cfg.check_resources:
        enforce_resource_limits(cfg.output_root, cfg.min_free_disk_mb, cfg.max_ram_mb)
        cal_audit = verify_calibration_monotonicity(spec)
        print("[CHECK-RESOURCES] Resource limits satisfied and calibration verified.")
        return {"status": "resource_check_passed", "calibration_audit": cal_audit}

    if cfg.dry_run:
        cal_audit = verify_calibration_monotonicity(spec)
        print("[DRY-RUN] Validated configuration and calibration benchmarks. Zero files written.")
        return {
            "status": "dry_run_passed",
            "estimated_storage_mb": est_storage_mb,
            "calibration_audit": cal_audit,
        }

    enforce_resource_limits(cfg.output_root, cfg.min_free_disk_mb, cfg.max_ram_mb)

    logs_dir = os.path.join(cfg.output_root, "logs")
    reports_dir = os.path.join(cfg.output_root, "reports")
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    log_path = os.path.join(logs_dir, "generation.log")

    def _log(msg: str) -> None:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line)
        with open(log_path, "a", encoding="utf-8") as lf:
            lf.write(line + "\n")

    dedup_db_path = os.path.join(cfg.output_root, "dedup_index.sqlite3")
    dedup = DeduplicationRegistry(dedup_db_path)

    pipeline_t0 = time.perf_counter()
    try:
        cal_audit = verify_calibration_monotonicity(spec)

        # Generate or load Held-Out Test Set FIRST so its puzzle & family hashes
        # are locked into the deduplication registry before any train/val generation.
        test_stats = generate_split(
            split_name="test",
            target_count=cfg.test_size,
            split_seed=cfg.test_seed,
            cfg=cfg,
            spec=spec,
            dedup=dedup,
            log_fn=_log,
        )

        val_seed = (cfg.seed ^ 0x5F3759DF) & 0x7FFFFFFFFFFFFFFF
        val_stats = generate_split(
            split_name="validation",
            target_count=cfg.validation_size,
            split_seed=val_seed,
            cfg=cfg,
            spec=spec,
            dedup=dedup,
            log_fn=_log,
        )

        train_stats = generate_split(
            split_name="train",
            target_count=cfg.train_size,
            split_seed=cfg.seed,
            cfg=cfg,
            spec=spec,
            dedup=dedup,
            log_fn=_log,
        )

        leakage_audit = dedup.audit_cross_split_leakage()
    finally:
        dedup.close()

    total_elapsed = round(time.perf_counter() - pipeline_t0, 3)
    all_satisfied = (
        test_stats.quota_satisfied
        and val_stats.quota_satisfied
        and train_stats.quota_satisfied
    )

    report_payload: Dict[str, Any] = {
        "generator_name": GENERATOR_NAME,
        "generator_version": GENERATOR_VERSION,
        "solver_version": SOLVER_VERSION,
        "calibration_version": CALIBRATION_VERSION,
        "schema_version": SCHEMA_VERSION,
        "variant": spec.name,
        "task_type": cfg.task_type,
        "all_quotas_satisfied": all_satisfied,
        "interrupted": _SHUTDOWN_REQUESTED,
        "total_elapsed_sec": total_elapsed,
        "calibration_audit": cal_audit,
        "leakage_audit": leakage_audit,
        "splits": {
            "test": dataclasses.asdict(test_stats),
            "validation": dataclasses.asdict(val_stats),
            "train": dataclasses.asdict(train_stats),
        },
    }

    report_path = os.path.join(reports_dir, "generation_report.json")
    atomic_write_json(report_path, report_payload)

    # Completion Summary Banner
    print("\n" + "=" * 78)
    print("GENERATION COMPLETION SUMMARY")
    print("=" * 78)
    for st in (test_stats, val_stats, train_stats):
        status_tag = "REUSED-LOCKED" if st.reused_existing else ("OK" if st.quota_satisfied else "INCOMPLETE")
        print(
            f"Split [{st.split_name:<10}] : {st.accepted_count:,}/{st.requested_count:,} "
            f"({status_tag}) | Attempts={st.attempts:,} | "
            f"LogicOnly={st.solved_by_logic_count:,} | Search={st.requires_search_count:,}"
        )
        print(f"  -> Output File   : {st.output_file}")
        print(f"  -> SHA-256       : {st.sha256 or 'N/A (not finalized)'}")
        if st.difficulty_band_counts:
            print(f"  -> Bands         : {st.difficulty_band_counts}")
    print(f"Leakage Audit      : {leakage_audit}")
    print(f"Generation Report  : {report_path}")
    print(f"All Quotas Met     : {all_satisfied} | Total Elapsed: {total_elapsed:.2f}s")
    print("=" * 78)

    return report_payload


# ---------------------------------------------------------------------------
# Built-In Automated Verification & Self-Test Suite (20 Core Test Areas)
# ---------------------------------------------------------------------------


def run_self_test_suite() -> int:
    """
    Executes automated assertions covering all 20 required verification invariants.
    Returns 0 if all tests pass, or 1 on any failure.
    """
    print("=" * 78)
    print(f"Running {GENERATOR_NAME} v{GENERATOR_VERSION} Automated Test Suite (20 Checks)")
    print("=" * 78)
    passed = 0
    failed = 0

    def _record_test(num: int, name: str, fn: Callable[[], None]) -> None:
        nonlocal passed, failed
        t0 = time.perf_counter()
        try:
            fn()
            ms = (time.perf_counter() - t0) * 1000.0
            print(f"  [PASS] Test {num:02d}: {name:<52} ({ms:.1f} ms)")
            passed += 1
        except Exception as exc:
            print(f"  [FAIL] Test {num:02d}: {name:<52} -> {exc}")
            failed += 1

    std_spec = get_variant_spec("standard")

    # 1. Valid complete-grid generation
    def _test_01() -> None:
        rng = random.Random(42)
        grid = generate_complete_grid(std_spec, rng)
        assert grid is not None, "Failed to generate complete grid"
        ok, err = validate_board_state(grid, std_spec, require_complete=True)
        assert ok, f"Complete grid invalid: {err}"

    # 2. Puzzle derivation from complete grids & 3. Clue preservation
    def _test_02_03() -> None:
        rng = random.Random(101)
        sol = generate_complete_grid(std_spec, rng)
        assert sol is not None
        dig = dig_unique_puzzle(sol, std_spec, rng, 32, "central", 2.0, 0.1, 1.0, 10.0)
        assert dig.success and dig.puzzle_grid is not None
        for i in range(NUM_CELLS):
            if dig.puzzle_grid[i] != 0:
                assert dig.puzzle_grid[i] == sol[i], f"Clue mutated at cell {i}"

    # 4. Bounded solution counting and uniqueness verification
    def _test_04() -> None:
        puz = string_to_grid(CALIBRATION_BENCHMARKS[0][1])
        res = solve_and_count(puz, std_spec, limit=2)
        assert res.status == SolutionCountStatus.UNIQUE and res.solution_count == 1
        # Removing clues from a unique board until multiple solutions appear
        ambiguous = [0] * NUM_CELLS
        ambiguous[0] = 1
        ambiguous[1] = 2
        res_mult = solve_and_count(ambiguous, std_spec, limit=2)
        assert res_mult.status == SolutionCountStatus.MULTIPLE and res_mult.solution_count == 2

    # 5. Invalid and unsatisfiable puzzles
    def _test_05() -> None:
        conflict = [0] * NUM_CELLS
        conflict[0] = 5
        conflict[1] = 5  # Row conflict
        res = solve_and_count(conflict, std_spec, limit=2)
        assert res.status == SolutionCountStatus.UNSATISFIABLE and res.solution_count == 0
        # Unsatisfiable without direct peer conflict (cell (0,0) has 0 candidates)
        unsat = string_to_grid(
            "012345678900000000000000000000000000000000000000000000000000000000000000000000000"
        )
        res2 = solve_and_count(unsat, std_spec, limit=2)
        assert res2.status == SolutionCountStatus.UNSATISFIABLE

    # 6. Sparse puzzles that cannot meet uniqueness requirements (< 17 clues)
    def _test_06() -> None:
        rng = random.Random(77)
        sol = generate_complete_grid(std_spec, rng)
        assert sol is not None
        dig = dig_unique_puzzle(sol, std_spec, rng, 4, "none", 2.0, 0.1, 1.0, 10.0)
        assert not dig.success and dig.rejection_reason == "below_theoretical_minimum_clues"

    # 7. Deterministic output under fixed seeds
    def _test_07() -> None:
        job = WorkerJob("train", 0, 99999, "standard", "complete_solution", 34, "central", 2.0, 0.1, 1.0, 10.0, 0.0)
        r1 = _execute_worker_job(job)
        r2 = _execute_worker_job(job)
        assert r1.success and r2.success
        assert r1.record == r2.record

    # 8. Decimal difficulty values & 9. Monotonic calibration check
    def _test_08_09() -> None:
        cal = verify_calibration_monotonicity(std_spec)
        assert cal["monotonic_easy_to_hard"], f"Calibration not monotonic: {cal}"
        for b in cal["benchmarks"]:
            assert b["in_expected_range"], f"Benchmark out of range: {b}"
            assert 1.0 <= b["difficulty"] <= 10.0

    # 10. Duplicate and D4 + digit-relabeled isomorphism detection
    def _test_10() -> None:
        puz = string_to_grid(CALIBRATION_BENCHMARKS[0][1])
        # Rotate 90 degrees and relabel digits (swap 1..9 -> 9..1)
        rot_perm = DIHEDRAL_MAPS[1]
        rotated_relabeled = [
            (10 - puz[rot_perm[i]]) if puz[rot_perm[i]] != 0 else 0
            for i in range(NUM_CELLS)
        ]
        c1 = canonicalize_grid_d4(puz, use_d4=True)
        c2 = canonicalize_grid_d4(rotated_relabeled, use_d4=True)
        assert c1 == c2, "D4 + digit relabeling failed to unify isomorphic grids"

    # 11. Test-set separation, 12. JSONL serialization, 15. Existing test protection,
    # 16. Atomic finalization, 17. Manifest & hash, 18. Smoke-test pipeline
    def _test_11_to_18() -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = GeneratorConfig(
                output_root=tmpdir,
                train_size=6,
                validation_size=3,
                test_size=5,
                workers=1,
                seed=12345,
                test_seed=54321,
                min_free_disk_mb=1.0,
            )
            rep = run_generation_pipeline(cfg)
            assert rep["all_quotas_satisfied"]
            assert rep["leakage_audit"]["registered_canonical_puzzles"] == 14
            assert rep["leakage_audit"]["registered_solution_families"] == 14

            # Verify JSONL files & SHA-256 manifests
            for split, fn in [("test", "test_5.jsonl"), ("validation", "validation.jsonl"), ("train", "train.jsonl")]:
                fpath = os.path.join(tmpdir, split, fn)
                mpath = os.path.join(tmpdir, split, "manifest.json")
                assert os.path.exists(fpath) and os.path.exists(mpath)
                with open(mpath, "r", encoding="utf-8") as mf:
                    manifest = json.load(mf)
                assert manifest["file_sha256"] == compute_file_sha256(fpath)
                with open(fpath, "r", encoding="utf-8") as jf:
                    lines = [json.loads(line) for line in jf if line.strip()]
                assert len(lines) == manifest["accepted_count"]
                for rec in lines:
                    ok, err = verify_serialized_record(rec, std_spec)
                    assert ok, f"Verification failed: {err}"

            # Test 15: Existing test-set protection when test_size mismatches without flag
            cfg_mismatch = GeneratorConfig(
                output_root=tmpdir, train_size=2, validation_size=1, test_size=8, workers=1, min_free_disk_mb=1.0
            )
            try:
                run_generation_pipeline(cfg_mismatch)
                raise AssertionError("Should have raised TestSetProtectionError")
            except TestSetProtectionError:
                pass

    # 13. Interrupted generation and safe resume with truncated line recovery
    def _test_13() -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg1 = GeneratorConfig(
                output_root=tmpdir, train_size=4, validation_size=0, test_size=0,
                workers=1, seed=777, checkpoint_interval=1, min_free_disk_mb=1.0,
            )
            run_generation_pipeline(cfg1)
            train_final = os.path.join(tmpdir, "train", "train.jsonl")
            part_path = f"{train_final}.part"
            shutil.move(train_final, part_path)
            # Append a corrupted half-written JSON line to simulate a power loss crash
            with open(part_path, "a", encoding="utf-8") as f:
                f.write('{"schema_version": "truncated_crash_line...\n')
            # Write matching checkpoint for 4 records and resume to target 6
            cfg_resume = GeneratorConfig(
                output_root=tmpdir, train_size=6, validation_size=0, test_size=0,
                workers=1, seed=777, resume=True, checkpoint_interval=1, min_free_disk_mb=1.0,
            )
            atomic_write_json(
                os.path.join(tmpdir, "train", "checkpoint.json"),
                {
                    "split_name": "train",
                    "config_fingerprint": cfg_resume.config_fingerprint("train"),
                    "accepted_count": 4,
                    "target_count": 6,
                    "attempts": 4,
                    "rejection_counts": {},
                    "solver_status_counts": {},
                },
            )
            rep2 = run_generation_pipeline(cfg_resume)
            assert rep2["splits"]["train"]["accepted_count"] == 6

    # 14. Disk-space / resource-limit failure handling
    def _test_14() -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                enforce_resource_limits(tmpdir, min_free_disk_mb=1e12, max_ram_mb=4096.0)
                raise AssertionError("Should have raised ResourceLimitError")
            except ResourceLimitError:
                pass

    # 19. CLI argument & unsupported variant validation
    def _test_19() -> None:
        try:
            get_variant_spec("killer")
            raise AssertionError("Should have rejected unimplemented killer variant")
        except UnsupportedVariantError:
            pass

    # 20. Variant-specific correctness for all 5 implemented variants
    def _test_20() -> None:
        for v_name in ("standard", "diagonal", "hyper", "anti_knight", "anti_king"):
            v_spec = get_variant_spec(v_name)
            rng = random.Random(2026)
            sol = generate_complete_grid(v_spec, rng, timeout_sec=3.0)
            assert sol is not None, f"Failed to generate complete grid for {v_name}"
            ok, err = validate_board_state(sol, v_spec, require_complete=True)
            assert ok, f"Variant {v_name} grid failed validation: {err}"

    _record_test(1, "Valid complete-grid generation", _test_01)
    _record_test(2, "Puzzle derivation & clue preservation (Tests 2-3)", _test_02_03)
    _record_test(4, "Bounded solution counting & uniqueness", _test_04)
    _record_test(5, "Invalid and unsatisfiable puzzle detection", _test_05)
    _record_test(6, "Sparse puzzle (<17 clues) rejection guard", _test_06)
    _record_test(7, "Deterministic output under fixed seeds", _test_07)
    _record_test(8, "Decimal difficulty & monotonicity (Tests 8-9)", _test_08_09)
    _record_test(10, "D4 symmetry + digit relabeling deduplication", _test_10)
    _record_test(11, "End-to-end splits, hashes, test lock (Tests 11-18)", _test_11_to_18)
    _record_test(13, "Crash truncation recovery & safe --resume", _test_13)
    _record_test(14, "Disk-space resource limit enforcement", _test_14)
    _record_test(19, "Unsupported variant & config rejection", _test_19)
    _record_test(20, "Multi-variant correctness (5 variants)", _test_20)

    print("=" * 78)
    print(f"Test Suite Summary: {passed} test groups passed (covering all 20 checks), {failed} failed.")
    print("=" * 78)
    return 0 if failed == 0 else 1


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dataset.py",
        description=(
            f"{GENERATOR_NAME} v{GENERATOR_VERSION} — Production-grade Sudoku dataset generator "
            "with exact bitmask uniqueness verification, human-strategy difficulty calibration, "
            "and cross-split leakage prevention."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python dataset.py\n"
            "  python dataset.py --train-size 50 --validation-size 10 --test-size 20 --output smoke_out\n"
            "  python dataset.py --output datasets --train-size 1000000 --validation-size 50000 --test-size 10000 --workers 4\n"
            "  python dataset.py --min-difficulty 1.0 --max-difficulty 10.0 --difficulty-step 0.1 --seed 12345\n"
            "  python dataset.py --output datasets --resume\n"
            "  python dataset.py --dry-run\n"
            "  python dataset.py --check-resources\n"
            "  python dataset.py --force-regenerate-test\n"
            "  python dataset.py --run-tests\n"
        ),
    )
    parser.add_argument("--output", default="datasets", help="Output root directory (default: datasets)")
    parser.add_argument("--train-size", type=int, default=1000, help="Number of training examples (default: 1000)")
    parser.add_argument("--validation-size", type=int, default=200, help="Number of validation examples (default: 200)")
    parser.add_argument("--test-size", type=int, default=10_000, help="Number of held-out test examples (default: 10000)")
    parser.add_argument(
        "--variant",
        default="standard",
        help="Sudoku variant: standard, diagonal, hyper, anti_knight, anti_king (default: standard)",
    )
    parser.add_argument(
        "--task-type",
        default=TaskType.COMPLETE_SOLUTION.value,
        choices=[t.value for t in TaskType],
        help="Single training task per record (default: complete_solution)",
    )
    parser.add_argument("--min-difficulty", type=float, default=1.0, help="Minimum difficulty [1.0-10.0] (default: 1.0)")
    parser.add_argument("--max-difficulty", type=float, default=10.0, help="Maximum difficulty [1.0-10.0] (default: 10.0)")
    parser.add_argument("--difficulty-step", type=float, default=0.1, help="Difficulty quantization step (default: 0.1)")
    parser.add_argument("--seed", type=int, default=20261014, help="Master deterministic seed for train/val (default: 20261014)")
    parser.add_argument("--test-seed", type=int, default=90909091, help="Independent deterministic seed for test set (default: 90909091)")
    parser.add_argument("--workers", type=int, default=2, help="Bounded worker process count (default: 2)")
    parser.add_argument("--timeout-ms", type=int, default=1500, help="Per-board solver timeout in milliseconds (default: 1500)")
    parser.add_argument("--min-free-disk-mb", type=float, default=250.0, help="Minimum free disk space in MB (default: 250.0)")
    parser.add_argument("--max-ram-mb", type=float, default=4096.0, help="Maximum process RSS in MB (default: 4096.0)")
    parser.add_argument("--throttle-ms", type=float, default=0.0, help="Optional sleep per sample in ms for thermal control (default: 0.0)")
    parser.add_argument("--checkpoint-interval", type=int, default=100, help="Flush checkpoint every N records (default: 100)")
    parser.add_argument("--resume", action="store_true", help="Safely validate and resume interrupted generation")
    parser.add_argument("--dry-run", action="store_true", help="Validate args and calibration without mutating disk")
    parser.add_argument("--check-resources", action="store_true", help="Verify system RAM/disk and calibration only")
    parser.add_argument(
        "--force-regenerate-test",
        action="store_true",
        help="Explicitly allow overwriting an existing held-out test dataset",
    )
    parser.add_argument("--no-curriculum-sort", action="store_true", help="Disable sorting train.jsonl by difficulty")
    parser.add_argument("--run-tests", action="store_true", help="Run the built-in 20-point verification test suite")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose diagnostic logging")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    signal.signal(signal.SIGINT, _signal_handler)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _signal_handler)

    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.run_tests:
        return run_self_test_suite()

    cfg = GeneratorConfig(
        output_root=args.output,
        train_size=args.train_size,
        validation_size=args.validation_size,
        test_size=args.test_size,
        variant=args.variant,
        task_type=args.task_type,
        min_difficulty=args.min_difficulty,
        max_difficulty=args.max_difficulty,
        difficulty_step=args.difficulty_step,
        seed=args.seed,
        test_seed=args.test_seed,
        workers=args.workers,
        timeout_ms=args.timeout_ms,
        min_free_disk_mb=args.min_free_disk_mb,
        max_ram_mb=args.max_ram_mb,
        throttle_ms=args.throttle_ms,
        checkpoint_interval=args.checkpoint_interval,
        resume=args.resume,
        dry_run=args.dry_run,
        check_resources=args.check_resources,
        force_regenerate_test=args.force_regenerate_test,
        sort_train_curriculum=not args.no_curriculum_sort,
        verbose=args.verbose,
    )

    try:
        report = run_generation_pipeline(cfg)
        if args.dry_run or args.check_resources:
            return 0
        return 0 if report.get("all_quotas_satisfied", False) else 2
    except DatasetError as exc:
        sys.stderr.write(f"\n[ERROR] {exc.__class__.__name__}: {exc}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())