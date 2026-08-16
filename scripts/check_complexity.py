#!/usr/bin/env python3
"""Enforce bounded Python function complexity without third-party dependencies."""

from __future__ import annotations

import argparse
import ast
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAX_COMPLEXITY = 30
MAX_FUNCTION_LINES = 200


@dataclass(frozen=True, slots=True)
class FunctionBudget:
    complexity: int
    lines: int


# Existing hotspots may improve but must not exceed these measured budgets. New
# functions and refactored helpers always use the stricter repository defaults.
LEGACY_BASELINE: dict[str, FunctionBudget] = {
    "src/vehicle_intelligence/infrastructure/training/dataset_registry_files.py:"
    "FileDatasetRegistryRepository._load_sources": FunctionBudget(32, 78),
    "src/vehicle_intelligence/interfaces/api.py:create_app": FunctionBudget(49, 787),
    "src/vehicle_intelligence/interfaces/dataset_review_api.py:build_dataset_review_router": (
        FunctionBudget(1, 215)
    ),
    "src/vehicle_intelligence/interfaces/policy_api.py:build_policy_router": FunctionBudget(2, 334),
    "src/vehicle_intelligence/training/cli.py:build_parser": FunctionBudget(1, 251),
    "src/vehicle_intelligence/training/cli.py:run": FunctionBudget(45, 317),
    "src/vehicle_intelligence/training/dataset.py:verify_detector_dataset": FunctionBudget(52, 106),
    "src/vehicle_intelligence/training/first_party.py:verify_first_party_detector_source": (
        FunctionBudget(35, 83)
    ),
    "src/vehicle_intelligence/training/review_promotion.py:"
    "ReviewedFirstPartySourceBuilder._materialize": FunctionBudget(11, 205),
    "src/vehicle_intelligence/training/video_extraction.py:"
    "VideoTrainingImageExtractor._process_frame": FunctionBudget(22, 204),
    "src/vehicle_intelligence/training/video_review_promotion.py:"
    "AttestedVideoReviewPromotionBuilder._materialize": FunctionBudget(9, 289),
    "src/vehicle_intelligence/training/video_review_source.py:verify_video_plate_review_source": (
        FunctionBudget(61, 121)
    ),
}


@dataclass(frozen=True, slots=True)
class FunctionMetric:
    path: str
    qualified_name: str
    line: int
    complexity: int
    lines: int

    @property
    def key(self) -> str:
        return f"{self.path}:{self.qualified_name}"


class _ComplexityVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.value = 1

    def visit_If(self, node: ast.If) -> None:
        self.value += 1
        self.generic_visit(node)

    def visit_For(self, node: ast.For | ast.AsyncFor) -> None:
        self.value += 1
        self.generic_visit(node)

    visit_AsyncFor = visit_For

    def visit_While(self, node: ast.While) -> None:
        self.value += 1
        self.generic_visit(node)

    def visit_IfExp(self, node: ast.IfExp) -> None:
        self.value += 1
        self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        self.value += 1
        self.generic_visit(node)

    def visit_BoolOp(self, node: ast.BoolOp) -> None:
        self.value += max(0, len(node.values) - 1)
        self.generic_visit(node)

    def visit_Match(self, node: ast.Match) -> None:
        self.value += len(node.cases)
        self.generic_visit(node)

    def _visit_comprehension(
        self,
        node: ast.ListComp | ast.SetComp | ast.DictComp | ast.GeneratorExp,
    ) -> None:
        self.value += len(node.generators)
        self.value += sum(len(generator.ifs) for generator in node.generators)
        self.generic_visit(node)

    visit_ListComp = _visit_comprehension
    visit_SetComp = _visit_comprehension
    visit_DictComp = _visit_comprehension
    visit_GeneratorExp = _visit_comprehension

    # A nested callable receives its own metric and must not inflate its parent.
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        del node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        del node

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        del node

    def visit_Lambda(self, node: ast.Lambda) -> None:
        del node


class _FunctionCollector(ast.NodeVisitor):
    def __init__(self, path: str) -> None:
        self._path = path
        self._scope: list[str] = []
        self.metrics: list[FunctionMetric] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._scope.append(node.name)
        self.generic_visit(node)
        self._scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        visitor = _ComplexityVisitor()
        for statement in node.body:
            visitor.visit(statement)
        start_line = min([node.lineno, *(item.lineno for item in node.decorator_list)])
        qualified_name = ".".join((*self._scope, node.name))
        self.metrics.append(
            FunctionMetric(
                path=self._path,
                qualified_name=qualified_name,
                line=start_line,
                complexity=visitor.value,
                lines=(node.end_lineno or node.lineno) - start_line + 1,
            )
        )
        self._scope.append(node.name)
        self.generic_visit(node)
        self._scope.pop()


def analyze_file(path: Path, *, root: Path | None = None) -> tuple[FunctionMetric, ...]:
    root = ROOT if root is None else root
    relative_path = path.resolve().relative_to(root.resolve()).as_posix()
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative_path)
    collector = _FunctionCollector(relative_path)
    collector.visit(tree)
    return tuple(collector.metrics)


def check_paths(
    paths: Iterable[Path] | None = None,
    *,
    root: Path | None = None,
    baseline: dict[str, FunctionBudget] | None = None,
    maximum_complexity: int = MAX_COMPLEXITY,
    maximum_function_lines: int = MAX_FUNCTION_LINES,
) -> list[str]:
    root = ROOT if root is None else root
    budgets = LEGACY_BASELINE if baseline is None else baseline
    failures: list[str] = []
    for path in _python_files(paths, root=root):
        try:
            metrics = analyze_file(path, root=root)
        except (OSError, SyntaxError, UnicodeError) as exc:
            relative_path = path.resolve().relative_to(root.resolve()).as_posix()
            failures.append(f"{relative_path}: cannot analyze Python source: {exc}")
            continue
        for metric in metrics:
            budget = budgets.get(metric.key)
            complexity_limit = budget.complexity if budget is not None else maximum_complexity
            line_limit = budget.lines if budget is not None else maximum_function_lines
            reasons: list[str] = []
            if metric.complexity > complexity_limit:
                reasons.append(f"complexity {metric.complexity}>{complexity_limit}")
            if metric.lines > line_limit:
                reasons.append(f"lines {metric.lines}>{line_limit}")
            if reasons:
                failures.append(
                    f"{metric.path}:{metric.line}: {metric.qualified_name}: {', '.join(reasons)}"
                )
    return failures


def _python_files(paths: Iterable[Path] | None, *, root: Path) -> tuple[Path, ...]:
    candidates = tuple(paths) if paths is not None else (root / "src", root / "scripts")
    files: set[Path] = set()
    for candidate in candidates:
        resolved = candidate if candidate.is_absolute() else root / candidate
        if resolved.is_file() and resolved.suffix == ".py":
            files.add(resolved)
        elif resolved.is_dir():
            files.update(resolved.rglob("*.py"))
    return tuple(sorted(files))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path)
    arguments = parser.parse_args(argv)
    failures = check_paths(arguments.paths or None)
    if failures:
        print("\n".join(failures))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
