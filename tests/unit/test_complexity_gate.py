from pathlib import Path

from scripts import check_complexity as complexity


def _write(root: Path, relative_path: str, content: str) -> Path:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def test_analyzer_counts_decisions_but_excludes_nested_functions(tmp_path) -> None:
    path = _write(
        tmp_path,
        "src/example.py",
        """\
def sample(flag, values):
    if flag and values:
        for value in values:
            if value > 0:
                return value
    def nested():
        if flag:
            return True
        return False
    return nested()
""",
    )

    metrics = {item.qualified_name: item for item in complexity.analyze_file(path, root=tmp_path)}

    assert metrics["sample"].complexity == 5
    assert metrics["sample.nested"].complexity == 2
    assert metrics["sample"].lines == 10


def test_new_function_fails_complexity_and_length_limits(tmp_path) -> None:
    path = _write(
        tmp_path,
        "src/too_complex.py",
        """\
def oversized(first, second):
    if first:
        first = False
    if second:
        second = False
    return first or second
""",
    )

    failures = complexity.check_paths(
        [path],
        root=tmp_path,
        baseline={},
        maximum_complexity=2,
        maximum_function_lines=4,
    )

    assert failures == ["src/too_complex.py:1: oversized: complexity 4>2, lines 6>4"]


def test_legacy_budget_allows_current_metric_but_rejects_regression(tmp_path) -> None:
    path = _write(
        tmp_path,
        "src/legacy.py",
        """\
def legacy(value):
    if value:
        return True
    result = False
    return result
""",
    )
    key = "src/legacy.py:legacy"

    assert (
        complexity.check_paths(
            [path],
            root=tmp_path,
            baseline={key: complexity.FunctionBudget(complexity=2, lines=5)},
            maximum_complexity=10,
            maximum_function_lines=10,
        )
        == []
    )

    _write(
        tmp_path,
        "src/legacy.py",
        """\
def legacy(value, fallback):
    if value:
        return True
    if fallback:
        return False
    result = False
    return result
""",
    )
    assert complexity.check_paths(
        [path],
        root=tmp_path,
        baseline={key: complexity.FunctionBudget(complexity=2, lines=5)},
        maximum_complexity=10,
        maximum_function_lines=10,
    ) == ["src/legacy.py:1: legacy: complexity 3>2, lines 7>5"]


def test_repository_complexity_baseline_is_current() -> None:
    assert complexity.check_paths() == []


def test_cli_returns_failure_for_invalid_python(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.setattr(complexity, "ROOT", tmp_path)
    _write(tmp_path, "src/broken.py", "def broken(:\n")

    assert complexity.main(["src/broken.py"]) == 1
    assert "cannot analyze Python source" in capsys.readouterr().out
