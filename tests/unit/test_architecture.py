import ast
from pathlib import Path


def test_application_layer_does_not_import_opencv() -> None:
    application_root = Path(__file__).resolve().parents[2] / "src/vehicle_intelligence/application"
    violations: list[str] = []

    for path in application_root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            imported: tuple[str, ...] = ()
            if isinstance(node, ast.Import):
                imported = tuple(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imported = (node.module,)
            if any(module == "cv2" or module.startswith("cv2.") for module in imported):
                relative = path.relative_to(application_root.parent.parent)
                violations.append(f"{relative}:{node.lineno}")

    assert not violations, f"OpenCV imports belong in infrastructure adapters: {violations}"
