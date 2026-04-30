"""
Dependency checker for the OCR evaluation environment.

Verifies that all required Python packages and system binaries are installed
for each OCR model and the benchmark infrastructure.

Usage:
    python -m ocr_evaluation.check_dependencies
    python -m ocr_evaluation.check_dependencies --model tesseract
    python -m ocr_evaluation.check_dependencies --verbose
"""

import argparse
import importlib
import shutil
import subprocess
import sys
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Dependency declarations
# ---------------------------------------------------------------------------

@dataclass
class PythonDep:
    import_name: str
    pip_name: str
    description: str = ""


@dataclass
class SystemDep:
    binary: str
    apt_package: str
    description: str = ""


@dataclass
class ModelDeps:
    name: str
    python: list[PythonDep] = field(default_factory=list)
    system: list[SystemDep] = field(default_factory=list)


# Shared deps used by multiple models
_PDF2IMAGE = PythonDep("pdf2image", "pdf2image", "PDF → image converter")
_NUMPY = PythonDep("numpy", "numpy", "Array processing")
_PILLOW = PythonDep("PIL", "Pillow", "Image handling")
_POPPLER = SystemDep("pdftoppm", "poppler-utils", "Poppler PDF utilities (required by pdf2image)")

MODEL_DEPS: dict[str, ModelDeps] = {
    "tesseract": ModelDeps(
        name="Tesseract",
        python=[
            PythonDep("pytesseract", "pytesseract", "Python wrapper for Tesseract"),
            _PDF2IMAGE,
            PythonDep("cv2", "opencv-python-headless", "Image preprocessing (OpenCV)"),
            _NUMPY,
            _PILLOW,
        ],
        system=[
            _POPPLER,
            SystemDep("tesseract", "tesseract-ocr", "Tesseract OCR engine"),
            SystemDep(
                "tesseract",  # checked via language list, not a separate binary
                "tesseract-ocr-por",
                "Tesseract Portuguese language data",
            ),
        ],
    ),
    "easyocr": ModelDeps(
        name="EasyOCR",
        python=[
            PythonDep("easyocr", "easyocr", "EasyOCR library"),
            _PDF2IMAGE,
            _NUMPY,
        ],
        system=[_POPPLER],
    ),
    "doctr": ModelDeps(
        name="docTR",
        python=[
            PythonDep("doctr", "python-doctr[torch]", "docTR OCR library"),
            _PDF2IMAGE,
        ],
        system=[_POPPLER],
    ),
    "unstructured": ModelDeps(
        name="Unstructured",
        python=[
            PythonDep("unstructured", "unstructured[pdf]", "Unstructured document parser"),
        ],
        system=[_POPPLER],
    ),
    "chandra": ModelDeps(
        name="Chandra OCR 2",
        python=[
            PythonDep("chandra", "chandra-ocr", "Chandra OCR 2 (datalab-to/chandra-ocr-2)"),
        ],
        system=[],
    ),
    "paddleocr": ModelDeps(
        name="PaddleOCR",
        python=[
            PythonDep("paddle", "paddlepaddle", "PaddlePaddle deep learning framework"),
            PythonDep("paddleocr", "paddleocr", "PaddleOCR library"),
            _PDF2IMAGE,
            _NUMPY,
        ],
        system=[_POPPLER],
    ),
}

BENCHMARK_DEPS = ModelDeps(
    name="Benchmark infrastructure",
    python=[
        PythonDep("loguru", "loguru", "Structured logging"),
        PythonDep("jiwer", "jiwer", "CER/WER computation"),
        PythonDep("tqdm", "tqdm", "Progress bars (optional)"),
    ],
    system=[],
)


# ---------------------------------------------------------------------------
# Check helpers
# ---------------------------------------------------------------------------

def _check_python(dep: PythonDep, verbose: bool) -> bool:
    """Return True if the Python package can be imported."""
    try:
        importlib.import_module(dep.import_name)
        if verbose:
            mod = sys.modules[dep.import_name]
            version = getattr(mod, "__version__", "?")
            print(f"    [OK] {dep.import_name} ({version})")
        return True
    except ImportError:
        return False


def _check_binary(dep: SystemDep, verbose: bool) -> bool:
    """Return True if the system binary is available on PATH."""
    found = shutil.which(dep.binary) is not None
    if verbose and found:
        print(f"    [OK] {dep.binary}")
    return found


def _check_tesseract_por(verbose: bool) -> bool:
    """Return True if Tesseract has the Portuguese ('por') language data installed."""
    if shutil.which("tesseract") is None:
        return False
    try:
        result = subprocess.run(
            ["tesseract", "--list-langs"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        output = result.stdout + result.stderr
        found = "por" in output.splitlines()
        if verbose and found:
            print("    [OK] tesseract language 'por' available")
        return found
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _check_model(model_deps: ModelDeps, verbose: bool) -> tuple[list[str], list[str]]:
    """
    Check all deps for a model group.

    Returns (ok_labels, missing_labels).
    """
    ok: list[str] = []
    missing: list[str] = []

    seen_system: set[str] = set()

    for dep in model_deps.python:
        if _check_python(dep, verbose):
            ok.append(dep.import_name)
        else:
            missing.append(f"pip install \"{dep.pip_name}\"  # {dep.description}")

    for dep in model_deps.system:
        key = dep.binary + dep.apt_package
        if key in seen_system:
            continue
        seen_system.add(key)

        # Special case: Tesseract Portuguese language data requires a different check
        if dep.apt_package == "tesseract-ocr-por":
            if _check_tesseract_por(verbose):
                ok.append("tesseract lang:por")
            else:
                missing.append(f"sudo apt-get install {dep.apt_package}  # {dep.description}")
        else:
            if _check_binary(dep, verbose):
                ok.append(dep.binary)
            else:
                missing.append(f"sudo apt-get install {dep.apt_package}  # {dep.description}")

    return ok, missing


def _run_checks(selected_models: list[str], verbose: bool) -> bool:
    """Run checks for each selected model group. Returns True if all pass."""
    groups: list[ModelDeps] = [BENCHMARK_DEPS] + [MODEL_DEPS[m] for m in selected_models]

    all_ok = True
    for group in groups:
        print(f"\n{'─' * 50}")
        print(f"  {group.name}")
        print(f"{'─' * 50}")

        ok, missing = _check_model(group, verbose)

        if not missing:
            status = f"[OK] {len(ok)} dependência(s) verificada(s)"
            print(f"  {status}")
        else:
            all_ok = False
            if ok:
                print(f"  [OK] {len(ok)} dependência(s) OK")
            print(f"  [MISSING] {len(missing)} dependência(s) ausente(s):")
            for item in missing:
                print(f"    → {item}")

    return all_ok


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verifica dependências do ambiente de avaliação OCR"
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help=(
            "Verificar apenas um modelo específico "
            f"({', '.join(MODEL_DEPS)}). Omita para verificar todos."
        ),
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Mostrar detalhes de cada dependência encontrada",
    )
    args = parser.parse_args()

    if args.model:
        name = args.model.strip().lower()
        if name not in MODEL_DEPS:
            print(f"Modelo desconhecido: '{name}'. Válidos: {list(MODEL_DEPS)}", file=sys.stderr)
            sys.exit(1)
        selected = [name]
    else:
        selected = list(MODEL_DEPS)

    print("=" * 50)
    print("  OCR DEPENDENCY CHECKER")
    print(f"  Python: {sys.version.split()[0]}  |  venv: {sys.prefix}")
    print("=" * 50)

    all_ok = _run_checks(selected, args.verbose)

    print(f"\n{'=' * 50}")
    if all_ok:
        print("  RESULTADO: todas as dependências estão instaladas.")
    else:
        print("  RESULTADO: dependências ausentes (veja os itens '[MISSING]' acima).")
    print("=" * 50)

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
