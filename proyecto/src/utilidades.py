from pathlib import Path

def ensure_dir(path: str) -> None:
    """Crea un directorio si no existe."""
    Path(path).mkdir(parents=True, exist_ok=True)

def safe_div(a: float, b: float) -> float:
    """División segura (evita división por cero)."""
    return a / b if b != 0 else 0.0