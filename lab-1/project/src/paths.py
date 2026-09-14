"""Шляхи до даних лабораторної роботи."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LAB_ROOT = PROJECT_ROOT.parent

DOCS = LAB_ROOT / "docs"
TASK = DOCS / "task"
SAMPLES = DOCS / "samples"
REPORT = DOCS / "report"

# Графіки та експорти лежать поруч зі звітом і трекаються в гіті -
# звіт має бути самодостатнім у репозиторії.
ASSETS = REPORT / "assets"

# Реальні дані з прикладів викладача – читаємо на місці, не копіюємо в project/.
OSCHADBANK_USD = SAMPLES / "Oschadbank (USD).xls"
LIVING_WAGE = SAMPLES / "Minfin_LivingWage.xlsx"


def ensure_assets() -> Path:
    """Створити каталог для графіків та експортів і повернути його."""
    ASSETS.mkdir(parents=True, exist_ok=True)
    return ASSETS
