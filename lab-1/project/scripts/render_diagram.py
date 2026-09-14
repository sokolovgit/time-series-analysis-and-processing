"""Рендерить блок-схему звіту з блоку mermaid у report.md.

Джерело діаграми лишається в самому звіті, тому markdown коректно виглядає в
редакторі, а .docx отримує растрову картинку – pandoc mermaid не вміє.

Потребує node (npx). Перший запуск тягне mermaid-cli з headless Chromium,
далі бере їх із кешу npx.

Запуск: just diagram
"""

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPORT_DIR = Path(__file__).resolve().parents[2] / "docs" / "report"
SOURCE = REPORT_DIR / "report.md"
TARGET = REPORT_DIR / "assets" / "00_blockscheme.png"

MERMAID_CLI = "@mermaid-js/mermaid-cli@11"
SCALE = "2"  # 2x до CSS-пікселів: різко на друці й вкладається в ширину сторінки

MERMAID_BLOCK = re.compile(r"```mermaid\n(.*?)\n```", re.S)


def extract_diagram(markdown: str) -> str:
    match = MERMAID_BLOCK.search(markdown)
    if match is None:
        sys.exit(f"у {SOURCE.name} немає блоку mermaid")
    return match.group(1)


def main() -> None:
    if shutil.which("npx") is None:
        sys.exit("npx не знайдено: потрібен Node.js")

    diagram = extract_diagram(SOURCE.read_text(encoding="utf-8"))
    TARGET.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile("w", suffix=".mmd", encoding="utf-8", delete=False) as tmp:
        tmp.write(diagram)
        source = Path(tmp.name)

    try:
        subprocess.run(
            [
                "npx",
                "-y",
                MERMAID_CLI,
                "-i",
                str(source),
                "-o",
                str(TARGET),
                "-b",
                "white",
                "-s",
                SCALE,
            ],
            check=True,
        )
    finally:
        source.unlink()

    print(f"Блок-схема: {TARGET} ({TARGET.stat().st_size // 1024} КБ)")


if __name__ == "__main__":
    main()
