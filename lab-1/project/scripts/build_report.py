"""Збирає звіт report.md у .docx зі стилями ДСТУ через pandoc.

Перед конвертацією markdown готується: титульний аркуш виноситься в окремий стиль
і закінчується розривом сторінки, а блок mermaid замінюється на зображення.

Запуск: just report [ім'я_файлу]
"""

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPORT_DIR = Path(__file__).resolve().parents[2] / "docs" / "report"
SOURCE = REPORT_DIR / "report.md"
REFERENCE = REPORT_DIR / "reference.docx"
BLOCK_SCHEME = REPORT_DIR / "assets" / "00_blockscheme.png"

# Блок-схема висока: без явної висоти pandoc масштабує її по ширині сторінки і
# отримує 26.5 см при текстовому полі 25.7 см - рисунок не влазить разом із підписом.
BLOCK_SCHEME_HEIGHT = "22cm"

# Рядки, які треба центрувати: самотнє зображення і підпис під ним.
IMAGE_LINE = re.compile(r"^!\[[^\]]*\]\([^)]+\)(\{[^}]*\})?$")
FIGURE_CAPTION = re.compile(r"^Рисунок \d+")
TABLE_CAPTION = re.compile(r"^Таблиця \d+")

# [[space:N]] у титулці розгортається в N порожніх абзаців - ними титулка
# розтягується на всю сторінку.
SPACER = re.compile(r"^\[\[space:(\d+)\]\]$")

# Розрив сторінки у форматі, який pandoc пропускає в .docx як є.
PAGE_BREAK = '\n```{=openxml}\n<w:p><w:r><w:br w:type="page"/></w:r></w:p>\n```\n'


def expand_spacers(markdown: str) -> str:
    """Розгорнути [[space:N]] у N порожніх абзаців."""
    lines = []
    for line in markdown.split("\n"):
        if match := SPACER.match(line):
            lines += ["&nbsp;", ""] * int(match.group(1))
        else:
            lines.append(line)
    return "\n".join(lines)


def split_title(markdown: str) -> tuple[str, str]:
    """Відділити титульний аркуш - усе до першої горизонтальної лінії."""
    title, separator, body = markdown.partition("\n---\n")
    return (title, body) if separator else ("", markdown)


def replace_mermaid(markdown: str) -> str:
    """Замінити блок mermaid на зображення блок-схеми, якщо воно згенероване."""
    pattern = re.compile(r"```mermaid\n.*?\n```", re.S)
    if not pattern.search(markdown):
        return markdown
    relative = BLOCK_SCHEME.relative_to(REPORT_DIR).as_posix()
    if BLOCK_SCHEME.exists():
        image = f"![Блок-схема алгоритму]({relative}){{height={BLOCK_SCHEME_HEIGHT}}}"
        return pattern.sub(image, markdown)
    print(
        f"  ! {relative} не знайдено - блок-схему в .docx пропущено."
        "\n    Згенеруй її командою `just diagram`.",
        file=sys.stderr,
    )
    return pattern.sub("", markdown)


def center_figures(markdown: str) -> str:
    """Розвести зображення й підписи по власних стилях: у них різні відступи."""
    lines = []
    for line in markdown.split("\n"):
        if IMAGE_LINE.match(line):
            style = "Рисунок"
        elif FIGURE_CAPTION.match(line):
            style = "Підпис"
        elif TABLE_CAPTION.match(line):
            # За ДСТУ підпис таблиці стоїть над нею і вирівнюється зліва.
            style = "Підпис таблиці"
        else:
            lines.append(line)
            continue
        lines += [f'::: {{custom-style="{style}"}}', line, ":::"]
    return "\n".join(lines)


def prepare(markdown: str) -> str:
    title, body = split_title(markdown)
    body = center_figures(replace_mermaid(body))
    if not title:
        return body
    title = expand_spacers(title)
    return f'::: {{custom-style="Титульний"}}\n{title}\n:::\n{PAGE_BREAK}\n{body}'


def main() -> None:
    parser = argparse.ArgumentParser(description="Збірка звіту в .docx")
    parser.add_argument(
        "name",
        nargs="?",
        default="report.docx",
        help="ім'я файлу; для здачі - Прізвище_Ім'я_група_1.docx",
    )
    args = parser.parse_args()

    if shutil.which("pandoc") is None:
        sys.exit("pandoc не знайдено: brew install pandoc")
    if not REFERENCE.exists():
        sys.exit(f"немає шаблону {REFERENCE.name}: спершу `just reference`")

    target = REPORT_DIR / args.name
    prepared = prepare(SOURCE.read_text(encoding="utf-8"))

    # Проміжний файл кладемо поруч зі звітом, щоб відносні шляхи до assets/ збіглися.
    with tempfile.NamedTemporaryFile(
        "w", suffix=".md", dir=REPORT_DIR, encoding="utf-8", delete=False
    ) as tmp:
        tmp.write(prepared)
        prepared_file = Path(tmp.name)

    try:
        subprocess.run(
            [
                "pandoc",
                str(prepared_file),
                "-o",
                str(target),
                f"--reference-doc={REFERENCE}",
                f"--resource-path={REPORT_DIR}",
                "--from=markdown-implicit_figures+tex_math_dollars+pipe_tables+raw_attribute",
            ],
            check=True,
        )
    finally:
        prepared_file.unlink()

    print(f"Звіт: {target} ({target.stat().st_size // 1024} КБ)")


if __name__ == "__main__":
    main()
