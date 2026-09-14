"""Генерує reference.docx зі стилями оформлення звіту для pandoc.

Word і LibreOffice не потрібні: шаблон береться з дефолтного reference.docx pandoc,
після чого патчаться styles.xml (шрифт, кегль, інтервал, абзац, заголовки, таблиці)
і document.xml (формат аркуша та поля).

Параметри зняті з оформлення курсової роботи: Times New Roman 14 pt, інтервал 1.5,
абзац 1.25 см, текст за шириною; таблиці 12 pt з одинарним інтервалом; заголовки
першого рівня по центру великими літерами; підписи рисунків по центру.

Запуск: just reference
"""

import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

REPORT_DIR = Path(__file__).resolve().parents[2] / "docs" / "report"
TARGET = REPORT_DIR / "reference.docx"

BODY_PT = 14  # основний текст
TABLE_PT = 12  # таблиці: 14 pt не влазить у широкі таблиці звіту
LINE_SPACING = 1.5  # міжрядковий інтервал основного тексту
INDENT_CM = 1.25  # абзацний відступ

# Поля з розділу "Технічні вимоги до звіту" завдання: 2 / 2 / 2 / 2.5 см.
MARGIN_TOP, MARGIN_RIGHT, MARGIN_BOTTOM, MARGIN_LEFT = 2.0, 2.0, 2.0, 2.5


def cm(value: float) -> int:
    """Сантиметри у twips (1 см = 567 twips)."""
    return round(value * 567)


def pt(value: float) -> int:
    """Пункти у половини пункта - саме так Word задає кегль."""
    return round(value * 2)


def spacing(line: float = 1.0, before: int = 0, after: int = 0) -> str:
    return (
        f'<w:spacing w:before="{before}" w:after="{after}"'
        f' w:line="{round(240 * line)}" w:lineRule="auto"/>'
    )


FONT = (
    '<w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"'
    ' w:cs="Times New Roman" w:eastAsia="Times New Roman"/>'
)
SIZE = f'<w:sz w:val="{pt(BODY_PT)}"/><w:szCs w:val="{pt(BODY_PT)}"/>'
TABLE_SIZE = f'<w:sz w:val="{pt(TABLE_PT)}"/><w:szCs w:val="{pt(TABLE_PT)}"/>'

CODE_PT = 10  # лістинги коду
MONO = (
    '<w:rFonts w:ascii="Courier New" w:hAnsi="Courier New" w:cs="Courier New"/>'
    f'<w:sz w:val="{pt(CODE_PT)}"/><w:szCs w:val="{pt(CODE_PT)}"/>'
)

INDENT = f'<w:ind w:firstLine="{cm(INDENT_CM)}"/>'
NO_INDENT = '<w:ind w:firstLine="0" w:left="0"/>'
JUSTIFY = '<w:jc w:val="both"/>'
CENTER = '<w:jc w:val="center"/>'
KEEP = "<w:keepNext/><w:keepLines/>"

# Рисунок і підпис під ним: по центру, одинарний інтервал, з повітрям навколо блока,
# але без розриву між самим рисунком і його підписом.
IMAGE = KEEP + spacing(before=240, after=60) + NO_INDENT + CENTER
CAPTION = spacing(before=0, after=240) + NO_INDENT + CENTER

# Кегль і міжрядковий задаються ТІЛЬКИ в docDefaults, а для таблиць – у стилі таблиці.
# Якби їх дублював стиль Normal, він перебивав би стиль таблиці: у Word пріоритет такий -
# docDefaults < стиль таблиці < стиль абзацу. Тому Normal задає лише відступ і вирівнювання,
# а розмір успадковується. Так списки в тексті йдуть 14 pt, а в клітинках – 12 pt.
BODY = INDENT + JUSTIFY

DOC_DEFAULTS = f"""<w:docDefaults>
<w:rPrDefault><w:rPr>{FONT}{SIZE}<w:lang w:val="uk-UA"/></w:rPr></w:rPrDefault>
<w:pPrDefault><w:pPr>{spacing(LINE_SPACING)}</w:pPr></w:pPrDefault>
</w:docDefaults>"""

# styleId -> (вміст w:pPr, вміст w:rPr)
STYLES: dict[str, tuple[str, str]] = {
    "Normal": (BODY, FONT),
    "BodyText": (BODY, ""),
    "FirstParagraph": (BODY, ""),
    # Compact дістається і спискам у тексті, і клітинкам таблиць, тому задає лише відступ.
    "Compact": (NO_INDENT, ""),
    # Розділи звіту: по центру, великими літерами, не відриваються від тексту.
    "Heading1": (
        KEEP + spacing(LINE_SPACING, before=240, after=120) + NO_INDENT + CENTER,
        "<w:b/><w:caps/>" + SIZE,
    ),
    "Heading2": (
        KEEP + spacing(LINE_SPACING, before=240, after=120) + NO_INDENT,
        "<w:b/>" + SIZE,
    ),
    "Heading3": (
        KEEP + spacing(LINE_SPACING, before=120, after=60) + NO_INDENT,
        "<w:b/>" + SIZE,
    ),
    "Heading4": (
        KEEP + spacing(LINE_SPACING, before=120, after=60) + NO_INDENT,
        "<w:i/>" + SIZE,
    ),
    "Figure": (IMAGE, ""),
    "CaptionedFigure": (IMAGE, ""),
    "ImageCaption": (CAPTION, ""),
    "TableCaption": (CAPTION, ""),
    "Caption": (CAPTION, ""),
    "BlockText": (spacing() + NO_INDENT, ""),
    "VerbatimChar": ("", MONO),
}

BORDER = '<w:{side} w:val="single" w:sz="4" w:space="0" w:color="000000"/>'
GRID = "".join(
    BORDER.format(side=side) for side in ("top", "left", "bottom", "right", "insideH", "insideV")
)

# Таблиця: суцільна сітка, шапка напівжирна та по центру.
# Порядок вкладених елементів у стилі таблиці жорстко заданий схемою OOXML:
# pPr -> rPr -> tblPr -> tblStylePr. Порушиш – Word мовчки проігнорує стиль.
TABLE_STYLE = f"""<w:style w:type="table" w:default="1" w:styleId="Table">
<w:name w:val="Table"/><w:basedOn w:val="TableNormal"/><w:qFormat/>
<w:pPr>{spacing()}{NO_INDENT}</w:pPr>
<w:rPr>{FONT}{TABLE_SIZE}</w:rPr>
<w:tblPr><w:tblInd w:w="0" w:type="dxa"/><w:tblBorders>{GRID}</w:tblBorders>
<w:tblCellMar><w:top w:w="28" w:type="dxa"/><w:left w:w="85" w:type="dxa"/>
<w:bottom w:w="28" w:type="dxa"/><w:right w:w="85" w:type="dxa"/></w:tblCellMar></w:tblPr>
<w:tblStylePr w:type="firstRow"><w:pPr>{CENTER}</w:pPr><w:rPr><w:b/></w:rPr></w:tblStylePr>
</w:style>"""

# Власні стилі. pandoc шукає їх за назвою (w:name), а не за ідентифікатором.
CUSTOM_STYLES = f"""<w:style w:type="paragraph" w:styleId="TitlePage">
<w:name w:val="Титульний"/><w:basedOn w:val="Normal"/><w:qFormat/>
<w:pPr>{spacing() + NO_INDENT + CENTER}</w:pPr>
</w:style>
<w:style w:type="paragraph" w:styleId="TableCaptionOwn">
<w:name w:val="Підпис таблиці"/><w:basedOn w:val="Normal"/><w:qFormat/>
<w:pPr>{KEEP}{spacing(before=240, after=60)}{NO_INDENT}<w:jc w:val="left"/></w:pPr>
</w:style>
<w:style w:type="paragraph" w:styleId="SourceCode">
<w:name w:val="Source Code"/><w:basedOn w:val="Normal"/><w:qFormat/>
<w:pPr>{spacing()}{NO_INDENT}<w:jc w:val="left"/></w:pPr>
<w:rPr>{MONO}</w:rPr>
</w:style>
<w:style w:type="paragraph" w:styleId="TitlePageLeft">
<w:name w:val="Титульний лівий"/><w:basedOn w:val="Normal"/><w:qFormat/>
<w:pPr>{spacing()}<w:ind w:left="{cm(1.25)}" w:firstLine="0"/><w:jc w:val="left"/></w:pPr>
</w:style>
<w:style w:type="paragraph" w:styleId="FigureImage">
<w:name w:val="Рисунок"/><w:basedOn w:val="Normal"/><w:qFormat/>
<w:pPr>{IMAGE}</w:pPr>
</w:style>
<w:style w:type="paragraph" w:styleId="FigureCaption">
<w:name w:val="Підпис"/><w:basedOn w:val="Normal"/><w:qFormat/>
<w:pPr>{CAPTION}</w:pPr>
</w:style>"""

SECT_PR = (
    '<w:sectPr><w:pgSz w:w="11906" w:h="16838"/>'
    f'<w:pgMar w:top="{cm(MARGIN_TOP)}" w:right="{cm(MARGIN_RIGHT)}"'
    f' w:bottom="{cm(MARGIN_BOTTOM)}" w:left="{cm(MARGIN_LEFT)}"'
    ' w:header="708" w:footer="708" w:gutter="0"/>'
    "</w:sectPr>"
)


def patch_style(xml: str, style_id: str, p_pr: str, r_pr: str) -> str:
    """Замінити w:pPr та w:rPr у визначенні стилю, лишивши w:name, w:basedOn тощо."""
    pattern = re.compile(rf'(<w:style [^>]*w:styleId="{style_id}".*?)</w:style>', re.S)
    match = pattern.search(xml)
    if match is None:
        print(f"  ! стиль {style_id} не знайдено, пропущено", file=sys.stderr)
        return xml

    body = re.sub(r"<w:pPr>.*?</w:pPr>", "", match.group(1), flags=re.S)
    body = re.sub(r"<w:rPr>.*?</w:rPr>", "", body, flags=re.S)

    inserted = (f"<w:pPr>{p_pr}</w:pPr>" if p_pr else "") + (
        f"<w:rPr>{r_pr}</w:rPr>" if r_pr else ""
    )
    # Схема OOXML вимагає порядку pPr -> rPr -> tblPr, тому вставляємо перед tblPr.
    if (table_pr := body.find("<w:tblPr>")) != -1:
        body = body[:table_pr] + inserted + body[table_pr:]
    else:
        body += inserted
    return xml[: match.start()] + body + "</w:style>" + xml[match.end() :]


def build_styles(xml: str) -> str:
    xml = re.sub(r"<w:docDefaults>.*?</w:docDefaults>", DOC_DEFAULTS, xml, flags=re.S)
    for style_id, (p_pr, r_pr) in STYLES.items():
        xml = patch_style(xml, style_id, p_pr, r_pr)
    xml = re.sub(
        r'<w:style w:type="table"[^>]*w:styleId="Table".*?</w:style>',
        TABLE_STYLE,
        xml,
        flags=re.S,
    )
    return xml.replace("</w:styles>", CUSTOM_STYLES + "</w:styles>")


def build_document(xml: str) -> str:
    """Замінити властивості секції - саме там задаються формат аркуша й поля."""
    return re.sub(r"<w:sectPr>.*?</w:sectPr>", SECT_PR, xml, flags=re.S)


def main() -> None:
    if shutil.which("pandoc") is None:
        sys.exit("pandoc не знайдено: brew install pandoc")

    default = subprocess.run(
        ["pandoc", "--print-default-data-file", "reference.docx"],
        check=True,
        capture_output=True,
    ).stdout

    source = Path(__file__).with_suffix(".tmp.docx")
    source.write_bytes(default)
    patched = {"word/styles.xml": build_styles, "word/document.xml": build_document}

    TARGET.parent.mkdir(parents=True, exist_ok=True)
    with (
        zipfile.ZipFile(source) as src,
        zipfile.ZipFile(TARGET, "w", zipfile.ZIP_DEFLATED) as dst,
    ):
        for item in src.infolist():
            data = src.read(item.filename)
            if transform := patched.get(item.filename):
                data = transform(data.decode("utf-8")).encode("utf-8")
            dst.writestr(item, data)
    source.unlink()

    print(f"Шаблон стилів: {TARGET}")
    print(
        f"  текст: Times New Roman {BODY_PT} pt, інтервал {LINE_SPACING},"
        f" абзац {INDENT_CM} см, за шириною"
    )
    print(f"  таблиці: {TABLE_PT} pt, одинарний інтервал, суцільна сітка")
    print(f"  лістинги: Courier New {CODE_PT} pt, одинарний інтервал")
    print(
        f"  поля: зверху {MARGIN_TOP}, знизу {MARGIN_BOTTOM},"
        f" справа {MARGIN_RIGHT}, зліва {MARGIN_LEFT} см"
    )


if __name__ == "__main__":
    main()
