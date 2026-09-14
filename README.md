# Аналіз та обробка часових рядів (Time Series)

Лабораторні роботи з дисципліни. КПІ, 4 курс, 1 семестр.
Сертифікатна програма Data Science із Sigma Software.

Матеріали дисципліни: [Google Drive](https://drive.google.com/drive/folders/1oPKrjJeMTcH9jHxhK2lcVwx2_kSDo2y4)

## Стек

Python 3.13, [uv](https://docs.astral.sh/uv/) для залежностей,
[ruff](https://docs.astral.sh/ruff/) для лінту й форматування,
[just](https://just.systems/) для скриптів. numpy / pandas / scipy / matplotlib,
requests + BeautifulSoup для парсингу.

## Запуск

```sh
cd lab-1/project
just sync   # встановити залежності
just run    # запустити
just check  # лінт + перевірка форматування
```

## Структура

| Шлях | Що це |
| --- | --- |
| `docs/` | матеріали викладача: силабус, завдання ЛР 1–9, МКР, тест |
| `lab-N/docs/task/` | завдання конкретної лаби + приклад звіту |
| `lab-N/docs/samples/` | приклади коду та дані від викладача |
| `lab-N/docs/report/` | власний звіт |
| `lab-N/docs/report/assets/` | графіки та експорти, згенеровані програмою |
| `lab-N/project/src/` | код лаби: модулі пласко, `main.py` – точка входу |
