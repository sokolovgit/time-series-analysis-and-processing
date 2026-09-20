# Репозиторій курсу «Аналіз та обробка часових рядів (Time Series)»

_default:
    @just --list

# Синхронізувати спільний .venv після зміни залежностей будь-де
sync:
    uv sync --all-packages

# Відформатувати і полагодити те, що виправляється автоматично
fmt:
    ruff format .
    ruff check --fix .

# Перевірити без змін
lint:
    ruff format --check --diff .
    ruff check .

# Перевірити типи (той самий тайпчекер, що в редакторі)
types:
    uvx ty check

# Прокинути команду в лабу: just lab 02 all
lab NUM *ARGS:
    cd labs/{{ NUM }}-*/project && just {{ ARGS }}
