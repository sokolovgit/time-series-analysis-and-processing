# Hierarchical Time Series Forecasting for Ukrainian Air Alerts

Ієрархічне прогнозування кількості повітряних тривог по областях України з агрегацією до загальної кількості по країні.

## Структура проекту

```
alert_hts/
├── data_loader.py              # Завантаження та обробка даних
├── visualization.py            # Функції візуалізації
├── main.py                     # Головний скрипт (вибір методу)
│
├── optimal_combination/        # Optimal Combination (OLS)
│   ├── forecaster.py
│   ├── main.py
│   └── __init__.py
│
├── bottom_up/                  # Bottom-Up метод
│   ├── forecaster.py
│   ├── main.py
│   └── __init__.py
│
├── top_down/                   # Top-Down метод
│   ├── forecaster.py
│   ├── main.py
│   └── __init__.py
│
└── middle_out/                 # Middle-Out метод
    ├── forecaster.py
    ├── main.py
    └── __init__.py
```

## Методи прогнозування

### 1. Optimal Combination (OLS) - `optimal_combination/`
**Метод**: Прогнозує на всіх рівнях, потім оптимально узгоджує через OLS
- ✅ Найкраща точність
- ✅ Perfect консистентність ієрархії
- ⚠️ Складніші обчислення

### 2. Bottom-Up - `bottom_up/`
**Метод**: Прогнозує тільки на нижньому рівні (області), потім сумує
- ✅ Зберігає деталізацію
- ✅ Автоматична консистентність
- ⚠️ Накопичення помилок

### 3. Top-Down - `top_down/`
**Метод**: Прогнозує на верхньому рівні (total), потім розподіляє вниз
- ✅ Стабільні прогнози на верхньому рівні
- ⚠️ Втрата деталізації на нижньому рівні

### 4. Middle-Out - `middle_out/`
**Метод**: Прогнозує на проміжному рівні, потім вгору (BU) і вниз (TD)
- ✅ Баланс між деталізацією та стабільністю
- ⚠️ Потребує вибору проміжного рівня
- ℹ️ Для 2-рівневої ієрархії ≈ Bottom-Up

## Використання

### Варіант 1: Головний скрипт з вибором методу

```bash
cd alert_hts

# Optimal Combination (за замовчуванням)
python main.py optimal

# Bottom-Up
python main.py bottom_up

# Top-Down
python main.py top_down

# Middle-Out
python main.py middle_out
```

### Варіант 2: Окремі скрипти для кожного методу

```bash
# Optimal Combination
python optimal_combination/main.py

# Bottom-Up
python bottom_up/main.py

# Top-Down
python top_down/main.py

# Middle-Out
python middle_out/main.py
```

## Встановлення

```bash
cd alert_hts

# Створити віртуальне середовище
python3 -m venv .venv

# Активувати
source .venv/bin/activate  # macOS/Linux
# або
.venv\Scripts\activate      # Windows

# Встановити залежності
pip install -r requirements.txt
```

## Що робить кожен скрипт

1. **Завантажує дані** з `../data/official_data_uk.csv`
2. **Фільтрує** тільки oblast-level тривоги
3. **Підраховує** кількість тривог по областях за день
4. **Створює ієрархію**: Області → Загальна кількість
5. **Тренує моделі** на всіх часових рядах
6. **Застосовує reconciliation** метод
7. **Показує інтерактивні графіки**:
   - Grid з усіма 25 областями
   - Загальна кількість по Україні

## Результати

**Якість прогнозування:**
- **Optimal (OLS)**: Найкраща точність, MAPE ~5-15%
- **Bottom-Up**: Добра деталізація, можливе накопичення помилок
- **Top-Down**: Стабільні прогнози на верхньому рівні
- **Middle-Out**: Баланс між методами

**Консистентність ієрархії:**
- **Optimal (OLS)**: Perfect (помилка ≈ 0)
- **Bottom-Up**: Perfect (автоматично)
- **Top-Down**: Perfect (автоматично)
- **Middle-Out**: Perfect (автоматично)

## Порівняння методів

| Метод | Точність | Деталізація | Консистентність | Складність |
|-------|----------|-------------|------------------|------------|
| **Optimal (OLS)** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Висока |
| **Bottom-Up** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Низька |
| **Top-Down** | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Низька |
| **Middle-Out** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Середня |

## Примітки

- Дані агрегуються до денного рівня (кількість тривог на день)
- Тільки oblast-level тривоги (виключено raion та hromada)
- Holt-Winters працює краще для стабільних часових рядів
- Для волатильних областей можуть бути більші помилки

## Troubleshooting

Якщо помилки з `scikit-hts` на Python 3.10+:

```bash
# Виправити collections.Iterable import
nano .venv/lib/python3.10/site-packages/hts/utilities/distribution.py
# Змінити: from collections import Iterable
# На: from collections.abc import Iterable
```

Або використати Python 3.8-3.9.
