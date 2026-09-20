"""Параметри варіанта та обчислювального експерименту."""

from dataclasses import dataclass

# Варіант = місяць народження (табл. 1 додатку 1, docs/task/lab-1.pdf).
VARIANT = 8

# Умови варіанта 8:
#   I рівень   - похибка: нормальна; тренд: квадратичний; реальні дані: 1 показник.
#   II рівень  - похибка: нормальна, експоненційна; тренд: квадратичний, лінійний;
#                комбінаторика похибка/тренд довільна; реальні дані: 3 показники.
#   III рівень – додатково парсинг обраного сайту, збереження результату у файл,
#                оцінка динаміки тренду реальних даних і синтез подібної їм моделі.
NOISE_LAWS = ("normal", "exponential")
TREND_LAWS = ("quadratic", "linear")

# Рівень складності завдання: 1, 2 або 3. III рівень – максимальні 10 балів.
LEVEL = 3


@dataclass(frozen=True, slots=True)
class Settings:
    """Налаштування прогону, які можна перевизначити з CLI."""

    variant: int = VARIANT
    level: int = LEVEL
    sample_size: int = 500
    seed: int = 42

    @property
    def noise_laws(self) -> tuple[str, ...]:
        """Закони розподілу похибки, задіяні на обраному рівні складності."""
        return NOISE_LAWS[:1] if self.level == 1 else NOISE_LAWS

    @property
    def trend_laws(self) -> tuple[str, ...]:
        """Закони зміни досліджуваного процесу, задіяні на обраному рівні складності."""
        return TREND_LAWS[:1] if self.level == 1 else TREND_LAWS
