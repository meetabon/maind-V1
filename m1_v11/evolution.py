"""
EvolutionManager - управление размножением и эволюцией мета-параметров
Триггеры размножения и мутации
"""

import numpy as np

class EvolutionManager:
    def __init__(self):
        """Инициализация менеджера эволюции"""
        self.mutation_rate = 0.1  # ±10% мутации
        self.min_steps_for_reproduction = 200  # минимум шагов до размножения

    def check_reproduction_trigger(self, error_history, energy_history, step_count):
        """Проверка условий размножения"""
        if step_count < self.min_steps_for_reproduction:
            return False

        if len(error_history) < 100 or len(energy_history) < 100:
            return False

        # Условие 1: ошибка растет относительно недавнего среднего
        recent_error = np.mean(error_history[-50:])
        older_error = np.mean(error_history[-100:-50])
        error_growing = recent_error > older_error * 1.2  # рост на 20%

        # Условие 2: энергия падает (костыль v1.1)
        recent_energy = np.mean(energy_history[-50:])
        max_energy = np.max(energy_history)
        energy_low = recent_energy < max_energy * 0.1  # меньше 10% от максимума

        # Оба условия должны выполняться
        return error_growing and energy_low

    def mutate_params(self, meta_params):
        """Мутация мета-параметров"""
        new_params = {}

        for key, value in meta_params.items():
            # Случайный шум ±10%
            noise = np.random.normal(0, self.mutation_rate * value)
            new_value = value + noise

            # Ограничения на параметры
            if key == 'eta':  # скорость обучения
                new_value = np.clip(new_value, 0.001, 0.1)
            elif key == 'lambda':  # вес ошибки
                new_value = np.clip(new_value, 0.01, 1.0)
            elif key == 'long_link_prob':  # склонность к дальним связям
                new_value = np.clip(new_value, 0.01, 0.2)

            new_params[key] = new_value

        return new_params

    def get_fitness_score(self, error_history, energy_history, step_count):
        """Расчет приспособленности (для анализа)"""
        if len(error_history) == 0 or len(energy_history) == 0:
            return 0.0

        # Простая метрика: время жизни / средняя ошибка
        avg_error = np.mean(error_history)
        if avg_error == 0:
            avg_error = 0.001  # избегаем деления на ноль

        fitness = step_count / avg_error
        return fitness