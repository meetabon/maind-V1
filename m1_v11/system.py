"""
M1System - главный класс нейроморфной системы
Управляет 8 областями, энергетикой и эволюцией
"""

import numpy as np
from environment import Environment
from neuron import NeuralRegion
from energy import EnergySystem
from evolution import EvolutionManager
from sdr_memory import SDRLayer
from checkpoint_system import MemoryPersistence

class M1System:
    @property
    def weights_changed(self):
        """Агрегированный флаг изменения весов во всех слоях"""
        return any(layer.weights_changed for layer in self.sdr_layers)

    def __init__(self, meta_params=None):
        """Инициализация системы M1"""
        # Мета-параметры (наследуемые при размножении)
        if meta_params is None:
            self.meta_params = {
                'eta': 0.01,      # скорость обучения
                'lambda': 0.1,    # вес ошибки предсказания
                'long_link_prob': 0.05,  # склонность к дальним связям
                'anomaly_threshold': 0.3  # Порог аномалии
            }
        else:
            self.meta_params = meta_params.copy()
            if 'anomaly_threshold' not in self.meta_params:
                self.meta_params['anomaly_threshold'] = 0.3

        # Компоненты системы
        self.environment = Environment()

        # Переход к SDR-слоям
        # Слой -1: Буквы (32 нейрона, разреженность 3-15)
        # Вход из среды (8x8x8 = 512)
        self.layer_minus_1 = SDRLayer("Letters", 32, 3, 15, input_dim=512)

        # Слой 0: Слова (64 нейрона, разреженность 3-30)
        # Вход из Слоя -1 (32)
        self.layer_0 = SDRLayer("Words", 64, 3, 30, input_dim=32)

        # Слой +1: Фразы (124 нейрона, разреженность 3-60)
        # Вход из Слоя 0 (64)
        self.layer_plus_1 = SDRLayer("Phrases", 124, 3, 60, input_dim=64)

        # Слой +2: Понятия (124 нейрона, разреженность 3-60)
        # Вход из Слоя +1 (124)
        self.layer_plus_2 = SDRLayer("Concepts", 124, 3, 60, input_dim=124)

        # Слой +3: Резерв (124 нейрона)
        self.layer_plus_3 = SDRLayer("Reserve1", 124, 3, 60, input_dim=124)

        # Слой +4: Резерв (124 нейрона)
        self.layer_plus_4 = SDRLayer("Reserve2", 124, 3, 60, input_dim=124)

        # Список всех слоев для итерации
        self.sdr_layers = [
            self.layer_minus_1,
            self.layer_0,
            self.layer_plus_1,
            self.layer_plus_2,
            self.layer_plus_3,
            self.layer_plus_4
        ]

        # Инициализация обратных (top-down) связей
        # Используем малые веса для старта
        self.layer_minus_1.top_down_weights = np.random.uniform(0, 0.01, (32, 64))
        self.layer_0.top_down_weights = np.random.uniform(0, 0.01, (64, 124))
        self.layer_plus_1.top_down_weights = np.random.uniform(0, 0.01, (124, 124))

        self.regions = [NeuralRegion(i, self.meta_params) for i in range(8)]
        self.energy_system = EnergySystem()
        self.evolution = EvolutionManager()
        self.persistence = MemoryPersistence()

        # Флаги состояния
        self.need_teacher = False
        self.NEED_TEACHER = False # Для обратной совместимости с новыми требованиями

        # Состояние системы
        self.energy = len(self.regions) * 32 * 0.5  # N_neurons * 0.5
        self.step_count = 0
        self.alive = True

        # Статистика
        self.error_history = []
        self.energy_history = []

    def step(self, external_input=None):
        """Один шаг симуляции"""
        if not self.alive:
            return False

        self.step_count += 1

        # 1. Обновление среды
        self.environment.update(self.step_count)

        # 2. Получение сигналов (извне или из среды)
        if external_input is not None:
            env_input = external_input
        else:
            env_input = self.environment.space.flatten()

        # --- ФАЗА 1: ОБРАТНОЕ ПРЕДСКАЗАНИЕ (Сверху Вниз) ---
        # Понятия -> Фразы -> Слова -> Буквы
        # Слой +2 (Понятия) дает контекст для +1
        self.layer_plus_1.update_predictive_state(self.layer_plus_2.activations)
        # Слой +1 (Фразы) дает контекст для 0
        self.layer_0.update_predictive_state(self.layer_plus_1.activations)
        # Слой 0 (Слова) дает контекст для -1 (Буквы) - Эффект Т9
        self.layer_minus_1.update_predictive_state(self.layer_0.activations)

        # --- ФАЗА 2: ПРЯМАЯ АКТИВАЦИЯ (Снизу Вверх) ---
        # Слой -1 (Буквы)
        act_m1 = self.layer_minus_1.update(env_input)

        # Слой 0 (Слова)
        act_0 = self.layer_0.update(act_m1)

        # Слой +1 (Фразы)
        act_p1 = self.layer_plus_1.update(act_0)

        # Слой +2 (Понятия)
        act_p2 = self.layer_plus_2.update(act_p1)

        # Резервные слои
        act_p3 = self.layer_plus_3.update(act_p2)
        self.layer_plus_4.update(act_p3)

        # --- ФАЗА 3: ОБУЧЕНИЕ (Hebbian Learning) ---
        eta = self.meta_params['eta']
        self.layer_minus_1.learn(env_input, eta=eta)
        self.layer_minus_1.learn_top_down(self.layer_0.activations, eta=eta)

        self.layer_0.learn(act_m1, eta=eta)
        self.layer_0.learn_top_down(self.layer_plus_1.activations, eta=eta)

        self.layer_plus_1.learn(act_0, eta=eta)
        self.layer_plus_1.learn_top_down(self.layer_plus_2.activations, eta=eta)

        self.layer_plus_2.learn(act_p1, eta=eta)

        # Обучение резервных слоев
        self.layer_plus_3.learn(act_p2, eta=eta)
        self.layer_plus_4.learn(act_p3, eta=eta)

        # 4. Получение сигналов из среды для каждой области (старая логика)
        region_signals = self.environment.get_region_signals()

        # 5. Обновление нейронных областей
        total_error = 0
        for i, region in enumerate(self.regions):
            error = region.update(region_signals[i])
            total_error += error

        # 6. Расчет энергетических затрат
        energy_cost = self.energy_system.calculate_cost(self.regions)

        # Добавляем стоимость SDR слоев
        for layer in self.sdr_layers:
            energy_cost += 0.01 * layer.get_activity_count()
            energy_cost += 0.001 * layer.get_total_weights()

        # 7. Получение энергии от среды
        energy_gain = self.energy_system.calculate_gain(self.regions, region_signals, sdr_layers=self.sdr_layers)

        # 8. Обновление энергии
        self.energy += energy_gain - energy_cost

        # 9. Штраф за ошибки предсказания
        error_penalty = self.meta_params['lambda'] * total_error
        self.energy -= error_penalty

        # 10. Детекция аномалий (HTM Anomaly)
        # anomaly_score = np.mean(np.abs(self.layer_minus_1.potentials - self.layer_minus_1.predictive_state))
        # Примечание: predictive_state бинаризован, potentials - сырые суммы.
        # Приведем к одной шкале для адекватности mean(abs).
        norm_potentials = self.layer_minus_1.potentials / (self.layer_minus_1.potentials.max() if self.layer_minus_1.potentials.max() > 0 else 1.0)
        anomaly_score = np.mean(np.abs(norm_potentials - self.layer_minus_1.predictive_state))

        self.NEED_TEACHER = anomaly_score > self.meta_params.get('anomaly_threshold', 0.4)
        self.need_teacher = self.NEED_TEACHER # Синхронизация флагов

        # 11. Персистентность (сохранение состояния)
        if self.weights_changed:
            self.persistence.save(self)

        # 8. Проверка выживания
        if self.energy <= 0:
            self.alive = False
            return False

        # 9. Обновление статистики
        self.error_history.append(total_error)
        self.energy_history.append(self.energy)

        return True

    def should_reproduce(self):
        """Проверка условий размножения"""
        return self.evolution.check_reproduction_trigger(
            self.error_history,
            self.energy_history,
            self.step_count
        )

    def reproduce(self):
        """Создание новой системы с мутированными параметрами"""
        new_meta_params = self.evolution.mutate_params(self.meta_params)
        return M1System(new_meta_params)

    def get_avg_error(self):
        """Средняя ошибка за последние шаги"""
        if len(self.error_history) == 0:
            return 0.0
        return np.mean(self.error_history[-100:])

    def get_long_links_count(self):
        """Количество дальних связей"""
        count = 0
        for region in self.regions:
            count += region.get_long_links_count()
        return count

    def is_self_aware(self):
        """Примитивная проверка самосознания (для логирования)"""
        # TODO: реализовать отслеживание инварианта "я/не-я"
        return False