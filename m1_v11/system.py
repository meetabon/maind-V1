"""
M1System - главный класс нейроморфной системы
Управляет 8 областями, энергетикой и эволюцией
"""

import numpy as np
from environment import Environment
from neuron import NeuralRegion
from energy import EnergySystem
from evolution import EvolutionManager

class M1System:
    def __init__(self, meta_params=None):
        """Инициализация системы M1"""
        # Мета-параметры (наследуемые при размножении)
        if meta_params is None:
            self.meta_params = {
                'eta': 0.01,      # скорость обучения
                'lambda': 0.1,    # вес ошибки предсказания
                'long_link_prob': 0.05  # склонность к дальним связям
            }
        else:
            self.meta_params = meta_params.copy()
        
        # Компоненты системы
        self.environment = Environment()
        self.regions = [NeuralRegion(i, self.meta_params) for i in range(8)]
        self.energy_system = EnergySystem()
        self.evolution = EvolutionManager()
        
        # Состояние системы
        self.energy = len(self.regions) * 32 * 0.5  # N_neurons * 0.5
        self.step_count = 0
        self.alive = True
        
        # Статистика
        self.error_history = []
        self.energy_history = []
        
    def step(self):
        """Один шаг симуляции"""
        if not self.alive:
            return False
            
        self.step_count += 1
        
        # 1. Обновление среды
        self.environment.update(self.step_count)
        
        # 2. Получение сигналов из среды для каждой области
        region_signals = self.environment.get_region_signals()
        
        # 3. Обновление нейронных областей
        total_error = 0
        for i, region in enumerate(self.regions):
            error = region.update(region_signals[i])
            total_error += error
        
        # 4. Расчет энергетических затрат
        energy_cost = self.energy_system.calculate_cost(self.regions)
        
        # 5. Получение энергии от среды
        energy_gain = self.energy_system.calculate_gain(self.regions, region_signals)
        
        # 6. Обновление энергии
        self.energy += energy_gain - energy_cost
        
        # 7. Штраф за ошибки предсказания
        error_penalty = self.meta_params['lambda'] * total_error
        self.energy -= error_penalty
        
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