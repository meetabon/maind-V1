"""
EnergySystem - управление энергетикой системы
Расчет затрат и поступлений энергии
"""

import numpy as np

class EnergySystem:
    def __init__(self):
        """Инициализация энергетической системы"""
        # Константы затрат
        self.c_a = 0.01   # стоимость активности
        self.c_w = 0.001  # стоимость весов
        self.c_d = 0.05   # стоимость дальних связей

    def calculate_cost(self, regions):
        """Расчет энергетических затрат за шаг"""
        total_cost = 0

        for region in regions:
            # Стоимость активности: c_a * Σ a_i
            activity_cost = self.c_a * region.get_total_activity()

            # Стоимость весов: c_w * Σ |w_ij|
            weight_cost = self.c_w * region.get_total_weights()

            # Стоимость дальних связей: c_d * (# дальних связей)
            long_links_cost = self.c_d * region.get_long_links_count()

            total_cost += activity_cost + weight_cost + long_links_cost

        return total_cost

    def calculate_gain(self, regions, region_signals, sdr_layers=None):
        """Расчет поступления энергии от среды и SDR слоев"""
        total_gain = 0

        for i, region in enumerate(regions):
            region_space = region_signals[i]

            # Поиск активных нейронов на положительных ячейках
            for nid, neuron in region.neurons.items():
                x, y, z = nid

                # Если нейрон активен и его ячейка = +1
                if neuron.activation == 1 and region_space[x, y, z] == 1:
                    total_gain += 1.0
                    # Ячейка поглощается (становится 0)
                    region_space[x, y, z] = 0

        # Дополнительный доход за успешные предсказания в SDR слоях
        if sdr_layers:
            for layer in sdr_layers:
                # Если нейрон активен и он был предсказан (predictive_state)
                # Даем бонус энергии
                matches = (layer.activations * layer.predictive_state).sum()
                total_gain += matches * 0.5

        return total_gain

    def get_energy_breakdown(self, regions):
        """Детальная разбивка энергетических затрат (для отладки)"""
        breakdown = {
            'activity': 0,
            'weights': 0,
            'long_links': 0
        }

        for region in regions:
            breakdown['activity'] += self.c_a * region.get_total_activity()
            breakdown['weights'] += self.c_w * region.get_total_weights()
            breakdown['long_links'] += self.c_d * region.get_long_links_count()

        return breakdown