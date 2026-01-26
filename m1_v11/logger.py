"""
SimulationLogger - логирование данных симуляции в CSV
Отслеживание ключевых метрик системы
"""

import csv
import numpy as np

class SimulationLogger:
    def __init__(self):
        """Инициализация логгера"""
        self.data = []
        self.headers = [
            'step', 'energy', 'avg_error', 'long_links', 'self_aware',
            'total_activity', 'total_weights', 'generation'
        ]
        self.current_generation = 1
    
    def log_step(self, step, system):
        """Логирование одного шага симуляции"""
        # Сбор метрик
        energy = system.energy
        avg_error = system.get_avg_error()
        long_links = system.get_long_links_count()
        self_aware = system.is_self_aware()
        
        # Дополнительные метрики
        total_activity = sum(region.get_total_activity() for region in system.regions)
        total_weights = sum(region.get_total_weights() for region in system.regions)
        
        # Запись данных
        row = [
            step, energy, avg_error, long_links, int(self_aware),
            total_activity, total_weights, self.current_generation
        ]
        
        self.data.append(row)
    
    def log_reproduction(self):
        """Отметка о размножении (смена поколения)"""
        self.current_generation += 1
    
    def save_to_csv(self, filename):
        """Сохранение данных в CSV файл"""
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(self.headers)
            writer.writerows(self.data)
        
        print(f"Данные сохранены в {filename} ({len(self.data)} записей)")
    
    def get_summary_stats(self):
        """Получение сводной статистики"""
        if not self.data:
            return {}
        
        # Конвертируем в numpy для удобства
        data_array = np.array(self.data)
        
        stats = {
            'total_steps': len(self.data),
            'generations': self.current_generation,
            'avg_energy': np.mean(data_array[:, 1]),
            'avg_error': np.mean(data_array[:, 2]),
            'max_long_links': np.max(data_array[:, 3]),
            'survival_time': len(self.data)
        }
        
        return stats
    
    def print_summary(self):
        """Вывод сводной статистики"""
        stats = self.get_summary_stats()
        
        print("\n=== Сводка симуляции ===")
        print(f"Общее количество шагов: {stats.get('total_steps', 0)}")
        print(f"Количество поколений: {stats.get('generations', 0)}")
        print(f"Средняя энергия: {stats.get('avg_energy', 0):.3f}")
        print(f"Средняя ошибка: {stats.get('avg_error', 0):.3f}")
        print(f"Максимум дальних связей: {stats.get('max_long_links', 0)}")
        print(f"Время выживания: {stats.get('survival_time', 0)} шагов")