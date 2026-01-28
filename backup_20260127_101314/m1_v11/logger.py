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
            'total_activity', 'total_weights', 'generation', 'perception_conflict'
        ]
        self.current_generation = 1
        
        # Дополнительные данные для анализа
        self.activities_per_region = [[] for _ in range(8)]  # активность по регионам
        self.errors_per_region = [[] for _ in range(8)]  # ошибки по регионам
        self.energy_timeline = []  # временной ряд энергии
        self.error_timeline = []   # временной ряд ошибок
        self.conflict_timeline = []  # временной ряд перцептивного конфликта
    
    def log_step(self, step, system):
        """Логирование одного шага симуляции"""
        # Сбор метрик
        energy = system.energy
        avg_error = system.get_avg_error()
        long_links = system.get_long_links_count()
        self_aware = system.is_self_aware()
        perception_conflict = system.get_perception_conflict()  # новая метрика
        
        # Дополнительные метрики
        total_activity = sum(region.get_total_activity() for region in system.regions)
        total_weights = sum(region.get_total_weights() for region in system.regions)
        
        # Запись данных
        row = [
            step, energy, avg_error, long_links, int(self_aware),
            total_activity, total_weights, self.current_generation, perception_conflict
        ]
        
        self.data.append(row)
        
        # Сбор детализированных данных для анализа
        self.energy_timeline.append(energy)
        self.error_timeline.append(avg_error)
        self.conflict_timeline.append(perception_conflict)  # логирование конфликта
        
        # Активность по каждому региону
        for i, region in enumerate(system.regions):
            activity = region.get_total_activity()
            self.activities_per_region[i].append(activity)
            
            # Ошибки по регионам
            if len(system.region_errors[i]) > 0:
                region_error = system.region_errors[i][-1]
                self.errors_per_region[i].append(region_error)
    
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
    
    def save_detailed_data(self):
        """Сохранение детализированных данных для графиков"""
        # 1. Временные ряды энергии, ошибок и конфликтов
        with open('m1_energy_error_timeline.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['step', 'energy', 'error', 'conflict'])
            for i, (e, err, conf) in enumerate(zip(self.energy_timeline, self.error_timeline, self.conflict_timeline)):
                writer.writerow([i, e, err, conf])
        
        # 2. Активность по регионам
        with open('m1_activity_per_region.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            headers = ['step'] + [f'region_{i}' for i in range(8)]
            writer.writerow(headers)
            
            max_steps = max(len(acts) for acts in self.activities_per_region)
            for step in range(max_steps):
                row = [step]
                for i in range(8):
                    if step < len(self.activities_per_region[i]):
                        # Активность в % (всего 64 нейрона на регион)
                        activity_pct = (self.activities_per_region[i][step] / 64.0) * 100
                        row.append(activity_pct)
                    else:
                        row.append(0)
                writer.writerow(row)
        
        # 3. Корреляция ошибок между регионами
        errors_array = np.array([
            errors + [0] * (len(self.errors_per_region[0]) - len(errors))
            for errors in self.errors_per_region
        ])
        
        if errors_array.shape[1] > 0:
            correlation_matrix = np.corrcoef(errors_array)
        else:
            correlation_matrix = np.zeros((8, 8))
        
        with open('m1_error_correlation.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            headers = ['region'] + [f'region_{i}' for i in range(8)]
            writer.writerow(headers)
            
            for i in range(8):
                row = [f'region_{i}'] + [f'{correlation_matrix[i, j]:.3f}' for j in range(8)]
                writer.writerow(row)
        
        print("✓ Детализированные данные сохранены:")
        print("  - m1_energy_error_timeline.csv (с конфликтом)")
        print("  - m1_activity_per_region.csv")
        print("  - m1_error_correlation.csv")