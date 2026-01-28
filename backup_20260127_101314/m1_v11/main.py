"""
M1 v1.1 - Минимально жизнеспособная нейроморфная система
Главный файл запуска симуляции
"""

from system import M1System
from logger import SimulationLogger
import numpy as np

def main():
    """Запуск симуляции M1 v1.1"""
    print("=== M1 v1.1 Нейроморфная система ===")
    
    # Параметры симуляции
    max_steps = 10000
    generation = 1
    
    # Инициализация
    system = M1System()
    logger = SimulationLogger()
    
    print(f"Поколение {generation}: Начальная энергия = {system.energy:.2f}")
    
    # Основной цикл симуляции
    for step in range(max_steps):
        # Шаг системы
        alive = system.step()
        
        # Логирование
        logger.log_step(step, system)
        
        # Проверка условий остановки
        if not alive:
            print(f"Шаг {step}: Система умерла (энергия = {system.energy:.2f})")
            break
            
        # Проверка размножения
        if system.should_reproduce():
            print(f"Шаг {step}: Размножение! Энергия = {system.energy:.2f}")
            system = system.reproduce()
            generation += 1
            print(f"Поколение {generation}: Новая система создана")
        
        # Периодический вывод
        if step % 1000 == 0:
            print(f"Шаг {step}: E={system.energy:.2f}, Ошибка={system.get_avg_error():.3f}")
    
    # Сохранение результатов
    logger.save_to_csv("m1_v11_results.csv")
    print(f"Симуляция завершена. Результаты сохранены в m1_v11_results.csv")

if __name__ == "__main__":
    main()