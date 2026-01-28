#!/usr/bin/env python3
import sys
import os

print("=== Тест запуска M1 v1.1 ===")
print(f"Python версия: {sys.version}")
print(f"Текущая директория: {os.getcwd()}")
print(f"Содержимое m1_v11/:")

try:
    files = os.listdir("m1_v11")
    for f in files:
        print(f"  - {f}")
except Exception as e:
    print(f"Ошибка чтения директории: {e}")

print("\nПопытка импорта модулей...")

try:
    sys.path.append("m1_v11")
    from system import M1System
    print("✓ system.py импортирован успешно")
    
    from logger import SimulationLogger
    print("✓ logger.py импортирован успешно")
    
    import numpy as np
    print("✓ numpy доступен")
    
    print("\nЗапуск симуляции...")
    system = M1System()
    logger = SimulationLogger()
    
    print(f"Система создана. Начальная энергия: {system.energy:.2f}")
    
    # Запустим несколько шагов
    for step in range(10):
        alive = system.step()
        logger.log_step(step, system)
        if not alive:
            print(f"Система умерла на шаге {step}")
            break
        print(f"Шаг {step}: E={system.energy:.2f}, Ошибка={system.get_avg_error():.3f}")
    
    print("Тест завершен успешно!")
    
except Exception as e:
    print(f"Ошибка: {e}")
    import traceback
    traceback.print_exc()