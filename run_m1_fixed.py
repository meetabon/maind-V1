#!/usr/bin/env python3
"""
Исправленный запуск M1 v1.1 с правильными импортами
"""

import sys
import os

# Добавляем путь к модулям
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'm1_v11'))

try:
    from system import M1System
    from logger import SimulationLogger
    import numpy as np
    
    print("=== M1 v1.1 Нейроморфная система ===")
    print("✓ Все модули импортированы успешно")
    
    # Параметры симуляции
    max_steps = 300  # ×3 от 100
    generation = 1
    
    # Инициализация
    print("Создание системы...")
    system = M1System()
    logger = SimulationLogger()
    
    print(f"Поколение {generation}: Начальная энергия = {system.energy:.2f}")
    
    diagnostics = True  # включить подробную диагностическую печать

    # Основной цикл симуляции
    for step in range(max_steps):
        # Шаг системы
        alive = system.step()

        # Логирование
        logger.log_step(step, system)

        # Диагностика: печатаем разложение по энергиям каждые 10 шагов или при смерти
        if diagnostics and hasattr(system, 'last_diagnostics'):
            diag = system.last_diagnostics
            if (step % 10 == 0) or (not alive):
                # Собираем последние ошибки по регионам для логирования
                region_errs = [ (system.region_errors[i][-1] if len(system.region_errors[i])>0 else 0.0) for i in range(len(system.region_errors)) ]
                print(f"[DIAG] Шаг={step} E_before={diag['energy_before']:.2f} gain={diag['energy_gain']:.2f} cost={diag['energy_cost']:.2f} base={diag['base_cost']:.3f} err_pen={diag['error_penalty']:.3f} conf={diag['conflict']:.3f} conf_pen={diag['conflict_penalty']:.3f} n_active={diag['n_active']}")
                print(f"       region_errors={region_errs}")
        
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
        if step % 10 == 0:
            print(f"Шаг {step}: E={system.energy:.2f}, Ошибка={system.get_avg_error():.3f}")
    
    # Сохранение результатов
    logger.save_to_csv("m1_v11_results.csv")
    logger.save_detailed_data()  # сохранение данных для графиков
    logger.print_summary()
    print(f"Симуляция завершена.")

except ImportError as e:
    print(f"✗ Ошибка импорта: {e}")
    print("Проверьте, что все файлы находятся в папке m1_v11/")
    
except Exception as e:
    print(f"✗ Ошибка выполнения: {e}")
    import traceback
    traceback.print_exc()