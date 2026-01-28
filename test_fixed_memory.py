#!/usr/bin/env python3
"""
Тест исправленной системы памяти
Сравнение: до исправлений vs после исправлений
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'm1_v11'))

from system import M1System
from logger import SimulationLogger

def run_fixed_test():
    """Запуск теста с исправленной памятью"""
    print("=== ТЕСТ ИСПРАВЛЕННОЙ ПАМЯТИ ===")
    
    max_steps = 300
    
    print("\n1. Тест A: без шумов (создание памяти)")
    system_a = M1System()
    # отключаем шумы
    system_a.environment.observation_noise['enabled'] = False
    system_a.environment.noise_params['enabled'] = False
    
    logger_a = SimulationLogger()
    for step in range(max_steps):
        alive = system_a.step()
        logger_a.log_step(step, system_a)
        if not alive:
            break
        if step % 50 == 0:
            print(f"  Шаг {step}: E={system_a.energy:.2f}, Ошибка={system_a.get_avg_error():.4f}")
    
    logger_a.save_to_csv("fixed_no_noise_results.csv")
    
    # Сохраняем память
    region_errors = [np.array(errs) for errs in system_a.region_errors]
    recent_errors = np.array(system_a.recent_errors)
    recent_conflicts = np.array(system_a.recent_conflicts)
    recent_activity = np.array(system_a.recent_activity)
    
    np.savez("fixed_memory_snapshot.npz",
             memory_strength=system_a.memory_strength,
             region_errors=np.array(region_errors, dtype=object),
             recent_errors=recent_errors,
             recent_conflicts=recent_conflicts,
             recent_activity=recent_activity,
             long_memory=np.array(system_a.long_memory, dtype=object))
    
    print(f"  Финальная ошибка A: {system_a.get_avg_error():.6f}")
    
    print("\n2. Тест B: с шумом + исправленная память")
    system_b = M1System()
    # включаем шумы
    system_b.environment.observation_noise['enabled'] = True
    system_b.environment.noise_params['enabled'] = True
    
    # Загружаем память
    data = np.load("fixed_memory_snapshot.npz", allow_pickle=True)
    loaded_region_errors = list(data['region_errors'])
    for i in range(len(system_b.region_errors)):
        system_b.region_errors[i] = list(loaded_region_errors[i])
    
    system_b.recent_errors.clear()
    system_b.recent_conflicts.clear()
    system_b.recent_activity.clear()
    for v in data['recent_errors']:
        system_b.recent_errors.append(float(v))
    for v in data['recent_conflicts']:
        system_b.recent_conflicts.append(float(v))
    for v in data['recent_activity']:
        system_b.recent_activity.append(int(v))
    
    system_b.memory_strength = float(data['memory_strength'])
    
    lm_obj = data['long_memory']
    if isinstance(lm_obj, np.ndarray):
        new_lm = []
        for region_entries in lm_obj:
            try:
                new_lm.append(list(region_entries))
            except Exception:
                new_lm.append([])
        system_b.long_memory = new_lm
    
    logger_b = SimulationLogger()
    for step in range(max_steps):
        alive = system_b.step()
        logger_b.log_step(step, system_b)
        if not alive:
            break
        if step % 50 == 0:
            print(f"  Шаг {step}: E={system_b.energy:.2f}, Ошибка={system_b.get_avg_error():.4f}")
    
    logger_b.save_to_csv("fixed_with_noise_results.csv")
    
    print(f"  Финальная ошибка B: {system_b.get_avg_error():.6f}")
    
    # Сравнение результатов
    print("\n=== СРАВНЕНИЕ РЕЗУЛЬТАТОВ ===")
    error_a = system_a.get_avg_error()
    error_b = system_b.get_avg_error()
    
    print(f"Ошибка A (без шума): {error_a:.6f}")
    print(f"Ошибка B (с шумом + память): {error_b:.6f}")
    print(f"Соотношение B/A: {error_b/error_a:.2f}x")
    
    # Загружаем старые результаты для сравнения
    try:
        import csv
        old_error_b = None
        with open('with_noise_loaded_memory_m1_v11_results.csv', 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            if rows:
                old_error_b = float(rows[-1]['avg_error'])
        
        if old_error_b:
            print(f"\n🎯 УЛУЧШЕНИЕ:")
            print(f"Старая ошибка B: {old_error_b:.6f}")
            print(f"Новая ошибка B: {error_b:.6f}")
            improvement = (old_error_b - error_b) / old_error_b * 100
            print(f"Улучшение: {improvement:.1f}%")
            
            if error_b < old_error_b:
                print("✅ ИСПРАВЛЕНИЯ РАБОТАЮТ!")
            else:
                print("❌ Нужны дополнительные исправления")
    except Exception as e:
        print(f"⚠ Не удалось сравнить со старыми результатами: {e}")

if __name__ == "__main__":
    run_fixed_test()