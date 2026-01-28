#!/usr/bin/env python3
"""
Тест гибридной памяти: позитив от теста A + негатив от теста B
Эксперимент: A -> B1 -> B2 (где B2 использует лучшие позитивы от A + негативы от B1)
"""

import os
import sys
import numpy as np
import copy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'm1_v11'))

from system import M1System
from logger import SimulationLogger

def save_memory_snapshot(system, path):
    """Сохранение снимка памяти"""
    region_errors = [np.array(errs) for errs in system.region_errors]
    recent_errors = np.array(system.recent_errors)
    recent_conflicts = np.array(system.recent_conflicts)
    recent_activity = np.array(system.recent_activity)
    
    np.savez(path,
             memory_strength=system.memory_strength,
             region_errors=np.array(region_errors, dtype=object),
             recent_errors=recent_errors,
             recent_conflicts=recent_conflicts,
             recent_activity=recent_activity,
             long_memory=np.array(system.long_memory, dtype=object))
    print(f"✓ Память сохранена: {path}")

def load_memory_snapshot(system, path):
    """Загрузка снимка памяти"""
    data = np.load(path, allow_pickle=True)
    
    # Загружаем region_errors
    loaded_region_errors = list(data['region_errors'])
    for i in range(len(system.region_errors)):
        system.region_errors[i] = list(loaded_region_errors[i])
    
    # Загружаем recent данные
    system.recent_errors.clear()
    system.recent_conflicts.clear()
    system.recent_activity.clear()
    for v in data['recent_errors']:
        system.recent_errors.append(float(v))
    for v in data['recent_conflicts']:
        system.recent_conflicts.append(float(v))
    for v in data['recent_activity']:
        system.recent_activity.append(int(v))
    
    # Загружаем memory_strength
    system.memory_strength = float(data['memory_strength'])
    
    # Загружаем long_memory
    lm_obj = data['long_memory']
    if isinstance(lm_obj, np.ndarray):
        new_lm = []
        for region_entries in lm_obj:
            try:
                new_lm.append(list(region_entries))
            except Exception:
                new_lm.append([])
        system.long_memory = new_lm
    
    print(f"✓ Память загружена: {path}")

def create_hybrid_memory(memory_a_path, memory_b_path, output_path):
    """Создание гибридной памяти: позитивы от A + негативы от B"""
    
    print(f"\n=== Создание гибридной памяти ===")
    
    # Загружаем обе памяти
    data_a = np.load(memory_a_path, allow_pickle=True)
    data_b = np.load(memory_b_path, allow_pickle=True)
    
    memory_a = data_a['long_memory']
    memory_b = data_b['long_memory']
    
    # Создаем гибридную память
    hybrid_memory = []
    
    for region_id in range(len(memory_a)):
        region_hybrid = []
        
        # Извлекаем позитивные примеры из памяти A
        positives_a = []
        if region_id < len(memory_a):
            for entry in memory_a[region_id]:
                if entry.get('type') == 'positive':
                    positives_a.append(entry)
        
        # Извлекаем негативные примеры из памяти B
        negatives_b = []
        if region_id < len(memory_b):
            for entry in memory_b[region_id]:
                if entry.get('type') != 'positive':
                    negatives_b.append(entry)
        
        # Собираем гибридную память для региона
        # Берем лучший позитивный от A
        if positives_a:
            best_positive = min(positives_a, key=lambda x: x.get('error', float('inf')))
            region_hybrid.append(best_positive)
            print(f"  Регион {region_id}: лучший позитив от A (ошибка={best_positive.get('error', 0):.4f})")
        
        # Добавляем негативы от B (ограничиваем до 19)
        negatives_b = sorted(negatives_b, key=lambda x: x.get('ts', 0), reverse=True)[:19]
        region_hybrid.extend(negatives_b)
        
        print(f"  Регион {region_id}: {len(negatives_b)} негативов от B")
        hybrid_memory.append(region_hybrid)
    
    # Сохраняем гибридную память (используем данные от B как основу)
    np.savez(output_path,
             memory_strength=data_b['memory_strength'],
             region_errors=data_b['region_errors'],
             recent_errors=data_b['recent_errors'],
             recent_conflicts=data_b['recent_conflicts'],
             recent_activity=data_b['recent_activity'],
             long_memory=np.array(hybrid_memory, dtype=object))
    
    print(f"✓ Гибридная память создана: {output_path}")

def run_system_test(system, max_steps, test_name, csv_name):
    """Запуск одного теста системы"""
    print(f"\n=== {test_name} ===")
    
    logger = SimulationLogger()
    
    for step in range(max_steps):
        alive = system.step()
        logger.log_step(step, system)
        
        if not alive:
            print(f"  Система умерла на шаге {step}")
            break
            
        if step % 50 == 0:
            print(f"  Шаг {step}: E={system.energy:.2f}, Ошибка={system.get_avg_error():.4f}")
    
    logger.save_to_csv(csv_name)
    final_error = system.get_avg_error()
    print(f"  Финальная ошибка: {final_error:.6f}")
    
    return system, final_error

def main():
    """Основной эксперимент"""
    print("=== ЭКСПЕРИМЕНТ: ГИБРИДНАЯ ПАМЯТЬ ===")
    print("A -> B1 -> B2 (позитив от A + негатив от B1)")
    
    max_steps = 300
    
    # ===== ТЕСТ A: Без шума (создание идеальной памяти) =====
    system_a = M1System()
    system_a.environment.observation_noise['enabled'] = False
    system_a.environment.noise_params['enabled'] = False
    
    system_a, error_a = run_system_test(system_a, max_steps, "ТЕСТ A: Без шума", "hybrid_test_A.csv")
    save_memory_snapshot(system_a, "hybrid_memory_A.npz")
    
    # ===== ТЕСТ B1: С шумом + память от A =====
    system_b1 = M1System()
    system_b1.environment.observation_noise['enabled'] = True
    system_b1.environment.noise_params['enabled'] = True
    
    load_memory_snapshot(system_b1, "hybrid_memory_A.npz")
    system_b1, error_b1 = run_system_test(system_b1, max_steps, "ТЕСТ B1: С шумом + память от A", "hybrid_test_B1.csv")
    save_memory_snapshot(system_b1, "hybrid_memory_B1.npz")
    
    # ===== СОЗДАНИЕ ГИБРИДНОЙ ПАМЯТИ =====
    create_hybrid_memory("hybrid_memory_A.npz", "hybrid_memory_B1.npz", "hybrid_memory_hybrid.npz")
    
    # ===== ТЕСТ B2: С шумом + гибридная память =====
    system_b2 = M1System()
    system_b2.environment.observation_noise['enabled'] = True
    system_b2.environment.noise_params['enabled'] = True
    
    load_memory_snapshot(system_b2, "hybrid_memory_hybrid.npz")
    system_b2, error_b2 = run_system_test(system_b2, max_steps, "ТЕСТ B2: С шумом + гибридная память", "hybrid_test_B2.csv")
    
    # ===== АНАЛИЗ РЕЗУЛЬТАТОВ =====
    print(f"\n" + "="*60)
    print("СРАВНЕНИЕ РЕЗУЛЬТАТОВ")
    print("="*60)
    print(f"Тест A  (без шума):           {error_a:.6f}")
    print(f"Тест B1 (шум + память A):     {error_b1:.6f}  ({error_b1/error_a:.1f}x от A)")
    print(f"Тест B2 (шум + гибрид A+B1):  {error_b2:.6f}  ({error_b2/error_a:.1f}x от A)")
    
    print(f"\n🎯 ЭФФЕКТ ГИБРИДНОЙ ПАМЯТИ:")
    if error_b2 < error_b1:
        improvement = (error_b1 - error_b2) / error_b1 * 100
        print(f"✅ УЛУЧШЕНИЕ: B2 лучше B1 на {improvement:.1f}%")
        print(f"   Гибридная память (позитив A + негатив B1) работает лучше!")
    else:
        degradation = (error_b2 - error_b1) / error_b1 * 100
        print(f"❌ УХУДШЕНИЕ: B2 хуже B1 на {degradation:.1f}%")
        print(f"   Негативы от B1 мешают больше, чем помогают")
    
    print(f"\n💡 ИНТЕРПРЕТАЦИЯ:")
    print(f"   - Позитивы от A: идеальные примеры без шума")
    print(f"   - Негативы от B1: реальные ошибки в условиях шума")
    print(f"   - Если B2 > B1: система лучше избегает реальных ошибок")
    print(f"   - Если B2 < B1: негативы от B1 создают ложные ограничения")

if __name__ == "__main__":
    main()