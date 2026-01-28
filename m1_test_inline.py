#!/usr/bin/env python3
"""
Встроенный тест M1 v1.1 - все модули в одном файле для диагностики
"""

import numpy as np
import csv

print("=== M1 v1.1 Inline Test ===")

# Проверяем numpy
try:
    arr = np.array([1, 2, 3])
    print(f"✓ NumPy работает: {arr}")
except Exception as e:
    print(f"✗ Ошибка NumPy: {e}")
    exit(1)

# Простая версия Environment
class SimpleEnvironment:
    def __init__(self):
        self.space = np.random.choice([-1, 0, 1], size=(8, 8, 8))
        print(f"✓ Среда создана: {self.space.shape}")
    
    def get_region_signals(self):
        signals = []
        for i in range(8):
            x_start = (i % 2) * 4
            y_start = ((i // 2) % 2) * 4  
            z_start = (i // 4) * 4
            region = self.space[x_start:x_start+4, y_start:y_start+4, z_start:z_start+4]
            signals.append(region)
        return signals

# Простая версия системы
class SimpleSystem:
    def __init__(self):
        self.energy = 128.0  # 8 * 32 * 0.5
        self.env = SimpleEnvironment()
        self.step_count = 0
        print(f"✓ Система создана. Энергия: {self.energy}")
    
    def step(self):
        self.step_count += 1
        
        # Простая логика: теряем энергию каждый шаг
        self.energy -= 0.5
        
        if self.energy <= 0:
            return False
        return True
    
    def get_avg_error(self):
        return 0.1 * np.random.random()

# Тест
try:
    system = SimpleSystem()
    
    for step in range(20):
        alive = system.step()
        error = system.get_avg_error()
        
        print(f"Шаг {step}: E={system.energy:.1f}, Ошибка={error:.3f}")
        
        if not alive:
            print(f"Система умерла на шаге {step}")
            break
    
    print("✓ Тест завершен успешно!")
    
except Exception as e:
    print(f"✗ Ошибка в тесте: {e}")
    import traceback
    traceback.print_exc()