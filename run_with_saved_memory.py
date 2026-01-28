#!/usr/bin/env python3
"""
Загрузить память из memory_snapshot_B.npz и прогнать симуляцию один раз (с шумом).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'm1_v11'))

from system import M1System
from run_memory_compare import load_memory_snapshot, run_system

if __name__ == '__main__':
    max_steps = 300
    print("Запуск с загруженной памятью memory_snapshot_B.npz")
    system = M1System()
    system.environment.observation_noise['enabled'] = True
    system.environment.noise_params['enabled'] = True
    load_memory_snapshot(system, path='memory_snapshot_B.npz')
    system, logger = run_system(system, max_steps=max_steps, diagnostics=True, logger_prefix='replay_B_memory')
    print('Прогон завершён.')
