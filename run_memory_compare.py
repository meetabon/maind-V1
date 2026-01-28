#!/usr/bin/env python3
"""
Прогон двух сессий: 1) без шумов, сохраняем память; 2) с шумом, загружаем память и прогоняем.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'm1_v11'))

from system import M1System
from logger import SimulationLogger


def run_system(system, max_steps=300, diagnostics=False, logger_prefix="run"):
    logger = SimulationLogger()
    for step in range(max_steps):
        alive = system.step()
        logger.log_step(step, system)
        if diagnostics and hasattr(system, 'last_diagnostics') and (step % 50 == 0 or not alive):
            diag = system.last_diagnostics
            print(f"[DIAG:{logger_prefix}] Step={step} E={system.energy:.2f} gain={diag['energy_gain']:.2f} cost={diag['energy_cost']:.2f}")
        if not alive:
            break
    # save
    fname = f"{logger_prefix}_m1_v11_results.csv"
    logger.save_to_csv(fname)
    logger.save_detailed_data()
    logger.print_summary()
    return system, logger


def save_memory_snapshot(system, path="memory_snapshot.npz"):
    # Сохраняем region_errors и краткую память
    region_errors = [np.array(errs) for errs in system.region_errors]
    recent_errors = np.array(system.recent_errors)
    recent_conflicts = np.array(system.recent_conflicts)
    recent_activity = np.array(system.recent_activity)
    # convert to object arrays so np.savez can store lists of arrays/dicts
    region_errors_obj = np.array(region_errors, dtype=object)
    long_memory_obj = np.array(system.long_memory, dtype=object)
    np.savez(path,
             memory_strength=system.memory_strength,
             region_errors=region_errors_obj,
             recent_errors=recent_errors,
             recent_conflicts=recent_conflicts,
             recent_activity=recent_activity,
             long_memory=long_memory_obj)
    print(f"Memory snapshot saved to {path}")


def load_memory_snapshot(system, path="memory_snapshot.npz"):
    data = np.load(path, allow_pickle=True)
    # restore
    loaded_region_errors = list(data['region_errors'])
    for i in range(len(system.region_errors)):
        system.region_errors[i] = list(loaded_region_errors[i])
    # recent deques
    system.recent_errors.clear()
    system.recent_conflicts.clear()
    system.recent_activity.clear()
    for v in data['recent_errors']:
        system.recent_errors.append(float(v))
    for v in data['recent_conflicts']:
        system.recent_conflicts.append(float(v))
    for v in data['recent_activity']:
        system.recent_activity.append(int(v))
    # memory_strength
    try:
        system.memory_strength = float(data['memory_strength'])
    except Exception:
        pass
    # long_memory (если есть)
    try:
        lm_obj = data['long_memory']
        # lm_obj may be a numpy array of objects; convert each element to a python list
        if isinstance(lm_obj, np.ndarray):
            new_lm = []
            for region_entries in lm_obj:
                try:
                    new_lm.append(list(region_entries))
                except Exception:
                    new_lm.append([])
            if len(new_lm) == len(system.long_memory):
                system.long_memory = new_lm
        else:
            lm = list(lm_obj)
            if len(lm) == len(system.long_memory):
                system.long_memory = lm
    except Exception:
        pass
    print(f"Memory snapshot loaded from {path}")


if __name__ == '__main__':
    max_steps = 300

    print("Run A: без шумов (перцептивный и наблюдательный отключены)")
    system_a = M1System()
    # отключаем оба типа шума
    system_a.environment.observation_noise['enabled'] = False
    system_a.environment.noise_params['enabled'] = False

    system_a, logger_a = run_system(system_a, max_steps=max_steps, diagnostics=True, logger_prefix="no_noise")
    save_memory_snapshot(system_a, path="memory_snapshot.npz")

    print("\nRun B: с шумом, загружаем память из Run A")
    system_b = M1System()
    # включаем шумы
    system_b.environment.observation_noise['enabled'] = True
    system_b.environment.noise_params['enabled'] = True
    # загружаем память
    load_memory_snapshot(system_b, path="memory_snapshot.npz")

    system_b, logger_b = run_system(system_b, max_steps=max_steps, diagnostics=True, logger_prefix="with_noise_loaded_memory")
    # Сохраняем снимок памяти после Run B
    save_memory_snapshot(system_b, path="memory_snapshot_B.npz")

    print("Два прогона завершены. Снимок Run B сохранён в memory_snapshot_B.npz")
