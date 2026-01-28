#!/usr/bin/env python3
import sys, os
import numpy as np

# ensure m1_v11 is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'm1_v11'))
from system import M1System

sys = M1System()
env = sys.environment
space = env.space

print('\n=== ENVIRONMENT ===')
vals, counts = np.unique(space, return_counts=True)
for v, c in zip(vals, counts):
    print(f'value={v}: count={c}')
print(f'total cells: {space.size}')

print('\n=== SYSTEM OVERVIEW ===')
print(f'regions: {len(sys.regions)}')
print(f'initial energy: {sys.energy}')
print(f'initial activations total: {sum(r.get_total_activity() for r in sys.regions)}')

region_signals = env.get_region_signals()

for i, region in enumerate(sys.regions):
    thresholds = [n.threshold for n in region.neurons.values()]
    local_weights = [abs(w) for n in region.neurons.values() for w in n.local_weights.values()]
    env_weights = [abs(w) for n in region.neurons.values() for w in n.env_weights.values()]

    avg_thresh = np.mean(thresholds)
    median_thresh = np.median(thresholds)
    avg_loc_w = np.mean(local_weights) if local_weights else 0.0
    avg_env_w = np.mean(env_weights) if env_weights else 0.0

    # compute env input per neuron with neighbor activations = 0
    region_space = region_signals[i]
    # use region._get_environment_signals to get per-neuron env signal lists
    env_signal_map = region._get_environment_signals(region_space)
    total_inputs = []
    for nid, neuron in region.neurons.items():
        signals = env_signal_map[nid]
        env_input = sum(neuron.env_weights[k] * signals[k] for k in range(len(neuron.env_weights)))
        total_inputs.append(env_input)

    avg_input = np.mean(total_inputs)
    max_input = np.max(total_inputs)
    min_input = np.min(total_inputs)

    print(f"\nRegion {i}:")
    print(f"  avg_threshold={avg_thresh:.3f}, median_threshold={median_thresh:.3f}")
    print(f"  avg_local_weight_abs={avg_loc_w:.4f}, avg_env_weight_abs={avg_env_w:.4f}")
    print(f"  avg_env_input={avg_input:.3f}, min={min_input:.3f}, max={max_input:.3f}")
    print(f"  total_neurons={len(region.neurons)}, long_links={region.get_long_links_count()}")

print('\n=== MEMORY & META ===')
print(f'memory_length={sys.memory_length}, recent_errors_len={len(sys.recent_errors)}')
print(f'k_base={sys.k_base}, k_conflict={sys.k_conflict}, conflict_threshold={sys.conflict_threshold}')

# Проверим, как память влияет на входы и активации
print('\n=== MEMORY EFFECT TEST ===')
for i, region in enumerate(sys.regions):
    errs = sys.region_errors[i][-sys.memory_length:]
    mean_err = float(np.mean(errs)) if len(errs) > 0 else 1.0
    memory_confidence = max(0.0, 1.0 - mean_err / 0.5)
    memory_bias = memory_confidence * sys.memory_strength

    region_space = env.get_region_signals()[i]
    env_signal_map = region._get_environment_signals(region_space)
    activations_if_memory = 0
    total_inputs = []
    thresholds = []
    for nid, neuron in region.neurons.items():
        signals = env_signal_map[nid]
        env_input = sum(neuron.env_weights[k] * signals[k] for k in range(len(neuron.env_weights)))
        total_input = env_input + memory_bias
        total_inputs.append(total_input)
        thresholds.append(neuron.threshold)
        if total_input > neuron.threshold:
            activations_if_memory += 1

    print(f"Region {i}: memory_bias={memory_bias:.3f}, activations_if_memory={activations_if_memory}/{len(region.neurons)}; avg_input_with_memory={np.mean(total_inputs):.3f}, avg_threshold={np.mean(thresholds):.3f}")

print('\nDone diagnostics')
