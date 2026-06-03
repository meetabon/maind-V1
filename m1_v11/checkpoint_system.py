import pickle
import os
import numpy as np

class MemoryPersistence:
    def __init__(self, checkpoint_path='m1_v11/memory_checkpoint.bin'):
        self.checkpoint_path = checkpoint_path

    def save_state(self, system):
        """Сохранение состояния всех слоев системы"""
        state = {
            'step_count': system.step_count,
            'energy': system.energy,
            'layers': []
        }

        for layer in system.sdr_layers:
            layer_data = {
                'name': layer.name,
                'proximal_weights': layer.proximal_weights,
                'distal_weights': layer.distal_weights,
                'top_down_weights': layer.top_down_weights,
                'proximal_links': layer.proximal_links,
                'distal_links': layer.distal_links,
                'top_down_links': layer.top_down_links
            }
            state['layers'].append(layer_data)

        with open(self.checkpoint_path, 'wb') as f:
            pickle.dump(state, f)
        # print(f"Checkpoint saved to {self.checkpoint_path}")

    def load_state(self, system):
        """Загрузка состояния системы"""
        if not os.path.exists(self.checkpoint_path):
            return False

        try:
            with open(self.checkpoint_path, 'rb') as f:
                state = pickle.load(f)

            system.step_count = state['step_count']
            system.energy = state['energy']

            for i, layer_data in enumerate(state['layers']):
                layer = system.sdr_layers[i]
                layer.proximal_weights = layer_data['proximal_weights']
                layer.distal_weights = layer_data['distal_weights']
                layer.top_down_weights = layer_data['top_down_weights']
                layer.proximal_links = layer_data['proximal_links']
                layer.distal_links = layer_data['distal_links']
                layer.top_down_links = layer_data['top_down_links']

            print(f"Memory state loaded from {self.checkpoint_path}")
            return True
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return False
