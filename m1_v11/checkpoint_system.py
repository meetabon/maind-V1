import pickle
import os
import numpy as np

class MemoryPersistence:
    def __init__(self, filepath="memory_state.pkl"):
        self.filepath = filepath

    def save(self, system_instance):
        """Сохранение весов и MemoryLink всех слоев системы"""
        state = {
            'step_count': system_instance.step_count,
            'energy': system_instance.energy,
            'layers': []
        }

        for layer in system_instance.sdr_layers:
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

        with open(self.filepath, 'wb') as f:
            pickle.dump(state, f)

    def load(self, system_class, meta_params=None):
        """Загрузка состояния и создание экземпляра системы"""
        if not os.path.exists(self.filepath):
            return None

        try:
            with open(self.filepath, 'rb') as f:
                state = pickle.load(f)

            system = system_class(meta_params)
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

            print(f"Memory state loaded from {self.filepath}")
            return system
        except Exception as e:
            print(f"Error loading memory state: {e}")
            return None
