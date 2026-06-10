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
        """Загрузка состояния и создание экземпляра системы с защитой от повреждений"""
        if not os.path.exists(self.filepath) or os.path.getsize(self.filepath) == 0:
            return None

        try:
            with open(self.filepath, 'rb') as f:
                state = pickle.load(f)

            # Базовая проверка структуры
            if not isinstance(state, dict) or 'layers' not in state:
                raise ValueError("Некорректная структура файла состояния")

            system = system_class(meta_params)
            system.step_count = state.get('step_count', 0)
            system.energy = state.get('energy', 100.0)

            for i, layer_data in enumerate(state['layers']):
                if i >= len(system.sdr_layers): break
                layer = system.sdr_layers[i]
                layer.proximal_weights = layer_data.get('proximal_weights')
                layer.distal_weights = layer_data.get('distal_weights')
                layer.top_down_weights = layer_data.get('top_down_weights')
                layer.proximal_links = layer_data.get('proximal_links', {})
                layer.distal_links = layer_data.get('distal_links', {})
                layer.top_down_links = layer_data.get('top_down_links', {})

            print(f"Memory state successfully loaded from {self.filepath}")
            return system
        except (pickle.UnpicklingError, EOFError, ValueError, KeyError, Exception) as e:
            print(f"!!! ВНИМАНИЕ: Файл состояния {self.filepath} поврежден или несовместим ({e}).")
            print("Инициализация чистого ядра памяти...")
            return None
