import numpy as np

class SDRDecoder:
    def __init__(self):
        # Хранилище: Ключ - tuple отсортированных индексов, Значение - строка
        self.vocabulary = {
            "minus_1": {}, # Буквы
            "0": {}        # Слова
        }

    def learn_mapping(self, layer_id, activations, label):
        """Обучение маппинга для конкретной активации"""
        indices = tuple(sorted(np.where(activations > 0)[0]))
        if not indices:
            return

        key = "minus_1" if layer_id == -1 else str(layer_id)
        if key in self.vocabulary:
            self.vocabulary[key][indices] = label

    def decode_layer(self, layer_id: str, activations: np.ndarray) -> str:
        """Расшифровка активации слоя"""
        indices = tuple(sorted(np.where(activations > 0)[0]))
        if not indices:
            return "[Empty]"

        if layer_id in self.vocabulary:
            mapping = self.vocabulary[layer_id]
            if indices in mapping:
                return mapping[indices]

        # Если совпадений нет, возвращаем сырые индексы
        return f"[SDR: {', '.join(map(str, indices))}]"
