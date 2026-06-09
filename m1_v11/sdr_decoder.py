import numpy as np

class SDRDecoder:
    def __init__(self):
        # Хранилище: Ключ - tuple отсортированных индексов, Значение - строка
        self.vocabulary = {
            "minus_1": {}, # Буквы
            "0": {}        # Слова
        }

    def learn_mapping(self, layer_id, activations, label):
        """Автоматическая калибровка: запись паттерна в словарь"""
        indices = tuple(sorted(np.where(activations > 0)[0]))
        if not indices or not label:
            return

        key = "minus_1" if layer_id == -1 else str(layer_id)
        if key in self.vocabulary:
            # Если для этого паттерна еще нет метки, записываем
            # (Или обновляем, если Учитель подает заведомо верные данные)
            if indices not in self.vocabulary[key]:
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
