import numpy as np

class SDRDecoder:
    def __init__(self):
        # Базовый ассоциативный словарь: Индексы <-> Символы
        self.letter_map = {} # {index_set: char}
        self.word_map = {}   # {index_set: word}

    def learn_mapping(self, layer_id, activations, label):
        """Обучение маппинга для конкретной активации"""
        indices = tuple(np.where(activations > 0)[0])
        if not indices:
            return

        if layer_id == -1: # Буквы
            self.letter_map[indices] = label
        elif layer_id == 0: # Слова
            self.word_map[indices] = label

    def decode_layer(self, layer_id, activations):
        """Расшифровка активации слоя"""
        indices = tuple(np.where(activations > 0)[0])
        if not indices:
            return "[Empty]"

        mapping = self.letter_map if layer_id == -1 else self.word_map

        # Прямое совпадение
        if indices in mapping:
            return mapping[indices]

        # Поиск наиболее похожего (overlap)
        best_match = None
        max_overlap = 0

        for stored_indices, label in mapping.items():
            overlap = len(set(indices) & set(stored_indices))
            if overlap > max_overlap:
                max_overlap = overlap
                best_match = label

        if best_match and max_overlap > 1: # Минимальный порог совпадения
            return f"{best_match}?" # Знак вопроса означает частичное совпадение

        return f"Indices:{list(indices)}"
