import numpy as np
from dataclasses import dataclass

@dataclass
class MemoryLink:
    """Низкоуровневая структура связи"""
    target_neuron_id: int  # адрес связанного нейрона
    link_strength: int = 1  # РЕЗЕРВ: сила связи (пока 1)
    time_marker: int = 0    # РЕЗЕРВ: метка времени (пока 0)
    table_marker: int = 32  # маркер разрядности таблицы (32, 64, 128)
    reserved_meta: int = 0  # РЕЗЕРВ: тип сигнала

class SDRLayer:
    def __init__(self, name, size, min_active, max_active, input_dim=None):
        """
        Инициализация слоя SDR.
        :param name: Название слоя.
        :param size: Количество нейронов в слое.
        :param min_active: Минимальное количество активных нейронов.
        :param max_active: Максимальное количество активных нейронов.
        :param input_dim: Размерность входного вектора (для проксимальных связей).
        """
        self.name = name
        self.size = size
        self.min_active = min_active
        self.max_active = max_active

        # Состояние активации (SDR)
        self.activations = np.zeros(size)

        # Вместо матриц весов используем списки MemoryLink для каждого нейрона
        self.proximal_links = {i: [] for i in range(size)}
        self.distal_links = {i: [] for i in range(size)}
        self.top_down_links = {i: [] for i in range(size)}

        # Для обратной совместимости и простоты математики в этой версии
        # будем пока использовать разреженные матрицы.
        # ВАЖНО: Инициализируем небольшими случайными значениями для "старта" обучения.
        self.input_dim = input_dim
        if input_dim:
            self.proximal_weights = np.random.uniform(0, 0.01, (size, input_dim))
        else:
            self.proximal_weights = None

        self.distal_weights = np.random.uniform(0, 0.01, (size, size))
        self.top_down_weights = None # Будет инициализирован как матрица

        # Маска предсказания (predictive state)
        self.predictive_state = np.zeros(size)

        # Потенциалы нейронов (входные сигналы до применения kWTA)
        self.potentials = np.zeros(size)

    def get_total_weights(self):
        """Сумма всех связей в слое"""
        total = 0
        for links in self.proximal_links.values():
            total += len(links)
        for links in self.distal_links.values():
            total += len(links)
        for links in self.top_down_links.values():
            total += len(links)
        return total if total > 0 else 1 # Чтобы не было 0

    def get_activity_count(self):
        """Количество активных нейронов"""
        return np.sum(self.activations)

    def _apply_kwta(self):
        """
        Механизм K-Winners-Take-All для поддержания разреженности.
        Выбирает топ-K нейронов с самым сильным входным сигналом.
        """
        # Определяем k в диапазоне [min_active, max_active]
        k = int((self.min_active + self.max_active) / 2)

        # Сбрасываем активации
        self.activations = np.zeros(self.size)

        # Находим индексы топ-K нейронов
        if np.any(self.potentials > 0.001):  # Добавим небольшой порог
            # Сортируем потенциалы
            indices = np.argsort(self.potentials)
            # Берем топ-K
            top_k_indices = indices[-k:]
            # Оставляем только те, что выше порога
            actual_indices = top_k_indices[self.potentials[top_k_indices] > 0.001]
            self.activations[actual_indices] = 1

    def update(self, proximal_input=None):
        """
        Обновление состояния слоя (Прямая фаза).
        :param proximal_input: Входной вектор от нижнего слоя или среды.
        """
        # 1. Считаем проксимальный вход (bottom-up)
        if self.proximal_weights is not None and proximal_input is not None:
            self.potentials = np.dot(self.proximal_weights, proximal_input)
        else:
            self.potentials = np.zeros(self.size)

        # 2. Добавляем дистальный вход (внутренняя временная память)
        distal_input = np.dot(self.distal_weights, self.activations)
        self.potentials += distal_input

        # 3. Эффект Т9: Если нейрон был в предсказанном состоянии, снижаем его порог (усиливаем потенциал)
        # Это позволяет предсказанным паттернам побеждать при kWTA даже при слабом входе
        self.potentials += self.predictive_state * 0.5

        # 4. Применяем kWTA
        self._apply_kwta()

        return self.activations

    def update_predictive_state(self, top_down_input=None):
        """
        Обратная фаза: расчет предсказаний на основе сигнала сверху.
        """
        if self.top_down_weights is not None and top_down_input is not None:
            raw_prediction = np.dot(self.top_down_weights, top_down_input)
            # Бинаризуем предсказание (нейрон либо ожидает вход, либо нет)
            self.predictive_state = (raw_prediction > 0.1).astype(float)
        else:
            self.predictive_state = np.zeros(self.size)
        return self.predictive_state

    def learn(self, proximal_input, eta=0.01):
        """
        Обучение связей. Теперь создаем MemoryLink.
        """
        if self.proximal_weights is not None and proximal_input is not None:
            active_inputs = np.where(proximal_input > 0)[0]
            active_neurons = np.where(self.activations > 0)[0]

            for i in active_neurons:
                for j in active_inputs:
                    # Создаем новую связь или обновляем существующую
                    if self.proximal_weights[i, j] == 0:
                        self.proximal_weights[i, j] = 1
                        self.proximal_links[i].append(MemoryLink(
                            target_neuron_id=j,
                            table_marker=len(proximal_input)
                        ))

        # Дистальные (латеральные)
        active_neurons = np.where(self.activations > 0)[0]
        for i in active_neurons:
            for j in active_neurons:
                if i != j and self.distal_weights[i, j] == 0:
                    self.distal_weights[i, j] = 1
                    self.distal_links[i].append(MemoryLink(
                        target_neuron_id=j,
                        table_marker=self.size
                    ))

    def learn_top_down(self, top_down_input, eta=0.01):
        """
        Обучение обратных связей через MemoryLink.
        """
        if self.top_down_weights is not None and top_down_input is not None:
            active_top = np.where(top_down_input > 0)[0]
            active_neurons = np.where(self.activations > 0)[0]

            for i in active_neurons:
                for j in active_top:
                    if self.top_down_weights[i, j] == 0:
                        self.top_down_weights[i, j] = 1
                        self.top_down_links[i].append(MemoryLink(
                            target_neuron_id=j,
                            table_marker=len(top_down_input)
                        ))
