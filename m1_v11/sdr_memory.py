import numpy as np

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

        # Проксимальные веса (восходящие)
        if input_dim:
            self.proximal_weights = np.random.uniform(0, 0.1, (size, input_dim))
        else:
            self.proximal_weights = None

        # Дистальные веса (латеральные внутри слоя для временной памяти)
        self.distal_weights = np.random.uniform(0, 0.05, (size, size))
        np.fill_diagonal(self.distal_weights, 0)

        # Обратные (дистальные сверху-вниз) веса для предсказания (contextual feedback)
        self.top_down_weights = None  # Инициализируется в системе при наличии верхнего слоя

        # Маска предсказания (predictive state)
        self.predictive_state = np.zeros(size)

        # Потенциалы нейронов (входные сигналы до применения kWTA)
        self.potentials = np.zeros(size)

    def get_total_weights(self):
        """Сумма абсолютных значений всех весов в слое"""
        total = 0
        if self.proximal_weights is not None:
            total += np.sum(np.abs(self.proximal_weights))
        total += np.sum(np.abs(self.distal_weights))
        return total

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
        Обучение весов по правилу Хебба.
        """
        # Обучаем проксимальные веса: усиливаем связи между активными входами и активными нейронами
        if self.proximal_weights is not None and proximal_input is not None:
            # Δw = η * (a_i * x_j - γ * w_ij)
            gamma = 0.001
            # Используем только положительные входы для обучения
            pos_input = np.maximum(0, proximal_input)
            outer_prod = np.outer(self.activations, pos_input)
            self.proximal_weights += eta * (outer_prod - gamma * self.proximal_weights)
            # Ограничиваем веса положительными значениями
            self.proximal_weights = np.maximum(0, self.proximal_weights)

        # Обучаем дистальные веса (латеральные)
        outer_distal = np.outer(self.activations, self.activations)
        np.fill_diagonal(outer_distal, 0)
        self.distal_weights += eta * (outer_distal - 0.001 * self.distal_weights)
        self.distal_weights = np.maximum(0, self.distal_weights)

    def learn_top_down(self, top_down_input, eta=0.01):
        """
        Обучение обратных связей.
        """
        if self.top_down_weights is not None and top_down_input is not None:
            # Усиливаем связи между активностью верхнего слоя и текущей активностью
            outer_prod = np.outer(self.activations, top_down_input)
            self.top_down_weights += eta * (outer_prod - 0.001 * self.top_down_weights)
            self.top_down_weights = np.maximum(0, self.top_down_weights)
