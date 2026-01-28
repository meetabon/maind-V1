"""
NeuralRegion - область нейронов 4x4x4 с локальным обучением
Каждый нейрон привязан к ячейке среды
"""

import numpy as np

class Neuron:
    def __init__(self, x, y, z, region_id):
        """Инициализация нейрона"""
        self.x, self.y, self.z = x, y, z
        self.region_id = region_id
        
        # Состояние
        self.activation = 0  # a_i ∈ {0,1}
        self.threshold = np.random.uniform(0.1, 0.9)  # θ_i
        
        # Веса (инициализируются в регионе)
        self.local_weights = {}  # w_ij для локальных связей
        self.env_weights = {}    # v_ik для связей со средой
        
    def calculate_input(self, neighbor_activations, env_signals):
        """Расчет входного сигнала I_i(t)"""
        # Вклад от соседних нейронов
        local_input = sum(
            self.local_weights.get(nid, 0) * activation 
            for nid, activation in neighbor_activations.items()
        )
        
        # Вклад от среды
        env_input = sum(
            self.env_weights.get(k, 0) * signal 
            for k, signal in enumerate(env_signals)
        )
        
        return local_input + env_input
    
    def update_activation(self, total_input):
        """Обновление активации a_i(t+1)

        Поддерживаем опциональный множитель порога через второй аргумент
        (для режима пониженной активности)."""
        # Старая сигнатура: update_activation(total_input)
        effective_threshold = self.threshold
        self.activation = 1 if total_input > effective_threshold else 0
        return self.activation

class NeuralRegion:
    def __init__(self, region_id, meta_params):
        """Инициализация области нейронов"""
        self.region_id = region_id
        self.meta_params = meta_params
        self.size = (4, 4, 4)  # подкуб 4x4x4
        
        # Создание нейронов
        self.neurons = {}
        for x in range(4):
            for y in range(4):
                for z in range(4):
                    nid = (x, y, z)
                    self.neurons[nid] = Neuron(x, y, z, region_id)
        
        # Инициализация весов
        self._initialize_weights()
        
        # Ожидания для предсказания
        self.expected_signal = np.zeros((4, 4, 4))
        self.signal_history = []
        
        # Дальние связи (межрегиональные)
        self.long_range_connections = {}
        
    def _initialize_weights(self):
        """Инициализация весов связей"""
        for nid, neuron in self.neurons.items():
            # Локальные связи с соседями в области
            neighbors = self._get_local_neighbors(nid)
            for neighbor_id in neighbors:
                weight = np.random.normal(0, 0.1)  # малые случайные веса
                neuron.local_weights[neighbor_id] = weight
            
            # Веса для сигналов среды (3x3x3 окрестность)
            for k in range(27):  # 3^3 = 27 соседей
                weight = np.random.normal(0, 0.1)
                neuron.env_weights[k] = weight
    
    def _get_local_neighbors(self, nid):
        """Получение локальных соседей нейрона в области"""
        x, y, z = nid
        neighbors = []
        
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    
                    nx, ny, nz = x + dx, y + dy, z + dz
                    if 0 <= nx < 4 and 0 <= ny < 4 and 0 <= nz < 4:
                        neighbors.append((nx, ny, nz))
        
        return neighbors
    
    def update(self, region_space):
        """Обновление области на один шаг"""
        # 1. Получение сигналов среды для каждого нейрона
        env_signals = self._get_environment_signals(region_space)
        
        # 2. Сбор активаций соседей
        neighbor_activations = {
            nid: neuron.activation for nid, neuron in self.neurons.items()
        }
        
        # 3. Обновление активаций всех нейронов
        new_activations = {}
        for nid, neuron in self.neurons.items():
            # Получаем соседей для этого нейрона
            local_neighbors = {
                neighbor_id: neighbor_activations[neighbor_id]
                for neighbor_id in self._get_local_neighbors(nid)
            }
            
            # Расчет входа и обновление активации
            total_input = neuron.calculate_input(local_neighbors, env_signals[nid])
            # Добавляем память (memory_bias) если область выставила
            memory_bias = getattr(self, 'memory_bias', 0.0)
            total_input += memory_bias

            # В режиме низкой активности увеличиваем порог требуемого входа
            threshold_multiplier = 1.5 if getattr(self, 'low_activity', False) else 1.0
            effective_threshold = neuron.threshold * threshold_multiplier

            # Временно применяем более высокий порог: напрямую сравним
            new_activations[nid] = 1 if total_input > effective_threshold else 0
        
        # 4. Обучение весов (пластичность)
        self._update_weights(new_activations)
        # Применяем новые активации к нейронам
        for nid, neuron in self.neurons.items():
            neuron.activation = new_activations[nid]
        
        # 5. Обновление ожиданий и расчет ошибки
        prediction_error = self._update_predictions(region_space)
        
        return prediction_error
    
    def _get_environment_signals(self, region_space):
        """Получение сигналов среды для каждого нейрона"""
        env_signals = {}
        
        for nid, neuron in self.neurons.items():
            x, y, z = nid
            
            # Сигнал своей ячейки + соседи 3x3x3
            signals = []
            
            # Собственная ячейка
            own_signal = region_space[x, y, z]
            
            # Соседи 3x3x3
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    for dz in [-1, 0, 1]:
                        nx = max(0, min(3, x + dx))
                        ny = max(0, min(3, y + dy))
                        nz = max(0, min(3, z + dz))
                        signals.append(region_space[nx, ny, nz])
            
            # Комбинированный сигнал: α * own + (1-α) * mean(neighbors)
            alpha = 0.7
            combined_signal = alpha * own_signal + (1 - alpha) * np.mean(signals)
            
            env_signals[nid] = signals  # полный набор для весов
        
        return env_signals
    
    def _update_weights(self, activations):
        """Обновление весов (локальное обучение)"""
        eta = self.meta_params['eta']
        gamma = 0.001  # регуляризация
        
        for nid, neuron in self.neurons.items():
            current_activation = activations[nid]
            
            # Обновление локальных весов (правило Хебба + регуляризация)
            for neighbor_id, weight in neuron.local_weights.items():
                neighbor_activation = activations[neighbor_id]
                
                # Δw_ij = η * a_i * a_j - γ * |w_ij|
                delta_w = (eta * current_activation * neighbor_activation - 
                          gamma * abs(weight))
                
                neuron.local_weights[neighbor_id] += delta_w
    
    def _update_predictions(self, region_space):
        """Обновление ожиданий и расчет ошибки предсказания"""
        # Текущий сигнал области (агрегированный)
        current_signal = np.mean(region_space)
        
        # Обновление ожидания (скользящее среднее)
        beta = 0.8
        if len(self.signal_history) > 0:
            expected = beta * self.signal_history[-1] + (1 - beta) * current_signal
        else:
            expected = current_signal
        
        # Ошибка предсказания
        if len(self.signal_history) > 0:
            prediction_error = abs(current_signal - self.signal_history[-1])
        else:
            prediction_error = 0.0
        
        # Обновление истории
        self.signal_history.append(current_signal)
        if len(self.signal_history) > 100:  # ограничиваем историю
            self.signal_history.pop(0)
        
        return prediction_error
    
    def get_long_links_count(self):
        """Количество дальних связей области"""
        return len(self.long_range_connections)

    def add_long_link(self, target_region_id, weight=None):
        """Добавить дальнюю связь к другому региону"""
        if weight is None:
            weight = np.random.normal(0, 0.1)
        self.long_range_connections[target_region_id] = weight

    def remove_long_link(self, target_region_id):
        """Удалить дальнюю связь"""
        if target_region_id in self.long_range_connections:
            del self.long_range_connections[target_region_id]
    
    def get_total_activity(self):
        """Общая активность области"""
        return sum(neuron.activation for neuron in self.neurons.values())
    
    def get_total_weights(self):
        """Сумма абсолютных значений весов"""
        total = 0
        for neuron in self.neurons.values():
            total += sum(abs(w) for w in neuron.local_weights.values())
            total += sum(abs(w) for w in neuron.env_weights.values())
        return total