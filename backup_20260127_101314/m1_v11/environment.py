"""
Environment - 3D среда с динамическими изменениями
Реализует дрейф и катастрофы
"""

import numpy as np

class Environment:
    def __init__(self, size=(8, 8, 8)):
        """Инициализация 3D среды"""
        self.size = size
        self.space = np.random.choice([-1, 0, 1], size=size)
        self.step_count = 0
        
        # Параметры динамики
        self.drift_interval = 100  # медленный дрейф каждые 100 шагов
        self.drift_rate = 0.1      # 10% ячеек меняется
        self.catastrophe_interval = 500  # катастрофы раз в 500 шагов
        self.catastrophe_rate = 0.2      # 20% ячеек меняется
        
        # === ЛОКАЛЬНАЯ ЛОЖЬ (Perceptual Bias) ===
        # Каждый регион имеет свой постоянный шум восприятия
        self.noise_params = {
            'enabled': True,
            'noise_magnitude': 0.15,      # амплитуда шума (% от значения)
            'noise_update_interval': 150,  # менять шум каждые 150 шагов
            'region_bias': {}  # постоянный шум для каждого региона
        }
        
        # Инициализируем шумы для 8 регионов
        self._initialize_perception_noise()

        # === НАБЛЮДАТЕЛЬНЫЙ ШУМ (Time-varying, SNR) ===
        # Noise added to observed region signals (simulates sensor noise)
        self.observation_noise = {
            'enabled': True,
            'magnitude': 0.25,            # RMS magnitude of observation noise
            'update_interval': 20,        # менять профиль шума каждые N шагов
            'region_noise': {}            # per-region static bias noise (4x4x4)
        }
        self._initialize_observation_noise()
    
    def _initialize_perception_noise(self):
        """Инициализация постоянного шума восприятия для каждого региона"""
        for region_id in range(8):
            # Генерируем шум для каждого нейрона в регионе (4×4×4 = 64)
            region_noise = np.random.uniform(
                -self.noise_params['noise_magnitude'],
                self.noise_params['noise_magnitude'],
                size=(4, 4, 4)
            )
            self.noise_params['region_bias'][region_id] = region_noise

    def _initialize_observation_noise(self):
        """Инициализация пер-области наблюдательного шума (статический профиль)"""
        for region_id in range(8):
            profile = np.random.normal(
                loc=0.0,
                scale=self.observation_noise['magnitude'] / 2.0,
                size=(4, 4, 4)
            )
            self.observation_noise['region_noise'][region_id] = profile

    def _update_observation_noise(self):
        """Плавное обновление наблюдательного шума (меняется профиль)"""
        for region_id in range(8):
            # небольшая дрейфующая компонента к профилю
            drift = np.random.normal(
                loc=0.0,
                scale=self.observation_noise['magnitude'] / 10.0,
                size=(4, 4, 4)
            )
            self.observation_noise['region_noise'][region_id] += drift
        
    def update(self, step):
        """Обновление среды на каждом шаге"""
        self.step_count = step
        
        # Медленный дрейф
        if step % self.drift_interval == 0 and step > 0:
            self._apply_drift()
        
        # Катастрофы
        if step % self.catastrophe_interval == 0 and step > 0:
            self._apply_catastrophe()
        
        # Обновление перцептивного шума
        if step % self.noise_params['noise_update_interval'] == 0 and step > 0:
            self._update_perception_noise()

        # Обновление наблюдательного шума
        if self.observation_noise['enabled'] and step % self.observation_noise['update_interval'] == 0 and step > 0:
            self._update_observation_noise()
    
    def _apply_drift(self):
        """Медленный дрейф - односторонние изменения"""
        total_cells = np.prod(self.size)
        n_changes = int(total_cells * self.drift_rate)
        
        # Случайные позиции для изменения
        positions = np.random.choice(total_cells, n_changes, replace=False)
        
        for pos in positions:
            x, y, z = np.unravel_index(pos, self.size)
            # Односторонние изменения (не симметричная инверсия)
            self.space[x, y, z] = np.random.choice([-1, 0, 1])
    
    def _apply_catastrophe(self):
        """Локальные катастрофы"""
        total_cells = np.prod(self.size)
        n_changes = int(total_cells * self.catastrophe_rate)
        
        # Выбираем центр катастрофы
        center = (
            np.random.randint(0, self.size[0]),
            np.random.randint(0, self.size[1]),
            np.random.randint(0, self.size[2])
        )
        
        # Локальные изменения вокруг центра
        radius = 2
        changed = 0
        
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                for dz in range(-radius, radius + 1):
                    if changed >= n_changes:
                        break
                    
                    x = (center[0] + dx) % self.size[0]
                    y = (center[1] + dy) % self.size[1]
                    z = (center[2] + dz) % self.size[2]
                    
                    if np.random.random() < 0.5:  # 50% шанс изменения
                        if np.random.random() < 0.5:
                            self.space[x, y, z] = 0  # обнуление
                        else:
                            self.space[x, y, z] = -1  # угроза
                        changed += 1
    
    def get_neighborhood(self, x, y, z, radius=1):
        """Получение сигналов соседей для позиции (x,y,z)"""
        signals = []
        
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                for dz in range(-radius, radius + 1):
                    nx = (x + dx) % self.size[0]
                    ny = (y + dy) % self.size[1]
                    nz = (z + dz) % self.size[2]
                    signals.append(self.space[nx, ny, nz])
        
        return np.array(signals)
    
    def get_region_signals(self):
        """Получение сигналов для всех 8 областей (подкубы 4x4x4)"""
        region_signals = []
        
        for region_id in range(8):
            # Определяем границы подкуба для области
            x_start = (region_id % 2) * 4
            y_start = ((region_id // 2) % 2) * 4
            z_start = (region_id // 4) * 4
            
            # Извлекаем подкуб 4x4x4
            region_space = self.space[
                x_start:x_start+4,
                y_start:y_start+4,
                z_start:z_start+4
            ].astype(float)
            
            # === ПРИМЕНЯЕМ ЛОКАЛЬНУЮ ЛОЖЬ ===
            # Добавляем постоянный региональный шум восприятия
            if self.noise_params['enabled']:
                region_noise = self.noise_params['region_bias'][region_id]
                region_space = region_space + region_noise
                # Обрезаем значения в диапазон [-1, 1]
                region_space = np.clip(region_space, -1, 1)
            # === ПРИМЕНЯЕМ НАБЛЮДАТЕЛЬНЫЙ ШУМ ===
            if self.observation_noise['enabled']:
                obs_profile = self.observation_noise['region_noise'][region_id]
                # Добавляем профиль шума и небольшую случайную флуктуацию
                fluct = np.random.normal(0.0, self.observation_noise['magnitude'] / 5.0, size=(4,4,4))
                region_space = region_space + obs_profile + fluct
                region_space = np.clip(region_space, -1, 1)

            region_signals.append(region_space)
        
        return region_signals
    
    def get_cell_value(self, x, y, z):
        """Получение значения конкретной ячейки"""
        return self.space[x % self.size[0], y % self.size[1], z % self.size[2]]
    
    def set_cell_value(self, x, y, z, value):
        """Установка значения ячейки (для поглощения энергии)"""
        self.space[x % self.size[0], y % self.size[1], z % self.size[2]] = value
    
    def _update_perception_noise(self):
        """Обновление перцептивного шума каждые N шагов
        
        Это зачаток "внутреннего конфликта":
        каждый регион переосмысливает свой опыт,
        может привести к расхождениям с другими.
        """
        for region_id in range(8):
            # Генерируем НОВЫЙ постоянный шум
            # (каждый регион "переобучается" новому искажению)
            new_noise = np.random.uniform(
                -self.noise_params['noise_magnitude'],
                self.noise_params['noise_magnitude'],
                size=(4, 4, 4)
            )
            self.noise_params['region_bias'][region_id] = new_noise
    
    def get_perception_bias_for_region(self, region_id):
        """Получить текущий перцептивный шум для региона (для логирования)"""
        if not self.noise_params['enabled']:
            return np.zeros((4, 4, 4))
        return self.noise_params['region_bias'][region_id]