"""
Environment - 3D среда с динамическими изменениями
Реализует дрейф и катастрофы
"""

import numpy as np
import requests
from bs4 import BeautifulSoup
import re

class InternetStreamer:
    def __init__(self):
        self.urls = [
            "https://ru.wikipedia.org/wiki/Нейрон",
            "https://ru.wikipedia.org/wiki/Искусственный_интеллект",
            "https://habr.com/ru/articles/731174/"
        ]
        self.text_buffer = ""
        self.char_index = 0
        self._load_next_url()

    def _load_next_url(self):
        if not self.urls:
            return
        url = self.urls.pop(0)
        try:
            response = requests.get(url, timeout=5)
            soup = BeautifulSoup(response.text, 'html.parser')
            # Убираем скрипты и стили
            for script in soup(["script", "style"]):
                script.decompose()
            text = soup.get_text()
            # Очистка текста
            text = re.sub(r'\s+', ' ', text)
            self.text_buffer += text
        except Exception as e:
            print(f"Error loading {url}: {e}")

    def get_next_char_sdr(self):
        """Возвращает SDR для следующего символа (упрощенно 512 бит)"""
        if not self.text_buffer:
            self._load_next_url()
            if not self.text_buffer:
                return np.zeros(512)

        # Берем первый символ и сразу удаляем его из буфера (Streaming)
        char = self.text_buffer[0]
        self.text_buffer = self.text_buffer[1:]

        # Кодируем символ в SDR (512 бит)
        # Для простоты используем хеш символа для выбора активных битов
        sdr = np.zeros(512)
        np.random.seed(ord(char))
        active_indices = np.random.choice(512, 15, replace=False)
        sdr[active_indices] = 1
        return sdr

class Environment:
    def __init__(self, size=(8, 8, 8)):
        """Инициализация 3D среды"""
        self.size = size
        self.space = np.random.choice([-1, 0, 1], size=size)
        self.step_count = 0
        self.streamer = InternetStreamer()

        # Параметры динамики
        self.drift_interval = 100  # медленный дрейф каждые 100 шагов
        self.drift_rate = 0.1      # 10% ячеек меняется
        self.catastrophe_interval = 500  # катастрофы раз в 500 шагов
        self.catastrophe_rate = 0.2      # 20% ячеек меняется

    def update(self, step):
        """Обновление среды на каждом шаге"""
        self.step_count = step

        # Медленный дрейф
        if step % self.drift_interval == 0 and step > 0:
            self._apply_drift()

        # Катастрофы
        if step % self.catastrophe_interval == 0 and step > 0:
            self._apply_catastrophe()

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
            ]

            region_signals.append(region_space)

        return region_signals

    def get_cell_value(self, x, y, z):
        """Получение значения конкретной ячейки"""
        return self.space[x % self.size[0], y % self.size[1], z % self.size[2]]

    def set_cell_value(self, x, y, z, value):
        """Установка значения ячейки (для поглощения энергии)"""
        self.space[x % self.size[0], y % self.size[1], z % self.size[2]] = value