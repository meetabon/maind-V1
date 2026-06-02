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
        """Возвращает SDR для следующего символа (упрощенно 512 бит) и удаляет его из буфера"""
        if not self.text_buffer:
            self._load_next_url()
            if not self.text_buffer:
                return np.zeros(512), None

        char = self.text_buffer[0]
        self.text_buffer = self.text_buffer[1:]

        # Кодируем символ в SDR (512 бит)
        sdr = np.zeros(512)
        np.random.seed(ord(char))
        active_indices = np.random.choice(512, 15, replace=False)
        sdr[active_indices] = 1
        return sdr, char
