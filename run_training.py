import sys
import os
import numpy as np

# Добавляем путь к m1_v11 для импортов
sys.path.append(os.path.join(os.getcwd(), 'm1_v11'))

from system import M1System
from Learning.streamer import InternetStreamer
from logger import SimulationLogger

def main():
    print("=== Запуск внешнего обучения M1 (Internet Streaming) ===")

    # Инициализация систем
    system = M1System()
    streamer = InternetStreamer()
    logger = SimulationLogger()

    max_steps = 1000

    for step in range(max_steps):
        # 1. Получаем SDR токен от Учителя
        env_input, char = streamer.get_next_char_sdr()

        if char is None:
            print("Поток данных завершен.")
            break

        # 2. Передаем сигнал в изолированную память
        alive = system.step(external_input=env_input)

        # 3. Логирование
        logger.log_step(step, system)

        if not alive:
            print(f"Шаг {step}: Система умерла (энергия = {system.energy:.2f})")
            break

        if step % 100 == 0:
            print(f"Шаг {step}: Обработан символ '{char}', E={system.energy:.2f}")

    logger.save_to_csv("m1_v11_results.csv")
    print("Обучение завершено. Результаты сохранены в m1_v11_results.csv")

if __name__ == "__main__":
    main()
