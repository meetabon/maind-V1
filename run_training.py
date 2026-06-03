import sys
import os
import numpy as np

# Добавляем путь к m1_v11 для импортов
sys.path.append(os.path.join(os.getcwd(), 'm1_v11'))

from system import M1System
from Learning.streamer import InternetStreamer
from Learning.scheduler import TaskScheduler
from logger import SimulationLogger
from sdr_decoder import SDRDecoder

def main():
    print("=== Запуск внешнего обучения M1 (Internet Streaming) ===")

    # Инициализация систем
    system = M1System()
    streamer = InternetStreamer()
    scheduler = TaskScheduler()
    logger = SimulationLogger()
    decoder = SDRDecoder()

    # Автоматическая загрузка чекпоинта
    if system.persistence.load_state(system):
        print("Возобновление обучения из сохраненного состояния.")

    max_steps = 1000

    for step in range(max_steps):
        # 0. Проверка планировщика
        new_task = scheduler.check_for_new_tasks()
        if new_task:
            print(f"ПЛАНИРОВЩИК: Новое задание - {new_task['instruction']}")

        # 1. Получаем SDR токен от Учителя
        env_input, char = streamer.get_next_char_sdr()

        if char is None:
            print("Поток данных завершен.")
            break

        # 2. Передаем сигнал в изолированную память
        alive = system.step(external_input=env_input)

        # 2.5 Обучение декодера и логирование активности
        if char:
            decoder.learn_mapping(-1, system.layer_minus_1.activations, char)
            # Упрощенно: "слово" формируется из накопленных символов (в HTM это сложнее)
            # Для отчета будем декодировать текущий символ
            decoded_char = decoder.decode_layer(-1, system.layer_minus_1.activations)
            print(f"Шаг {step} | Вход: '{char}' | SDR Декодер: '{decoded_char}' | Anomaly: {system.need_teacher}")

        # Обработка аномалии (флаг NEED_TEACHER)
        if system.need_teacher:
            print(f"!!! АНОМАЛИЯ на шаге {step}: Запрос к Учителю для дообучения.")

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
