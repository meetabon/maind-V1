import sys
import os
import numpy as np
import time

# Добавляем путь к m1_v11 для импортов
sys.path.append(os.path.join(os.getcwd(), 'm1_v11'))

from system import M1System
from Learning.streamer import InternetStreamer
from Learning.scheduler import TaskScheduler
from logger import SimulationLogger
from sdr_decoder import SDRDecoder
from checkpoint_system import MemoryPersistence
from time_hardware import HardwareTimeCore

def main():
    print("=== Запуск внешнего обучения M1 (Internet Streaming) ===")

    # Инициализация систем
    persistence = MemoryPersistence(filepath="memory_state.pkl")

    try:
        loaded_system = persistence.load(M1System)
        if loaded_system:
            system = loaded_system
            print("Возобновление обучения из memory_state.pkl")
        else:
            system = M1System()
    except Exception as e:
        print(f"Ошибка загрузки: {e}. Создание новой системы.")
        system = M1System()

    streamer = InternetStreamer()
    scheduler = TaskScheduler()
    logger = SimulationLogger()
    decoder = SDRDecoder()
    hw_clock = HardwareTimeCore()

    # max_steps = 1000
    start_time = time.time()
    test_duration = 60 # секунд
    step = 0
    last_progress_time = start_time

    while time.time() - start_time < test_duration:
        step += 1
        # 0. Проверка планировщика
        new_task = scheduler.check_for_new_tasks()
        if new_task:
            print(f"ПЛАНИРОВЩИК: Новое задание - {new_task['instruction']}")

        # 1. Логика SSD/Процессор: Получаем SDR токен только при аномалии или на первом шаге
        if step == 1 or system.NEED_TEACHER:
            env_input, char = streamer.get_next_char_sdr()
            if char is None:
                print("Поток данных завершен.")
                break
        else:
            # Режим "мышления" (внутренний цикл)
            env_input = None
            char = None

        # 2. Передаем сигнал в изолированную память
        alive = system.step(external_input=env_input)

        # 2.5 Обучение декодера и вывод лога
        if char:
            decoder.learn_mapping(-1, system.layer_minus_1.activations, char)

        decoded_out = decoder.decode_layer('minus_1', system.layer_minus_1.activations)
        activity_count = sum(layer.get_activity_count() for layer in system.sdr_layers)

        print(f"Шаг {step} | Bio-Tick: {system.biological_tick} | HW-Time: {hw_clock.get_current_timestamp()} | Активность: {activity_count}")
        print(f"  Layer -1: {decoded_out} | Anomaly: {system.NEED_TEACHER}")

        # Обработка аномалии (флаг NEED_TEACHER)
        if system.NEED_TEACHER:
            print(f"!!! АНОМАЛИЯ: Вызов Учителя.")

        # 3. Логирование
        logger.log_step(step, system)

        if not alive:
            print(f"Шаг {step}: Система умерла (энергия = {system.energy:.2f})")
            break

        # Прогресс раз в 60 секунд
        current_time = time.time()
        if current_time - last_progress_time >= 60:
            elapsed = int(current_time - start_time)
            print(f"Прогресс: прошло {elapsed}с... Шаг {step}, E={system.energy:.2f}")
            last_progress_time = current_time

    logger.save_to_csv("m1_v11_results.csv")
    print("Обучение завершено. Результаты сохранены в m1_v11_results.csv")

if __name__ == "__main__":
    main()
