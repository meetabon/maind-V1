#!/usr/bin/env python3
"""
Создание точки восстановления перед изменениями в коде памяти
"""

import os
import shutil
from datetime import datetime

def create_backup():
    """Создает бэкап всех важных файлов"""
    
    # Создаем папку для бэкапа с временной меткой
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = f"backup_{timestamp}"
    
    print(f"=== Создание точки восстановления ===")
    print(f"Папка бэкапа: {backup_dir}")
    
    os.makedirs(backup_dir, exist_ok=True)
    
    # Список файлов для бэкапа
    files_to_backup = [
        # Основные файлы системы
        "m1_v11/system.py",
        "m1_v11/neuron.py", 
        "m1_v11/environment.py",
        "m1_v11/energy.py",
        "m1_v11/evolution.py",
        "m1_v11/logger.py",
        "m1_v11/main.py",
        
        # Скрипты запуска и анализа
        "run_memory_compare.py",
        "plot_memory_comparison.py",
        "plot_results.py",
        
        # Результаты тестов
        "no_noise_m1_v11_results.csv",
        "with_noise_loaded_memory_m1_v11_results.csv",
        "memory_snapshot.npz",
        "memory_snapshot_B.npz",
        
        # Графики
        "memory_comparison_analysis.png",
        "memory_usage_analysis.png",
    ]
    
    # Копируем файлы
    backed_up = []
    for file_path in files_to_backup:
        if os.path.exists(file_path):
            # Создаем структуру папок в бэкапе
            backup_path = os.path.join(backup_dir, file_path)
            backup_folder = os.path.dirname(backup_path)
            
            if backup_folder:
                os.makedirs(backup_folder, exist_ok=True)
            
            shutil.copy2(file_path, backup_path)
            backed_up.append(file_path)
            print(f"✓ {file_path}")
        else:
            print(f"⚠ Файл не найден: {file_path}")
    
    # Создаем README для бэкапа
    readme_content = f"""# Точка восстановления M1 v1.1
Создано: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Состояние системы:
- Тест A (без шума): ошибка 0.002, энергия ~17425
- Тест B (с шумом + память): ошибка 0.036, энергия ~4165
- ПРОБЛЕМА: память не используется эффективно

## Забэкапленные файлы:
{chr(10).join(f"- {f}" for f in backed_up)}

## Для восстановления:
1. Скопируйте файлы из этой папки обратно в корень проекта
2. Перезапустите тесты: python run_memory_compare.py

## Планируемые изменения:
- Исправление приоритизации памяти в system.py
- Усиление влияния лучших позитивных примеров
- Ослабление влияния негативных примеров
"""
    
    with open(os.path.join(backup_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(readme_content)
    
    print(f"\n✓ Бэкап создан: {len(backed_up)} файлов")
    print(f"✓ README создан: {backup_dir}/README.md")
    print(f"\n🔄 Для восстановления:")
    print(f"   cp -r {backup_dir}/* .")
    print(f"   # или вручную скопируйте нужные файлы")
    
    return backup_dir

if __name__ == "__main__":
    backup_dir = create_backup()
    print(f"\n🎯 Точка восстановления готова: {backup_dir}")
    print("Теперь можно безопасно вносить изменения!")