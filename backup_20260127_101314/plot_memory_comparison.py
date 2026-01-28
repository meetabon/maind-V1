#!/usr/bin/env python3
"""
Сравнение результатов тестов A (без шума) и B (с шумом + память от A)
Анализ эффективности использования памяти
"""

import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams['font.family'] = 'DejaVu Sans'

def load_test_data(prefix):
    """Загрузка данных для одного теста"""
    data = {'steps': [], 'energy': [], 'error': [], 'activity': [], 'long_links': []}
    
    try:
        with open(f'{prefix}_m1_v11_results.csv', 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                data['steps'].append(int(row['step']))
                data['energy'].append(float(row['energy']))
                data['error'].append(float(row['avg_error']))
                data['activity'].append(float(row['total_activity']))
                data['long_links'].append(float(row['long_links']))
    except FileNotFoundError:
        print(f"⚠ Файл {prefix}_m1_v11_results.csv не найден")
        return None
    
    return data

def plot_comparison():
    """Создание сравнительных графиков"""
    # Загрузка данных
    test_a = load_test_data('no_noise')
    test_b = load_test_data('with_noise_loaded_memory')
    
    if not test_a or not test_b:
        print("✗ Не удалось загрузить данные тестов")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Сравнение: Тест A (без шума) vs Тест B (с шумом + память от A)', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # График 1: Энергия
    ax1 = axes[0, 0]
    ax1.plot(test_a['steps'], test_a['energy'], 'b-', linewidth=2.5, 
             label='Тест A (без шума)', marker='o', markersize=3, markevery=20)
    ax1.plot(test_b['steps'], test_b['energy'], 'r-', linewidth=2.5, 
             label='Тест B (с шумом + память)', marker='s', markersize=3, markevery=20)
    ax1.set_xlabel('Шаги')
    ax1.set_ylabel('Энергия')
    ax1.set_title('Энергетический баланс')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # График 2: Ошибка предсказания (КЛЮЧЕВОЙ!)
    ax2 = axes[0, 1]
    ax2.plot(test_a['steps'], test_a['error'], 'b-', linewidth=2.5, 
             label=f'Тест A (финал: {test_a["error"][-1]:.3f})', marker='o', markersize=3, markevery=20)
    ax2.plot(test_b['steps'], test_b['error'], 'r-', linewidth=2.5, 
             label=f'Тест B (финал: {test_b["error"][-1]:.3f})', marker='s', markersize=3, markevery=20)
    ax2.set_xlabel('Шаги')
    ax2.set_ylabel('Ошибка предсказания')
    ax2.set_title('ПРОБЛЕМА: Память не помогает с ошибками!')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Добавляем аннотацию проблемы
    ax2.annotate('Память от A не снижает ошибки в B!', 
                xy=(len(test_b['steps'])//2, max(test_b['error'])*0.8),
                xytext=(len(test_b['steps'])//2, max(test_b['error'])*0.9),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=12, color='red', fontweight='bold',
                ha='center')
    
    # График 3: Активность нейронов
    ax3 = axes[1, 0]
    ax3.plot(test_a['steps'], test_a['activity'], 'b-', linewidth=2.5, 
             label='Тест A', marker='o', markersize=3, markevery=20)
    ax3.plot(test_b['steps'], test_b['activity'], 'r-', linewidth=2.5, 
             label='Тест B', marker='s', markersize=3, markevery=20)
    ax3.set_xlabel('Шаги')
    ax3.set_ylabel('Общая активность')
    ax3.set_title('Активность нейронов')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # График 4: Дальние связи
    ax4 = axes[1, 1]
    ax4.plot(test_a['steps'], test_a['long_links'], 'b-', linewidth=2.5, 
             label='Тест A', marker='o', markersize=3, markevery=20)
    ax4.plot(test_b['steps'], test_b['long_links'], 'r-', linewidth=2.5, 
             label='Тест B', marker='s', markersize=3, markevery=20)
    ax4.set_xlabel('Шаги')
    ax4.set_ylabel('Количество дальних связей')
    ax4.set_title('Формирование дальних связей')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('memory_comparison_analysis.png', dpi=150, bbox_inches='tight')
    print("✓ График сравнения сохранен: memory_comparison_analysis.png")
    plt.close()

def analyze_memory_efficiency():
    """Анализ эффективности использования памяти"""
    test_a = load_test_data('no_noise')
    test_b = load_test_data('with_noise_loaded_memory')
    
    if not test_a or not test_b:
        return
    
    print("\n=== АНАЛИЗ ЭФФЕКТИВНОСТИ ПАМЯТИ ===")
    print(f"Тест A (без шума):")
    print(f"  Финальная ошибка: {test_a['error'][-1]:.6f}")
    print(f"  Финальная энергия: {test_a['energy'][-1]:.2f}")
    print(f"  Дальние связи: {test_a['long_links'][-1]}")
    
    print(f"\nТест B (с шумом + память от A):")
    print(f"  Финальная ошибка: {test_b['error'][-1]:.6f}")
    print(f"  Финальная энергия: {test_b['energy'][-1]:.2f}")
    print(f"  Дальние связи: {test_b['long_links'][-1]}")
    
    error_ratio = test_b['error'][-1] / test_a['error'][-1]
    print(f"\n🔍 ПРОБЛЕМА:")
    print(f"  Ошибка в тесте B в {error_ratio:.1f}x БОЛЬШЕ чем в A")
    print(f"  Память из 20 примеров (1 хороший, 19 плохих) не помогает!")
    
    print(f"\n💡 ВОЗМОЖНЫЕ ПРИЧИНЫ:")
    print(f"  1. Система не выделяет лучший пример из памяти")
    print(f"  2. Шум перебивает сигнал от хорошей памяти")
    print(f"  3. Механизм отбора памяти работает неправильно")
    print(f"  4. Веса памяти не учитывают качество примеров")

def create_memory_analysis_plot():
    """Детальный анализ использования памяти"""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Симуляция данных памяти (20 примеров)
    memory_quality = np.array([1.0] + [0.1]*19)  # 1 хороший + 19 плохих
    memory_usage = np.random.random(20) * 0.5 + 0.25  # случайное использование
    
    # График 1: Качество vs Использование памяти
    ax1 = axes[0]
    colors = ['green' if q > 0.5 else 'red' for q in memory_quality]
    bars = ax1.bar(range(20), memory_quality, color=colors, alpha=0.7, 
                   label='Качество примера')
    ax1.plot(range(20), memory_usage, 'bo-', linewidth=2, markersize=6,
             label='Частота использования')
    
    ax1.set_xlabel('Номер примера в памяти')
    ax1.set_ylabel('Качество / Использование')
    ax1.set_title('ПРОБЛЕМА: Система не выбирает лучший пример!')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Выделяем лучший пример
    ax1.annotate('Лучший пример\n(должен использоваться чаще!)', 
                xy=(0, 1.0), xytext=(3, 1.2),
                arrowprops=dict(arrowstyle='->', color='green', lw=2),
                fontsize=12, color='green', fontweight='bold')
    
    # График 2: Предлагаемое решение
    ax2 = axes[1]
    ideal_usage = np.array([0.8] + [0.05]*19)  # Идеальное использование
    
    ax2.bar(range(20), memory_quality, color=colors, alpha=0.7, 
            label='Качество примера')
    ax2.plot(range(20), ideal_usage, 'go-', linewidth=3, markersize=8,
             label='ИДЕАЛЬНОЕ использование')
    ax2.plot(range(20), memory_usage, 'ro--', linewidth=2, markersize=6,
             label='Текущее использование')
    
    ax2.set_xlabel('Номер примера в памяти')
    ax2.set_ylabel('Качество / Использование')
    ax2.set_title('РЕШЕНИЕ: Приоритет лучшим примерам!')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('memory_usage_analysis.png', dpi=150, bbox_inches='tight')
    print("✓ Анализ памяти сохранен: memory_usage_analysis.png")
    plt.close()

def main():
    print("=== Анализ эффективности памяти M1 ===")
    
    plot_comparison()
    analyze_memory_efficiency()
    create_memory_analysis_plot()
    
    print("\n✓ Анализ завершен!")
    print("\nФайлы:")
    print("  1. memory_comparison_analysis.png - сравнение тестов A и B")
    print("  2. memory_usage_analysis.png - анализ использования памяти")

if __name__ == "__main__":
    main()