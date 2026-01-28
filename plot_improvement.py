#!/usr/bin/env python3
"""
График сравнения: до и после исправлений памяти
"""

import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'DejaVu Sans'

def load_csv_data(filename):
    """Загрузка данных из CSV"""
    data = {'steps': [], 'error': [], 'energy': []}
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                data['steps'].append(int(row['step']))
                data['error'].append(float(row['avg_error']))
                data['energy'].append(float(row['energy']))
    except FileNotFoundError:
        print(f"⚠ Файл {filename} не найден")
        return None
    return data

def create_improvement_plot():
    """Создание графика улучшений"""
    
    # Загружаем данные
    old_b = load_csv_data('with_noise_loaded_memory_m1_v11_results.csv')
    new_b = load_csv_data('fixed_with_noise_results.csv')
    
    if not old_b or not new_b:
        print("Не удалось загрузить данные")
        return
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # График 1: Ошибка предсказания
    ax1.plot(old_b['steps'], old_b['error'], 'r-', linewidth=2.5, 
             label=f'ДО исправлений (финал: {old_b["error"][-1]:.4f})', 
             marker='o', markersize=3, markevery=20)
    ax1.plot(new_b['steps'], new_b['error'], 'g-', linewidth=2.5, 
             label=f'ПОСЛЕ исправлений (финал: {new_b["error"][-1]:.4f})', 
             marker='s', markersize=3, markevery=20)
    
    ax1.set_xlabel('Шаги')
    ax1.set_ylabel('Ошибка предсказания')
    ax1.set_title('УЛУЧШЕНИЕ: Эффективность использования памяти', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Добавляем аннотацию улучшения
    improvement = (old_b['error'][-1] - new_b['error'][-1]) / old_b['error'][-1] * 100
    ax1.annotate(f'Улучшение: {improvement:.1f}%', 
                xy=(len(new_b['steps'])//2, max(old_b['error'])*0.8),
                xytext=(len(new_b['steps'])//2, max(old_b['error'])*0.9),
                arrowprops=dict(arrowstyle='->', color='green', lw=2),
                fontsize=14, color='green', fontweight='bold',
                ha='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
    
    # График 2: Энергия
    ax2.plot(old_b['steps'], old_b['energy'], 'r-', linewidth=2.5, 
             label=f'ДО исправлений (финал: {old_b["energy"][-1]:.0f})', 
             marker='o', markersize=3, markevery=20)
    ax2.plot(new_b['steps'], new_b['energy'], 'g-', linewidth=2.5, 
             label=f'ПОСЛЕ исправлений (финал: {new_b["energy"][-1]:.0f})', 
             marker='s', markersize=3, markevery=20)
    
    ax2.set_xlabel('Шаги')
    ax2.set_ylabel('Энергия')
    ax2.set_title('Энергетический баланс', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('memory_improvement_comparison.png', dpi=150, bbox_inches='tight')
    print("✓ График улучшений сохранен: memory_improvement_comparison.png")
    plt.close()
    
    # Статистика
    print(f"\n=== СТАТИСТИКА УЛУЧШЕНИЙ ===")
    print(f"Ошибка ДО:     {old_b['error'][-1]:.6f}")
    print(f"Ошибка ПОСЛЕ:  {new_b['error'][-1]:.6f}")
    print(f"Улучшение:     {improvement:.1f}%")
    print(f"Энергия ДО:    {old_b['energy'][-1]:.0f}")
    print(f"Энергия ПОСЛЕ: {new_b['energy'][-1]:.0f}")

if __name__ == "__main__":
    create_improvement_plot()