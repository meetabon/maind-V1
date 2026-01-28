#!/usr/bin/env python3
"""
Визуализация результатов симуляции M1 v1.1
Создание графиков: E(t), ε(t), активность нейронов, корреляция ошибок
"""

import csv
import sys
import os

import numpy as np

# Установка backend перед импортом pyplot
import matplotlib
matplotlib.use('Agg')  # Используем backend без display

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

# Устанавливаем кириллицу
plt.rcParams['font.family'] = 'DejaVu Sans'

def load_data():
    """Загрузка данных из CSV файлов"""
    
    # 1. Энергия, ошибки и конфликты
    energy_data = {'step': [], 'energy': [], 'error': [], 'conflict': []}
    try:
        with open('m1_energy_error_timeline.csv', 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                energy_data['step'].append(int(row['step']))
                energy_data['energy'].append(float(row['energy']))
                energy_data['error'].append(float(row['error']))
                energy_data['conflict'].append(float(row.get('conflict', 0)))  # новая колонка
    except FileNotFoundError:
        print("⚠ Файл m1_energy_error_timeline.csv не найден")
        return None
    
    # 2. Активность по регионам
    activity_data = {'step': [], 'regions': [[] for _ in range(8)]}
    try:
        with open('m1_activity_per_region.csv', 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                activity_data['step'].append(int(row['step']))
                for i in range(8):
                    activity_data['regions'][i].append(float(row[f'region_{i}']))
    except FileNotFoundError:
        print("⚠ Файл m1_activity_per_region.csv не найден")
        return None
    
    # 3. Корреляция ошибок
    correlation_matrix = np.zeros((8, 8))
    try:
        with open('m1_error_correlation.csv', 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                for j in range(8):
                    correlation_matrix[i, j] = float(row[f'region_{j}'])
    except FileNotFoundError:
        print("⚠ Файл m1_error_correlation.csv не найден")
        return None
    
    return {
        'energy': energy_data,
        'activity': activity_data,
        'correlation': correlation_matrix
    }

def plot_energy_and_error(energy_data):
    """График 1: Энергия E(t), ошибка ε(t) и конфликт"""
    fig, ax1 = plt.subplots(figsize=(14, 6))
    
    steps = np.array(energy_data['step'])
    energies = np.array(energy_data['energy'])
    errors = np.array(energy_data['error'])
    conflicts = np.array(energy_data['conflict'])
    
    # Энергия (левая ось)
    color1 = '#2E86AB'  # синий
    ax1.set_xlabel('Время (шаги)', fontsize=12)
    ax1.set_ylabel('Энергия E(t)', color=color1, fontsize=12)
    line1 = ax1.plot(steps, energies, color=color1, linewidth=2.5, label='Энергия', marker='o', markersize=3, markevery=max(1, len(steps)//20))
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3)
    
    # Ошибка и конфликт (правые оси)
    ax2 = ax1.twinx()
    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 60))
    
    color2 = '#A23B72'  # красный
    color3 = '#F18F01'  # оранжевый
    
    ax2.set_ylabel('Ошибка ε(t)', color=color2, fontsize=12)
    line2 = ax2.plot(steps, errors, color=color2, linewidth=2.5, label='Ошибка', marker='s', markersize=3, markevery=max(1, len(steps)//20))
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylabel('Ошибка ε(t)', color=color2, fontsize=12)
    line2 = ax2.plot(steps, errors, color=color2, linewidth=2.5, label='Ошибка', marker='s', markersize=3, markevery=max(1, len(steps)//20))
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # Конфликт (третья ось справа)
    ax3.set_ylabel('Перцептивный конфликт', color=color3, fontsize=12)
    line3 = ax3.plot(steps, conflicts, color=color3, linewidth=2.5, label='Конфликт', marker='^', markersize=3, markevery=max(1, len(steps)//20))
    ax3.tick_params(axis='y', labelcolor=color3)
    
    # Легенда
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right', fontsize=11)
    
    plt.title('M1 v1.1: Энергия, ошибка и перцептивный конфликт', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('m1_energy_error_conflict_plot.png', dpi=150, bbox_inches='tight')
    print("✓ График 1 сохранён: m1_energy_error_conflict_plot.png")
    plt.close()

def plot_activity_per_region(activity_data):
    """График 2: Активность нейронов по регионам"""
    fig, ax = plt.subplots(figsize=(13, 7))
    
    steps = np.array(activity_data['step'])
    colors = plt.cm.Set3(np.linspace(0, 1, 8))  # 8 разных цветов
    
    region_names = [f'Регион {i}' for i in range(8)]
    
    for i in range(8):
        activities = np.array(activity_data['regions'][i])
        ax.plot(steps, activities, color=colors[i], linewidth=2.5, 
               label=region_names[i], marker='o', markersize=3, markevery=max(1, len(steps)//20))
    
    ax.set_xlabel('Время (шаги)', fontsize=12)
    ax.set_ylabel('Активность нейронов (%)', fontsize=12)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10, ncol=2, framealpha=0.9)
    
    plt.title('M1 v1.1: Активность нейронов по регионам', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('m1_activity_plot.png', dpi=150, bbox_inches='tight')
    print("✓ График 2 сохранён: m1_activity_plot.png")
    plt.close()

def plot_correlation_matrix(correlation_matrix):
    """График 3: Корреляционная матрица ошибок"""
    fig, ax = plt.subplots(figsize=(10, 9))
    
    # Создаём кастомную цветовую шкалу: синий (-1) → белый (0) → красный (+1)
    colors_list = ['#0000FF', '#FFFFFF', '#FF0000']  # синий, белый, красный
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list('correlation', colors_list, N=n_bins)
    
    # Визуализация
    im = ax.imshow(correlation_matrix, cmap=cmap, vmin=-1, vmax=1, aspect='auto')
    
    # Оси
    region_labels = [f'Рег. {i}' for i in range(8)]
    ax.set_xticks(range(8))
    ax.set_yticks(range(8))
    ax.set_xticklabels(region_labels, fontsize=10)
    ax.set_yticklabels(region_labels, fontsize=10)
    
    # Значения в ячейках
    for i in range(8):
        for j in range(8):
            value = correlation_matrix[i, j]
            text_color = 'white' if abs(value) > 0.5 else 'black'
            ax.text(j, i, f'{value:.2f}', ha='center', va='center',
                   color=text_color, fontsize=9, fontweight='bold')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Корреляция', fontsize=11)
    
    ax.set_xlabel('Регион', fontsize=12)
    ax.set_ylabel('Регион', fontsize=12)
    plt.title('M1 v1.1: Матрица корреляций ошибок предсказания между регионами', 
             fontsize=13, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('m1_correlation_matrix.png', dpi=150, bbox_inches='tight')
    print("✓ График 3 сохранён: m1_correlation_matrix.png")
    plt.close()

def create_summary_figure(data):
    """Создание сводного графика (все компоненты в одной фигуре)"""
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)
    
    # График 1: E(t), ε(t) и конфликт
    energy_data = data['energy']
    steps = np.array(energy_data['step'])
    energies = np.array(energy_data['energy'])
    errors = np.array(energy_data['error'])
    conflicts = np.array(energy_data['conflict'])
    
    ax1 = fig.add_subplot(gs[0, :])
    ax1_e = ax1.twinx()
    ax1_c = ax1.twinx()
    ax1_c.spines['right'].set_position(('outward', 60))
    
    ax1.plot(steps, energies, color='#2E86AB', linewidth=2.5, label='Энергия', marker='o', markersize=4, markevery=max(1, len(steps)//15))
    ax1_e.plot(steps, errors, color='#A23B72', linewidth=2.5, label='Ошибка', marker='s', markersize=4, markevery=max(1, len(steps)//15))
    ax1_c.plot(steps, conflicts, color='#F18F01', linewidth=2.5, label='Конфликт', marker='^', markersize=4, markevery=max(1, len(steps)//15))
    
    ax1.set_ylabel('Энергия E(t)', fontsize=11, color='#2E86AB')
    ax1_e.set_ylabel('Ошибка ε(t)', fontsize=11, color='#A23B72')
    ax1_c.set_ylabel('Конфликт', fontsize=11, color='#F18F01')
    ax1.grid(True, alpha=0.3)
    
    lines1 = ax1.get_lines() + ax1_e.get_lines() + ax1_c.get_lines()
    labels1 = [l.get_label() for l in lines1]
    ax1.legend(lines1, labels1, loc='upper right', fontsize=10, ncol=3)
    ax1.set_title('Энергия, ошибка и перцептивный конфликт', fontsize=12, fontweight='bold')
    
    # График 2: Активность по регионам (левая часть)
    ax2 = fig.add_subplot(gs[1:, 0])
    activity_data = data['activity']
    colors = plt.cm.Set3(np.linspace(0, 1, 8))
    for i in range(8):
        activities = np.array(activity_data['regions'][i])
        ax2.plot(steps, activities, color=colors[i], linewidth=2, label=f'Рег. {i}', marker='o', markersize=3, markevery=max(1, len(steps)//15))
    
    ax2.set_xlabel('Время (шаги)', fontsize=11)
    ax2.set_ylabel('Активность (%)', fontsize=11)
    ax2.set_ylim(0, 100)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9, ncol=2)
    ax2.set_title('Активность нейронов по регионам', fontsize=12, fontweight='bold')
    
    # График 3: Корреляционная матрица (правая часть)
    ax3 = fig.add_subplot(gs[1:, 1])
    correlation_matrix = data['correlation']
    
    colors_list = ['#0000FF', '#FFFFFF', '#FF0000']
    cmap = LinearSegmentedColormap.from_list('correlation', colors_list, N=100)
    im = ax3.imshow(correlation_matrix, cmap=cmap, vmin=-1, vmax=1, aspect='auto')
    
    region_labels = [f'Рег. {i}' for i in range(8)]
    ax3.set_xticks(range(8))
    ax3.set_yticks(range(8))
    ax3.set_xticklabels(region_labels, fontsize=9)
    ax3.set_yticklabels(region_labels, fontsize=9)
    
    for i in range(8):
        for j in range(8):
            value = correlation_matrix[i, j]
            text_color = 'white' if abs(value) > 0.5 else 'black'
            ax3.text(j, i, f'{value:.2f}', ha='center', va='center',
                    color=text_color, fontsize=8)
    
    cbar = plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
    cbar.set_label('Корреляция', fontsize=10)
    ax3.set_title('Матрица корреляций ошибок', fontsize=12, fontweight='bold')
    
    fig.suptitle('M1 v1.1 - Полный анализ с локальной ложью (Local Bias)', fontsize=14, fontweight='bold', y=0.995)
    
    plt.savefig('m1_full_summary_with_conflict.png', dpi=150, bbox_inches='tight')
    print("✓ Сводный график сохранён: m1_full_summary_with_conflict.png")
    plt.close()

def main():
    """Основная функция"""
    print("\n=== Визуализация результатов M1 v1.1 ===\n")
    
    # Загрузка данных
    data = load_data()
    if data is None:
        print("✗ Не удалось загрузить данные. Запустите сначала симуляцию:")
        print("  python run_m1_fixed.py")
        return
    
    print("✓ Данные загружены успешно\n")
    
    # Создание графиков
    plot_energy_and_error(data['energy'])
    plot_activity_per_region(data['activity'])
    plot_correlation_matrix(data['correlation'])
    create_summary_figure(data)
    
    print("\n✓ Все графики созданы успешно!")
    print("\nФайлы графиков:")
    print("  1. m1_energy_error_conflict_plot.png - энергия, ошибка и КОНФЛИКТ")
    print("  2. m1_activity_plot.png - активность по регионам")
    print("  3. m1_correlation_matrix.png - корреляционная матрица")
    print("  4. m1_full_summary_with_conflict.png - полный анализ с конфликтом")

if __name__ == "__main__":
    main()
