#!/usr/bin/env python3
"""
Исправление приоритизации памяти в M1 системе
Проблема: система не использует лучший позитивный пример эффективно
"""

def create_fixed_denoising_code():
    """Создает исправленный код для denoising с правильной приоритизацией"""
    
    fixed_code = '''
        # --- ИСПРАВЛЕННЫЙ Denoising: приоритизация лучших примеров ---
        cleaned_region_signals = []
        pos_min_sim = 0.05
        neg_penalty_scale = 0.2  # УМЕНЬШИЛИ влияние негативных
        max_blend = 0.98  # УВЕЛИЧИЛИ максимальное влияние позитивных

        for i, obs in enumerate(region_signals):
            cleaned = obs.copy()
            lm_list = self.long_memory[i]
            mem_conf = getattr(self.regions[i], 'memory_bias', 0.0)

            # ИСПРАВЛЕНИЕ 1: Найти ЛУЧШИЙ позитивный пример (по ошибке)
            best_positive = None
            negatives = []
            
            for e in lm_list:
                if e.get('type') == 'positive':
                    if best_positive is None or e.get('error', float('inf')) < best_positive.get('error', float('inf')):
                        best_positive = e
                else:
                    negatives.append(e)

            # ИСПРАВЛЕНИЕ 2: Усиленное влияние ЛУЧШЕГО позитивного примера
            if best_positive is not None and 'template' in best_positive:
                tpl = best_positive['template']
                sim_pos = _sim(obs, tpl)
                if sim_pos > 0.0:
                    # УВЕЛИЧИВАЕМ вес лучшего примера в 3 раза!
                    enhanced_weight = float(best_positive.get('weight', 1.0)) * mem_conf * 3.0
                    quality_bonus = max(0.0, 1.0 - best_positive.get('error', 1.0))  # бонус за качество
                    
                    alpha = min(max_blend, enhanced_weight * sim_pos * (1.0 + quality_bonus))
                    cleaned = (1.0 - alpha) * obs + alpha * tpl
                    
                    # Логирование для отладки
                    if self.step_count % 50 == 0:
                        print(f"[MEMORY] Регион {i}: используем лучший пример (ошибка={best_positive.get('error', 0):.4f}, alpha={alpha:.3f})")

            # ИСПРАВЛЕНИЕ 3: Выбираем наиболее релевантный негативный пример
            if negatives:
                # Сортируем негативные по релевантности (similarity * recency)
                scored_negatives = []
                for neg in negatives:
                    if 'template' in neg:
                        sim = _sim(obs, neg['template'])
                        recency = 1.0 / (self.step_count - neg.get('ts', 0) + 1)
                        score = sim * recency
                        scored_negatives.append((score, neg))
                
                if scored_negatives:
                    # Берем наиболее релевантный негативный
                    _, best_negative = max(scored_negatives, key=lambda x: x[0])
                    
                    tpln = best_negative['template']
                    sim_neg = _sim(obs, tpln)
                    if sim_neg > 0.0:
                        # УМЕНЬШИЛИ влияние негативных примеров
                        penalty = min(0.5, neg_penalty_scale * float(best_negative.get('weight', 0.5)) * mem_conf * sim_neg)
                        cleaned = cleaned * (1.0 - penalty)

            # ИСПРАВЛЕНИЕ 4: Слабое притяжение к лучшему позитивному даже при низкой схожести
            if best_positive is not None and 'template' in best_positive:
                tpl = best_positive['template']
                sim_pos = _sim(obs, tpl)
                if sim_pos < pos_min_sim:
                    # Увеличиваем слабое притяжение к лучшему примеру
                    quality_factor = max(0.1, 1.0 - best_positive.get('error', 1.0))
                    weak_alpha = min(0.4, 0.3 * mem_conf * float(best_positive.get('weight',1.0)) * quality_factor)
                    cleaned = (1.0 - weak_alpha) * cleaned + weak_alpha * tpl

            cleaned_region_signals.append(cleaned)
    '''
    
    return fixed_code

def analyze_memory_usage():
    """Анализ текущего использования памяти"""
    print("=== АНАЛИЗ ПРОБЛЕМЫ ПАМЯТИ ===")
    print()
    print("🔍 ТЕКУЩИЕ ПРОБЛЕМЫ:")
    print("1. Система использует первый негативный пример, а не лучший")
    print("2. Вес позитивного примера равен весу негативных")
    print("3. Нет бонуса за качество позитивного примера")
    print("4. Слабое притяжение к хорошим примерам")
    print()
    print("🔧 ПРЕДЛАГАЕМЫЕ ИСПРАВЛЕНИЯ:")
    print("1. Выбор ЛУЧШЕГО позитивного примера по ошибке")
    print("2. Утроение веса лучшего позитивного примера")
    print("3. Бонус за качество (1 - error)")
    print("4. Умный выбор негативных по релевантности")
    print("5. Усиленное слабое притяжение к качественным примерам")
    print()
    print("📊 ОЖИДАЕМЫЙ РЕЗУЛЬТАТ:")
    print("- Ошибка в тесте B должна снизиться с 0.036 до ~0.005-0.010")
    print("- Система будет активно использовать лучший опыт из памяти")
    print("- Негативные примеры будут меньше мешать")

if __name__ == "__main__":
    analyze_memory_usage()
    
    print("\n" + "="*60)
    print("ИСПРАВЛЕННЫЙ КОД ДЛЯ ВСТАВКИ В system.py:")
    print("="*60)
    print(create_fixed_denoising_code())