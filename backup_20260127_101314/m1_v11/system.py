"""
M1System - главный класс нейроморфной системы
Управляет 8 областями, энергетикой и эволюцией
"""

import numpy as np
from collections import deque
from environment import Environment
from neuron import NeuralRegion
from energy import EnergySystem
from evolution import EvolutionManager

class M1System:
    def __init__(self, meta_params=None):
        """Инициализация системы M1"""
        # Мета-параметры (наследуемые при размножении)
        if meta_params is None:
            self.meta_params = {
                'eta': 0.01,      # скорость обучения
                'lambda': 0.1,    # вес ошибки предсказания
                'long_link_prob': 0.05  # склонность к дальним связям
            }
        else:
            self.meta_params = meta_params.copy()
        
        # Компоненты системы
        self.environment = Environment()
        self.regions = [NeuralRegion(i, self.meta_params) for i in range(8)]
        # Ensure regions have low_activity flag
        for region in self.regions:
            setattr(region, 'low_activity', False)
        self.energy_system = EnergySystem()
        self.evolution = EvolutionManager()
        
        # Состояние системы
        self.energy = len(self.regions) * 32 * 1.0  # N_neurons * 1.0 (×2 от 0.5)
        self.step_count = 0
        self.alive = True
        # Сохраняем начальную энергию для порогов
        self.initial_energy = self.energy
        
        # Статистика
        self.error_history = []
        self.energy_history = []
        self.region_errors = [[] for _ in range(8)]  # ошибки по каждому региону
        # Параметр штрафа за перцептивный конфликт (κ)
        self.k_conflict = 5.0  # можно уменьшить до 2.0 для мягкого старта
        # Порог для применения штрафа (если 0 - штраф всегда применяется пропорционально конфликту)
        self.conflict_threshold = 0.0
        # Базовый метаболический расход: коэффициент на активный нейрон
        # self.k_base = 0.01 — по умолчанию небольшой постоянный расход
        self.k_base = 0.01

        # Короткая память (следы во времени)
        self.memory_length = 50  # можно в диапазоне 30-70
        self.recent_errors = deque(maxlen=self.memory_length)
        self.recent_conflicts = deque(maxlen=self.memory_length)
        self.recent_activity = deque(maxlen=self.memory_length)

        # Long-term memory (persistent between sessions) - per-region
        # Каждый элемент: dict { 'template': np.array(4x4x4), 'error': float, 'weight': float, 'ts': int, 'type': 'positive'|'negative' }
        self.long_memory = [[] for _ in range(len(self.regions))]
        self.max_long_memory_entries = 20  # including one positive + up to 19 negatives
        self.memory_delta = 0.001  # порог значимости при сравнении ошибок

        # Buffers for since-last-active period per region (used to compute recent_avg_error)
        self.since_active_errors = [[] for _ in range(len(self.regions))]
        self.since_active_signal_sum = [np.zeros((4,4,4)) for _ in range(len(self.regions))]
        self.since_active_count = [0 for _ in range(len(self.regions))]

        # Flags: mark regions that will enter rest on this allocation cycle; dump after updates
        self.will_rest = [False for _ in range(len(self.regions))]

        # Гомеостаз (режим пониженной активности) при критически низкой энергии
        self.energy_homeostasis_threshold = 0.15 * self.initial_energy
        self.in_homeostasis = False
        # Планирование отдыха/распределения задач
        self.allocation_interval = 10  # каждые N шагов пересматривать распределение
        self.rest_fraction = 0.5  # доля регионов, которые могут отдыхать по умолчанию
        # Порог корреляции для формирования дальних связей
        self.long_link_corr_threshold = 0.6
        # Параметры памяти
        self.memory_strength = 1.0  # добавочный вход от памяти (увеличено для эффекта seed)
        # Инициализация seed для памяти: небольшие значения ошибок по регионам
        seed_len = max(5, int(self.memory_length/10))
        for i in range(len(self.region_errors)):
            # небольшие начальные ошибки, чтобы память не была пуста
            self.region_errors[i] = [0.05 + 0.01 * np.random.random() for _ in range(seed_len)]
        # заполним recent deques начальными значениями
        for _ in range(seed_len):
            self.recent_errors.append(np.mean([np.mean(errs) for errs in self.region_errors]))
            self.recent_conflicts.append(0.0)
            self.recent_activity.append(0)
        # Базовый метаболический расход: коэффициент на активный нейрон
        # self.k_base = 0.01 — по умолчанию небольшой постоянный расход
        self.k_base = 0.01
        
    def step(self):
        """Один шаг симуляции"""
        if not self.alive:
            return False
            
        self.step_count += 1
        
        # 1. Обновление среды
        self.environment.update(self.step_count)
        
        # 2. Получение сигналов из среды для каждой области
        region_signals = self.environment.get_region_signals()
        
        # Планирование отдыха/доли активности на основе памяти (каждые allocation_interval шагов)
        if self.step_count % self.allocation_interval == 0:
            # Рассчитываем среднюю ошибку по регионам за окно памяти
            avg_errors = []
            for i in range(len(self.regions)):
                errors = self.region_errors[i][-self.memory_length:]
                if len(errors) == 0:
                    avg_errors.append(float('inf'))
                else:
                    avg_errors.append(np.mean(errors))

            # Если нет данных — пропускаем планирование
            if any(np.isfinite(avg_errors)):
                # сортируем регионы по средней ошибке (меньше -> можно отдыхать)
                idx_sorted = np.argsort(avg_errors)
                rest_count = max(1, int(len(self.regions) * self.rest_fraction))

                # При очень низкой энергии увеличиваем число отдыхающих
                if self.energy < 0.5 * self.initial_energy:
                    rest_count = min(len(self.regions)-1, rest_count + 1)

                # Пометим регионы для отдыха (low_activity)
                rest_idx = set(idx_sorted[:rest_count])
                # запомним предыдущие состояния low_activity, чтобы отметить входящие в отдых
                prev_low = [region.low_activity for region in self.regions]
                for i, region in enumerate(self.regions):
                    region.low_activity = (i in rest_idx)
                    # если регион переключается из active -> low_activity, пометим для дампа позже
                    if (i in rest_idx) and (not prev_low[i]):
                        self.will_rest[i] = True

                print(f"[ALLOC] Step={self.step_count}: отдыхают регионы={sorted(list(rest_idx))}")

                # Формирование дальних связей на основе корреляции ошибок в окне памяти
                # Собираем матрицу ошибок (регион x time)
                window_len = min(self.memory_length, max(len(errs) for errs in self.region_errors))
                if window_len >= 5:
                    errors_array = np.array([
                        (self.region_errors[i][-window_len:] + [0]*(window_len - len(self.region_errors[i][-window_len:])))
                        for i in range(len(self.regions))
                    ])
                    try:
                        corr = np.corrcoef(errors_array)
                        for i in range(len(self.regions)):
                            for j in range(i+1, len(self.regions)):
                                if corr[i, j] >= self.long_link_corr_threshold:
                                    # если связь ещё не создана — создать с вероятностью meta_params
                                    if j not in self.regions[i].long_range_connections and np.random.random() < self.meta_params.get('long_link_prob', 0.05):
                                        self.regions[i].add_long_link(j)
                                        self.regions[j].add_long_link(i)
                                        print(f"[LINK] Создана дальняя связь между {i} и {j} (corr={corr[i,j]:.2f})")
                    except Exception:
                        pass
        
        # 3. Обновление нейронных областей
        # Перед обновлением регионов вычислим memory_bias для каждого региона
        total_error = 0
        for i, region in enumerate(self.regions):
            errs = self.region_errors[i][-self.memory_length:]
            if len(errs) == 0:
                mean_err = 1.0
            else:
                mean_err = float(np.mean(errs))

            # Преобразуем ошибку в confidence памяти: меньше ошибка -> выше уверенность
            memory_confidence = max(0.0, 1.0 - mean_err / 0.5)
            region.memory_bias = memory_confidence * self.memory_strength

        # --- Denoising: применяем long_memory как корректор входов ---
        # Для каждого региона проверяем наличие положительного/негативного шаблона
        cleaned_region_signals = []
        # thresholds and caps
        pos_min_sim = 0.05  # minimal similarity to consider positive pull
        neg_penalty_scale = 0.3  # negatives produce only light penalty
        max_blend = 0.95

        for i, obs in enumerate(region_signals):
            cleaned = obs.copy()
            lm_list = self.long_memory[i]
            mem_conf = getattr(self.regions[i], 'memory_bias', 0.0)

            # find positive and negatives
            positive = None
            negatives = []
            for e in lm_list:
                if e.get('type') == 'positive':
                    positive = e
                else:
                    negatives.append(e)

            # similarity measure: normalized inverse-MSE -> higher = more similar
            def _sim(a, b):
                try:
                    mse = float(np.mean((a - b) ** 2))
                    denom = float(np.mean(a ** 2) + 1e-6)
                    sim = max(0.0, 1.0 - mse / (denom))
                    return min(1.0, max(0.0, sim))
                except Exception:
                    return 0.0

            # Positive anchor: pull towards positive template depending on similarity and confidence
            if positive is not None and 'template' in positive:
                tpl = positive['template']
                sim_pos = _sim(obs, tpl)
                if sim_pos > 0.0:
                    weight = float(positive.get('weight', 1.0)) * mem_conf
                    # even moderate similarity should pull input toward template
                    alpha = min(max_blend, weight * sim_pos)
                    cleaned = (1.0 - alpha) * obs + alpha * tpl

            # Negative experience: small suppressive bias if matches
            if negatives:
                # use most recent negative (first in list)
                neg = negatives[0]
                if neg is not None and 'template' in neg:
                    tpln = neg['template']
                    sim_neg = _sim(obs, tpln)
                    if sim_neg > 0.0:
                        # light penalty: scale down signal slightly toward zero
                        penalty = min(0.9, neg_penalty_scale * float(neg.get('weight', 0.5)) * mem_conf * sim_neg)
                        cleaned = cleaned * (1.0 - penalty)

            # If no confident positive match but a positive template exists, still bias slightly towards it
            if positive is not None and 'template' in positive:
                tpl = positive['template']
                sim_pos = _sim(obs, tpl)
                if sim_pos < pos_min_sim:
                    # weak exploratory pull to search for familiar shape
                    weak_alpha = min(0.2, 0.2 * mem_conf * float(positive.get('weight',1.0)))
                    cleaned = (1.0 - weak_alpha) * cleaned + weak_alpha * tpl

            cleaned_region_signals.append(cleaned)

        region_signals = cleaned_region_signals

        for i, region in enumerate(self.regions):
            error = region.update(region_signals[i])
            self.region_errors[i].append(error)  # логирование ошибок по регионам
            total_error += error
            # Если регион был активен в этом шаге, накопим ошибки и сигнал для периода активности
            if not region.low_activity:
                self.since_active_errors[i].append(error)
                try:
                    self.since_active_signal_sum[i] += region_signals[i]
                    self.since_active_count[i] += 1
                except Exception:
                    pass
        
        # сохраняем энергию до расчётов для диагностики
        energy_before = self.energy

        # 4. Расчет энергетических затрат
        energy_cost = self.energy_system.calculate_cost(self.regions)
        
        # 5. Получение энергии от среды
        energy_gain = self.energy_system.calculate_gain(self.regions, region_signals)
        
        # 6. Базовый метаболический расход (независимый от ошибки и среды)
        # Может зависеть от числа активных нейронов или быть константой
        # По умолчанию используем зависимость от числа активных нейронов
        try:
            n_active = sum(region.get_total_activity() for region in self.regions)
        except Exception:
            n_active = 0

        base_cost = self.k_base * n_active if hasattr(self, 'k_base') else 0.0

        # 6. Обновление энергии с учётом базового расхода и обычных затрат/приходов
        self.energy += energy_gain - energy_cost - base_cost
        
        # 7. Штраф за ошибки предсказания
        error_penalty = self.meta_params['lambda'] * total_error
        self.energy -= error_penalty

        # Штраф за неправильную уверенность памяти: confidence × error
        try:
            mem_penalty_sum = 0.0
            for i, region in enumerate(self.regions):
                last_err = self.region_errors[i][-1] if len(self.region_errors[i])>0 else 0.0
                mem_penalty_sum += region.memory_bias * float(last_err)
            # коэффициент штрафа памяти
            k_memory_penalty = getattr(self, 'k_memory_penalty', 5.0)
            memory_penalty = k_memory_penalty * mem_penalty_sum
            self.energy -= memory_penalty
        except Exception:
            pass

        # штраф за локальную перцептивную ложь / несогласованность регионов
        conflict = self.get_perception_conflict()
        if conflict > self.conflict_threshold:
            conflict_penalty = self.k_conflict * conflict
            self.energy -= conflict_penalty
        
        # 8. Проверка выживания
        if self.energy <= 0:
            self.alive = False
            # Сохраняем диагностику перед выходом
            self.last_diagnostics = {
                'energy_before': energy_before,
                'energy_gain': energy_gain,
                'energy_cost': energy_cost,
                'energy_cost_breakdown': self.energy_system.get_energy_breakdown(self.regions),
                'base_cost': base_cost,
                'error_penalty': error_penalty,
                'conflict': conflict,
                'conflict_penalty': (self.k_conflict * conflict) if conflict > self.conflict_threshold else 0.0,
                'n_active': n_active,
                'energy_after': self.energy
            }
            return False
        
        # 9. Обновление статистики
        self.error_history.append(total_error)
        self.energy_history.append(self.energy)

        # Сохраняем диагностику шага
        self.last_diagnostics = {
            'energy_before': energy_before,
            'energy_gain': energy_gain,
            'energy_cost': energy_cost,
            'energy_cost_breakdown': self.energy_system.get_energy_breakdown(self.regions),
            'base_cost': base_cost,
            'error_penalty': error_penalty,
            'conflict': conflict,
            'conflict_penalty': (self.k_conflict * conflict) if conflict > self.conflict_threshold else 0.0,
            'n_active': n_active,
            'energy_after': self.energy
        }

        # Обновление короткой памяти
        self.recent_errors.append(total_error)
        self.recent_conflicts.append(conflict)
        self.recent_activity.append(n_active)

        # После обновления регионов: обработка селективного дампа при отдыхе
        for i in range(len(self.regions)):
            if self.will_rest[i]:
                # вычисляем recent_avg_error за последний активный период региона
                errs = self.since_active_errors[i]
                if len(errs) > 0:
                    recent_avg_error = float(np.mean(errs))
                else:
                    recent_avg_error = float('inf')

                # template как среднее собранных сигналов
                tpl = None
                if self.since_active_count[i] > 0:
                    tpl = (self.since_active_signal_sum[i] / float(self.since_active_count[i])).astype(float)
                else:
                    tpl = np.zeros((4,4,4))

                # найдём существующий положительный опыт (если есть)
                positive = None
                for entry in self.long_memory[i]:
                    if entry.get('type') == 'positive':
                        positive = entry
                        break

                ts = self.step_count

                # Решение о положительном опыте: сохраняем как positive ТОЛЬКО если видно улучшение
                # Требование: error(t_last) < error(t_prev) (последний шаг показал снижение ошибки)
                improvement = False
                if len(self.since_active_errors[i]) >= 2:
                    prev_err = float(self.since_active_errors[i][-2])
                    last_err = float(self.since_active_errors[i][-1])
                    if last_err + self.memory_delta < prev_err:
                        improvement = True

                if positive is None:
                    # если нет положительного опыта — добавим текущий ТОЛЬКО при улучшении
                    if improvement:
                        new_entry = {'template': tpl, 'error': recent_avg_error, 'weight': 1.0, 'ts': ts, 'type': 'positive'}
                        self.long_memory[i].insert(0, new_entry)
                else:
                    # если улучшение по сравнению с лучшим — обновляем положительный опыт
                    if improvement and (recent_avg_error + self.memory_delta < positive.get('error', float('inf'))):
                        positive.update({'template': tpl, 'error': recent_avg_error, 'weight': positive.get('weight',1.0)+0.5, 'ts': ts})
                    # если ухудшение — сохраняем как negative опыт
                    elif recent_avg_error > positive.get('error', float('inf')) + self.memory_delta:
                        neg_entry = {'template': tpl, 'error': recent_avg_error, 'weight': 0.5, 'ts': ts, 'type': 'negative'}
                        # добавляем негативный опыт в конец
                        self.long_memory[i].append(neg_entry)

                # Ограничиваем размер long_memory: держим максимум max_long_memory_entries,
                # при этом гарантируем максимум 1 positive (в начале списка) и остальные negative
                # Сначала сохраним positive отдельно
                positives = [e for e in self.long_memory[i] if e.get('type') == 'positive']
                negatives = [e for e in self.long_memory[i] if e.get('type') != 'positive']
                # оставляем только самый лучший positive
                if len(positives) > 1:
                    positives = [min(positives, key=lambda x: x.get('error', float('inf')))]
                # оставляем последние (по времени) up to max-1 negatives
                negatives = sorted(negatives, key=lambda x: x.get('ts',0), reverse=True)[:max(0, self.max_long_memory_entries - len(positives))]
                # собираем обратно
                self.long_memory[i] = positives + negatives

                # очистка буферов периода активности
                self.since_active_errors[i].clear()
                self.since_active_signal_sum[i][:] = 0
                self.since_active_count[i] = 0
                self.will_rest[i] = False

        # Управление гомеостазом: при критически низкой энергии — снижать активность регионов
        if self.energy <= self.energy_homeostasis_threshold and not self.in_homeostasis:
            self.in_homeostasis = True
            for region in self.regions:
                region.low_activity = True
            print(f"! Вход в гомеостаз: энергия={self.energy:.2f} <= threshold={self.energy_homeostasis_threshold:.2f}")
        elif self.energy > self.energy_homeostasis_threshold and self.in_homeostasis:
            self.in_homeostasis = False
            for region in self.regions:
                region.low_activity = False
            print(f"! Выход из гомеостаза: энергия={self.energy:.2f} > threshold={self.energy_homeostasis_threshold:.2f}")
        
        return True
    
    def should_reproduce(self):
        """Проверка условий размножения"""
        return self.evolution.check_reproduction_trigger(
            self.error_history, 
            self.energy_history,
            self.step_count
        )
    
    def reproduce(self):
        """Создание новой системы с мутированными параметрами"""
        new_meta_params = self.evolution.mutate_params(self.meta_params)
        return M1System(new_meta_params)
    
    def get_avg_error(self):
        """Средняя ошибка за последние шаги"""
        if len(self.error_history) == 0:
            return 0.0
        return np.mean(self.error_history[-100:])
    
    def get_long_links_count(self):
        """Количество дальних связей"""
        count = 0
        for region in self.regions:
            count += region.get_long_links_count()
        return count
    
    def is_self_aware(self):
        """Примитивная проверка самосознания (для логирования)"""
        # TODO: реализовать отслеживание инварианта "я/не-я"
        return False
    
    def get_perception_conflict(self):
        """Степень перцептивного конфликта между регионами
        
        Если регионы видят одинаково → конфликт = 0
        Если видят совсем по-разному → конфликт = 1
        
        Вычисляется на основе расхождений в ошибках.
        """
        if len(self.region_errors[0]) == 0:
            return 0.0
        
        # Берём последние ошибки для каждого региона
        recent_errors = [
            errors[-1] if len(errors) > 0 else 0.0
            for errors in self.region_errors
        ]
        
        # Конфликт = отношение макс-мин ошибок к их среднему
        if len(recent_errors) == 0 or max(recent_errors) == 0:
            return 0.0
        
        max_error = max(recent_errors)
        min_error = min(recent_errors)
        mean_error = np.mean(recent_errors)
        
        if mean_error == 0:
            return 0.0
        
        # Нормализованный конфликт: (max - min) / mean
        conflict = (max_error - min_error) / (mean_error + 1e-6)
        return min(conflict, 1.0)  # Обрезаем до [0, 1]