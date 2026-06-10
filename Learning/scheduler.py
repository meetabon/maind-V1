import json
import os
import time

class TaskScheduler:
    def __init__(self, task_file='Learning/tasks.json'):
        self.task_file = task_file
        self.current_task = None

    def check_for_new_tasks(self):
        """Проверка наличия новых задач от WebUI"""
        if os.path.exists(self.task_file):
            try:
                with open(self.task_file, 'r') as f:
                    self.current_task = json.load(f)
                # Удаляем файл после прочтения, чтобы не обрабатывать повторно
                os.remove(self.task_file)
                return self.current_task
            except Exception as e:
                print(f"Error reading task file: {e}")
        return None

    def get_search_queries(self):
        """Превращает текстовую задачу в список поисковых запросов для стримера"""
        if not self.current_task:
            return []

        instruction = self.current_task.get('instruction', '')
        # Простая логика xAI: извлекаем ключевые слова для поиска
        # В реальной системе здесь будет LLM-процессинг
        return [instruction, f"{instruction} основы", f"{instruction} википедия"]

def run_scheduler_loop():
    scheduler = TaskScheduler()
    while True:
        task = scheduler.check_for_new_tasks()
        if task:
            print(f"New Task Received: {task}")
            # Здесь можно инициировать запуск InternetStreamer с новыми URL
        time.sleep(5)

if __name__ == "__main__":
    run_scheduler_loop()
