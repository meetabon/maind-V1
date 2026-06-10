import datetime

class HardwareTimeCore:
    """Аппаратный регистр текущего системного времени"""

    def get_current_timestamp(self) -> str:
        """Возвращает строку формата YYYY-MM-DD HH:MM"""
        now = datetime.datetime.now()
        return now.strftime("%Y-%m-%d %H:%M")
