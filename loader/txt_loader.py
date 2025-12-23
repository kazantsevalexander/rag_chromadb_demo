"""
Загрузчик для текстовых файлов.
Читает содержимое TXT-файла и выполняет базовую очистку текста.
"""

import re


def load_txt(file_path: str) -> str:
    """
    Загружает содержимое TXT-файла и выполняет очистку.
    
    Args:
        file_path: Путь к текстовому файлу
        
    Returns:
        Очищенный текст из файла
    """
    try:
        # Читаем файл с явным указанием кодировки
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Выполняем очистку текста
        cleaned_content = clean_text(content)
        
        return cleaned_content
    
    except FileNotFoundError:
        raise FileNotFoundError(f"Файл не найден: {file_path}")
    except Exception as e:
        raise Exception(f"Ошибка при чтении файла: {str(e)}")


def clean_text(text: str) -> str:
    """
    Очищает текст от лишних пробелов и пустых строк.
    
    Args:
        text: Исходный текст
        
    Returns:
        Очищенный текст
    """
    # Удаляем множественные пробелы и заменяем их на один
    text = re.sub(r' +', ' ', text)
    
    # Удаляем множественные переносы строк (более 2 подряд)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Удаляем пробелы в начале и конце строк
    lines = [line.strip() for line in text.split('\n')]
    
    # Удаляем полностью пустые строки (опционально, можно оставить для структуры)
    lines = [line for line in lines if line]
    
    # Соединяем строки обратно
    cleaned = '\n'.join(lines)
    
    # Удаляем пробелы в начале и конце всего текста
    cleaned = cleaned.strip()
    
    return cleaned


if __name__ == "__main__":
    # Пример использования
    text = load_txt("../data/sample.txt")
    print(f"Загружено {len(text)} символов")
    print("\nПервые 200 символов:")
    print(text[:200])

