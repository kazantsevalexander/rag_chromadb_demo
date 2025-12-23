"""
Загрузчик для HTML-файлов.
Парсит HTML с помощью BeautifulSoup и извлекает чистый текст.
"""

import re
from bs4 import BeautifulSoup


def load_html(file_path: str) -> str:
    """
    Загружает HTML-файл, парсит его и извлекает текстовое содержимое.
    
    Args:
        file_path: Путь к HTML-файлу
        
    Returns:
        Очищенный текст из HTML
    """
    try:
        # Читаем HTML-файл
        with open(file_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Парсим HTML с помощью BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Удаляем script и style теги (они не содержат полезного текста)
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Извлекаем текст
        text = soup.get_text()
        
        # Очищаем текст
        cleaned_text = clean_html_text(text)
        
        return cleaned_text
    
    except FileNotFoundError:
        raise FileNotFoundError(f"Файл не найден: {file_path}")
    except Exception as e:
        raise Exception(f"Ошибка при парсинге HTML: {str(e)}")


def clean_html_text(text: str) -> str:
    """
    Очищает текст, извлеченный из HTML.
    Удаляет избыточные пробелы, переносы строк и нормализует форматирование.
    
    Args:
        text: Текст, извлеченный из HTML
        
    Returns:
        Очищенный текст
    """
    # Разбиваем на строки и удаляем пробелы с краев каждой строки
    lines = [line.strip() for line in text.split('\n')]
    
    # Удаляем пустые строки
    lines = [line for line in lines if line]
    
    # Соединяем строки обратно
    text = '\n'.join(lines)
    
    # Удаляем множественные пробелы
    text = re.sub(r' +', ' ', text)
    
    # Удаляем множественные переносы строк (оставляем максимум два подряд)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Убираем пробелы в начале и конце
    text = text.strip()
    
    return text


def extract_metadata_from_html(file_path: str) -> dict:
    """
    Извлекает метаданные из HTML (title, meta tags и т.д.).
    Полезно для сохранения дополнительной информации о документе.
    
    Args:
        file_path: Путь к HTML-файлу
        
    Returns:
        Словарь с метаданными
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        soup = BeautifulSoup(html_content, 'html.parser')
        
        metadata = {}
        
        # Извлекаем title
        if soup.title:
            metadata['title'] = soup.title.string
        
        # Извлекаем meta description
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        if meta_desc and meta_desc.get('content'):
            metadata['description'] = meta_desc['content']
        
        return metadata
    
    except Exception as e:
        print(f"Ошибка при извлечении метаданных: {str(e)}")
        return {}


if __name__ == "__main__":
    # Пример использования
    text = load_html("../data/sample.html")
    print(f"Загружено {len(text)} символов")
    print("\nПервые 200 символов:")
    print(text[:200])
    
    # Пример извлечения метаданных
    print("\n\nМетаданные:")
    metadata = extract_metadata_from_html("../data/sample.html")
    print(metadata)

