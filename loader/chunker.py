"""
Модуль для разбиения текста на чанки (chunks).

Чанкинг - это процесс разделения большого текста на меньшие, управляемые части.
Это необходимо, потому что:
1. Модели эмбеддингов имеют ограничение на длину входного текста
2. Меньшие чанки дают более точные результаты поиска
3. Overlap (перекрытие) между чанками помогает не терять контекст на границах
"""

from typing import List, Dict


def chunk_text(
    text: str, 
    chunk_size: int = 500, 
    overlap: int = 100
) -> List[str]:
    """
    Разбивает текст на чанки с перекрытием.
    
    Args:
        text: Исходный текст для разбиения
        chunk_size: Размер одного чанка в символах (по умолчанию 500)
        overlap: Количество символов перекрытия между чанками (по умолчанию 100)
        
    Returns:
        Список текстовых чанков
        
    Example:
        >>> text = "Это длинный текст..." * 100
        >>> chunks = chunk_text(text, chunk_size=500, overlap=100)
        >>> print(f"Получено {len(chunks)} чанков")
    """
    # Проверяем входные параметры
    if chunk_size <= 0:
        raise ValueError("chunk_size должен быть больше 0")
    
    if overlap < 0:
        raise ValueError("overlap не может быть отрицательным")
    
    if overlap >= chunk_size:
        raise ValueError("overlap должен быть меньше chunk_size")
    
    # Если текст короче одного чанка, возвращаем его целиком
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    # Разбиваем текст на чанки с перекрытием
    while start < len(text):
        # Определяем конец текущего чанка
        end = start + chunk_size
        
        # Извлекаем чанк
        chunk = text[start:end]
        
        # Добавляем чанк в список
        chunks.append(chunk)
        
        # Сдвигаем начало для следующего чанка
        # Используем (chunk_size - overlap) чтобы создать перекрытие
        start += (chunk_size - overlap)
    
    return chunks


def chunk_text_smart(
    text: str, 
    chunk_size: int = 500, 
    overlap: int = 100,
    separators: List[str] = None
) -> List[str]:
    """
    "Умное" разбиение текста на чанки.
    Пытается разбивать текст по естественным границам (абзацы, предложения),
    а не просто по количеству символов.
    
    Args:
        text: Исходный текст
        chunk_size: Целевой размер чанка
        overlap: Перекрытие между чанками
        separators: Список разделителей в порядке приоритета
        
    Returns:
        Список текстовых чанков
    """
    # Разделители по умолчанию (от более важных к менее важным)
    if separators is None:
        separators = [
            '\n\n',  # Разрыв между абзацами
            '\n',    # Перенос строки
            '. ',    # Конец предложения
            '! ',    # Восклицательный знак
            '? ',    # Вопросительный знак
            '; ',    # Точка с запятой
            ', ',    # Запятая
            ' '      # Пробел (последний вариант)
        ]
    
    # Если текст короткий, возвращаем его целиком
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    
    # Рекурсивная функция для разбиения текста
    def split_text(text: str, separators: List[str]) -> List[str]:
        """Рекурсивно разбивает текст по разделителям"""
        if not text:
            return []
        
        if len(text) <= chunk_size:
            return [text]
        
        # Пробуем разделители по порядку
        for separator in separators:
            if separator in text:
                # Разбиваем текст по текущему разделителю
                parts = text.split(separator)
                
                result = []
                current_chunk = ""
                
                for part in parts:
                    # Добавляем разделитель обратно (кроме последней части)
                    part_with_sep = part + separator if part != parts[-1] else part
                    
                    # Если добавление части не превышает размер чанка
                    if len(current_chunk) + len(part_with_sep) <= chunk_size:
                        current_chunk += part_with_sep
                    else:
                        # Сохраняем текущий чанк, если он не пустой
                        if current_chunk:
                            result.append(current_chunk.strip())
                        
                        # Если сама часть слишком большая, разбиваем рекурсивно
                        if len(part_with_sep) > chunk_size:
                            # Используем следующие разделители
                            next_separators = separators[separators.index(separator) + 1:]
                            if next_separators:
                                result.extend(split_text(part_with_sep, next_separators))
                            else:
                                # Если больше нет разделителей, используем простое разбиение
                                result.extend(chunk_text(part_with_sep, chunk_size, overlap))
                            current_chunk = ""
                        else:
                            current_chunk = part_with_sep
                
                # Добавляем последний чанк
                if current_chunk:
                    result.append(current_chunk.strip())
                
                return result
        
        # Если не нашли подходящих разделителей, используем простое разбиение
        return chunk_text(text, chunk_size, overlap)
    
    # Выполняем разбиение
    chunks = split_text(text, separators)
    
    # Фильтруем пустые чанки
    chunks = [chunk for chunk in chunks if chunk.strip()]
    
    return chunks


def create_chunks_with_metadata(
    text: str,
    chunk_size: int = 500,
    overlap: int = 100,
    source: str = "",
    doc_type: str = "txt"
) -> List[Dict[str, any]]:
    """
    Создает чанки с метаданными.
    
    Args:
        text: Исходный текст
        chunk_size: Размер чанка
        overlap: Перекрытие
        source: Источник документа (имя файла)
        doc_type: Тип документа (txt, html и т.д.)
        
    Returns:
        Список словарей с чанками и метаданными
    """
    # Используем "умное" разбиение
    chunks = chunk_text_smart(text, chunk_size, overlap)
    
    # Создаем список чанков с метаданными
    chunks_with_metadata = []
    
    for i, chunk in enumerate(chunks):
        chunk_data = {
            "text": chunk,
            "metadata": {
                "source": source,
                "type": doc_type,
                "chunk_id": i,
                "total_chunks": len(chunks),
                "char_count": len(chunk)
            }
        }
        chunks_with_metadata.append(chunk_data)
    
    return chunks_with_metadata


if __name__ == "__main__":
    # Пример использования
    sample_text = """
    Это первый абзац текста. Он содержит несколько предложений. 
    Каждое предложение добавляет информацию.
    
    Это второй абзац. Он начинается с новой строки после пустой строки.
    Это помогает разделить логические блоки текста.
    
    Третий абзац может быть очень длинным и содержать много информации.
    В этом случае система чанкинга попытается разбить его на более мелкие части.
    """ * 10  # Умножаем, чтобы получить длинный текст
    
    print("=== Простое разбиение ===")
    simple_chunks = chunk_text(sample_text, chunk_size=200, overlap=50)
    print(f"Количество чанков: {len(simple_chunks)}")
    print(f"Первый чанк ({len(simple_chunks[0])} символов):")
    print(simple_chunks[0][:100] + "...")
    
    print("\n=== Умное разбиение ===")
    smart_chunks = chunk_text_smart(sample_text, chunk_size=200, overlap=50)
    print(f"Количество чанков: {len(smart_chunks)}")
    print(f"Первый чанк ({len(smart_chunks[0])} символов):")
    print(smart_chunks[0][:100] + "...")
    
    print("\n=== С метаданными ===")
    chunks_meta = create_chunks_with_metadata(
        sample_text[:500],  # Берем первые 500 символов для примера
        chunk_size=200,
        overlap=50,
        source="sample.txt",
        doc_type="txt"
    )
    print(f"Количество чанков с метаданными: {len(chunks_meta)}")
    print(f"Пример чанка с метаданными:")
    print(chunks_meta[0])

