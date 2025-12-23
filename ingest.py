"""
Скрипт для загрузки документов в ChromaDB.

Этот скрипт:
1. Загружает текстовые и HTML файлы из папки data/
2. Разбивает их на чанки
3. Создает эмбеддинги
4. Сохраняет в ChromaDB для последующего поиска

Использование:
    python ingest.py
    
    # Или с указанием конкретных файлов:
    python ingest.py --files data/sample.txt data/sample.html
"""

import os
import sys
from pathlib import Path
from typing import List, Dict
import argparse

# Импортируем наши модули
from loader.txt_loader import load_txt
from loader.html_loader import load_html
from loader.chunker import create_chunks_with_metadata
from chroma.chroma_client import ChromaDBClient


def load_document(file_path: str) -> tuple[str, str]:
    """
    Загружает документ в зависимости от его типа.
    
    Args:
        file_path: Путь к файлу
        
    Returns:
        Кортеж (текст, тип_документа)
    """
    file_ext = Path(file_path).suffix.lower()
    
    print(f"Загрузка файла: {file_path}")
    
    if file_ext == '.txt':
        text = load_txt(file_path)
        return text, 'txt'
    elif file_ext in ['.html', '.htm']:
        text = load_html(file_path)
        return text, 'html'
    else:
        raise ValueError(f"Неподдерживаемый формат файла: {file_ext}")


def process_documents(
    file_paths: List[str],
    chunk_size: int = 500,
    overlap: int = 100
) -> tuple[List[str], List[Dict], List[str]]:
    """
    Обрабатывает список документов: загружает и разбивает на чанки.
    
    Args:
        file_paths: Список путей к файлам
        chunk_size: Размер чанка
        overlap: Перекрытие между чанками
        
    Returns:
        Кортеж (тексты_чанков, метаданные, идентификаторы)
    """
    all_chunks = []
    all_metadatas = []
    all_ids = []
    
    doc_counter = 0  # Счетчик документов
    
    for file_path in file_paths:
        try:
            # Загружаем документ
            text, doc_type = load_document(file_path)
            
            print(f"  Загружено {len(text)} символов")
            
            # Разбиваем на чанки с метаданными
            chunks_with_meta = create_chunks_with_metadata(
                text=text,
                chunk_size=chunk_size,
                overlap=overlap,
                source=Path(file_path).name,
                doc_type=doc_type
            )
            
            print(f"  Создано {len(chunks_with_meta)} чанков")
            
            # Подготавливаем данные для ChromaDB
            for chunk_data in chunks_with_meta:
                all_chunks.append(chunk_data['text'])
                all_metadatas.append(chunk_data['metadata'])
                
                # Создаем уникальный ID для каждого чанка
                chunk_id = f"doc_{doc_counter}_chunk_{chunk_data['metadata']['chunk_id']}"
                all_ids.append(chunk_id)
            
            doc_counter += 1
            print(f"  ✓ Файл обработан успешно\n")
            
        except Exception as e:
            print(f"  ✗ Ошибка при обработке файла: {str(e)}\n")
            continue
    
    return all_chunks, all_metadatas, all_ids


def ingest_to_chromadb(
    texts: List[str],
    metadatas: List[Dict],
    ids: List[str],
    use_openai: bool = False,
    openai_api_key: str = None,
    persist_directory: str = "./chroma_db",
    collection_name: str = "documents"
):
    """
    Загружает данные в ChromaDB.
    
    Args:
        texts: Список текстов
        metadatas: Список метаданных
        ids: Список ID
        use_openai: Использовать ли OpenAI для эмбеддингов
        openai_api_key: API ключ OpenAI
        persist_directory: Директория для хранения ChromaDB
        collection_name: Имя коллекции
    """
    print("=" * 60)
    print("Инициализация ChromaDB...")
    print("=" * 60)
    
    # Создаем клиента ChromaDB
    client = ChromaDBClient(
        persist_directory=persist_directory,
        collection_name=collection_name
    )
    
    # Получаем или создаем коллекцию
    client.get_or_create_collection()
    
    # Добавляем документы
    print("\n" + "=" * 60)
    print("Добавление документов в ChromaDB...")
    print("=" * 60)
    
    try:
        if use_openai:
            # Используем OpenAI эмбеддинги
            print("Режим: OpenAI Embeddings")
            client.add_documents_with_openai_embeddings(
                texts=texts,
                metadatas=metadatas,
                ids=ids,
                openai_api_key=openai_api_key
            )
        else:
            # Используем встроенные эмбеддинги ChromaDB
            print("Режим: ChromaDB встроенные эмбеддинги")
            client.add_documents(
                texts=texts,
                metadatas=metadatas,
                ids=ids
            )
        
        print("\n✓ Все документы успешно добавлены!")
        
        # Выводим статистику
        stats = client.get_collection_stats()
        print("\n" + "=" * 60)
        print("Статистика коллекции:")
        print("=" * 60)
        print(f"Название: {stats['name']}")
        print(f"Количество документов: {stats['document_count']}")
        
    except Exception as e:
        print(f"\n✗ Ошибка при добавлении документов: {str(e)}")
        sys.exit(1)


def main():
    """
    Основная функция скрипта.
    """
    # Парсинг аргументов командной строки
    parser = argparse.ArgumentParser(
        description="Загрузка документов в ChromaDB для RAG"
    )
    parser.add_argument(
        '--files',
        nargs='+',
        help='Список файлов для загрузки (по умолчанию все файлы из data/)'
    )
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=500,
        help='Размер чанка в символах (по умолчанию: 500)'
    )
    parser.add_argument(
        '--overlap',
        type=int,
        default=100,
        help='Перекрытие между чанками (по умолчанию: 100)'
    )
    parser.add_argument(
        '--no-openai',
        action='store_true',
        help='НЕ использовать OpenAI (использовать встроенные эмбеддинги ChromaDB)'
    )
    parser.add_argument(
        '--openai-key',
        type=str,
        help='API ключ OpenAI (или используйте переменную окружения OPENAI_API_KEY)'
    )
    parser.add_argument(
        '--collection',
        type=str,
        default='documents',
        help='Имя коллекции ChromaDB (по умолчанию: documents)'
    )
    
    args = parser.parse_args()
    
    # Определяем список файлов для обработки
    if args.files:
        file_paths = args.files
    else:
        # По умолчанию обрабатываем все файлы из data/
        data_dir = Path(__file__).parent / 'data'
        if not data_dir.exists():
            print(f"✗ Директория {data_dir} не найдена!")
            sys.exit(1)
        
        file_paths = [
            str(f) for f in data_dir.iterdir() 
            if f.suffix.lower() in ['.txt', '.html', '.htm']
        ]
    
    if not file_paths:
        print("✗ Не найдено файлов для обработки!")
        sys.exit(1)
    
    # Определяем, использовать ли OpenAI (по умолчанию ДА)
    use_openai = not args.no_openai
    
    print("=" * 60)
    print("RAG CHROMADB - ЗАГРУЗКА ДОКУМЕНТОВ")
    print("=" * 60)
    print(f"Файлов для обработки: {len(file_paths)}")
    print(f"Размер чанка: {args.chunk_size}")
    print(f"Перекрытие: {args.overlap}")
    print(f"Коллекция: {args.collection}")
    print(f"Эмбеддинги: {'OpenAI' if use_openai else 'ChromaDB встроенные'}")
    print("=" * 60)
    print()
    
    # Обрабатываем документы
    texts, metadatas, ids = process_documents(
        file_paths=file_paths,
        chunk_size=args.chunk_size,
        overlap=args.overlap
    )
    
    if not texts:
        print("✗ Не удалось обработать ни одного документа!")
        sys.exit(1)
    
    print(f"Всего обработано чанков: {len(texts)}\n")
    
    # Загружаем в ChromaDB
    ingest_to_chromadb(
        texts=texts,
        metadatas=metadatas,
        ids=ids,
        use_openai=use_openai,
        openai_api_key=args.openai_key,
        collection_name=args.collection
    )
    
    print("\n" + "=" * 60)
    print("ЗАГРУЗКА ЗАВЕРШЕНА!")
    print("=" * 60)
    print("\nТеперь вы можете использовать search.py для поиска по документам.")


if __name__ == "__main__":
    main()

