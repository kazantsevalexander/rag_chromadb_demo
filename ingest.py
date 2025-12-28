"""
Скрипт для загрузки документов в FAISS.

Этот скрипт:
1. Загружает текстовые и HTML файлы из папки data/
2. Разбивает их на чанки
3. Создает эмбеддинги через OpenAI
4. Сохраняет в FAISS для последующего поиска

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

from loader.txt_loader import load_txt
from loader.html_loader import load_html
from loader.chunker import create_chunks_with_metadata
from faiss_store.faiss_client import FAISSClient


def load_document(file_path: str) -> tuple[str, str]:
    """Загружает документ в зависимости от его типа."""
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
    """Обрабатывает список документов: загружает и разбивает на чанки."""
    all_chunks = []
    all_metadatas = []
    all_ids = []
    doc_counter = 0
    
    for file_path in file_paths:
        try:
            text, doc_type = load_document(file_path)
            print(f"  Загружено {len(text)} символов")
            
            chunks_with_meta = create_chunks_with_metadata(
                text=text,
                chunk_size=chunk_size,
                overlap=overlap,
                source=Path(file_path).name,
                doc_type=doc_type
            )
            
            print(f"  Создано {len(chunks_with_meta)} чанков")
            
            for chunk_data in chunks_with_meta:
                all_chunks.append(chunk_data['text'])
                all_metadatas.append(chunk_data['metadata'])
                chunk_id = f"doc_{doc_counter}_chunk_{chunk_data['metadata']['chunk_id']}"
                all_ids.append(chunk_id)
            
            doc_counter += 1
            print(f"  ✓ Файл обработан успешно\n")
            
        except Exception as e:
            print(f"  ✗ Ошибка при обработке файла: {str(e)}\n")
            continue
    
    return all_chunks, all_metadatas, all_ids


def ingest_to_faiss(
    texts: List[str],
    metadatas: List[Dict],
    ids: List[str],
    openai_api_key: str = None,
    persist_directory: str = "./faiss_db",
    index_name: str = "documents"
):
    """Загружает данные в FAISS."""
    print("=" * 60)
    print("Инициализация FAISS...")
    print("=" * 60)
    
    client = FAISSClient(
        persist_directory=persist_directory,
        index_name=index_name
    )
    
    print("\n" + "=" * 60)
    print("Добавление документов в FAISS...")
    print("=" * 60)
    
    try:
        client.add_documents(
            texts=texts,
            metadatas=metadatas,
            ids=ids,
            openai_api_key=openai_api_key
        )
        
        print("\n✓ Все документы успешно добавлены!")
        
        stats = client.get_index_stats()
        print("\n" + "=" * 60)
        print("Статистика индекса:")
        print("=" * 60)
        print(f"Название: {stats['name']}")
        print(f"Количество документов: {stats['document_count']}")
        
    except Exception as e:
        print(f"\n✗ Ошибка при добавлении документов: {str(e)}")
        sys.exit(1)


def main():
    """Основная функция скрипта."""
    parser = argparse.ArgumentParser(
        description="Загрузка документов в FAISS для RAG"
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
        '--openai-key',
        type=str,
        help='API ключ OpenAI (или используйте переменную окружения OPENAI_API_KEY)'
    )
    parser.add_argument(
        '--index',
        type=str,
        default='documents',
        help='Имя индекса FAISS (по умолчанию: documents)'
    )
    
    args = parser.parse_args()
    
    if args.files:
        file_paths = args.files
    else:
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
    
    print("=" * 60)
    print("RAG FAISS - ЗАГРУЗКА ДОКУМЕНТОВ")
    print("=" * 60)
    print(f"Файлов для обработки: {len(file_paths)}")
    print(f"Размер чанка: {args.chunk_size}")
    print(f"Перекрытие: {args.overlap}")
    print(f"Индекс: {args.index}")
    print("=" * 60)
    print()
    
    texts, metadatas, ids = process_documents(
        file_paths=file_paths,
        chunk_size=args.chunk_size,
        overlap=args.overlap
    )
    
    if not texts:
        print("✗ Не удалось обработать ни одного документа!")
        sys.exit(1)
    
    print(f"Всего обработано чанков: {len(texts)}\n")
    
    ingest_to_faiss(
        texts=texts,
        metadatas=metadatas,
        ids=ids,
        openai_api_key=args.openai_key,
        index_name=args.index
    )
    
    print("\n" + "=" * 60)
    print("ЗАГРУЗКА ЗАВЕРШЕНА!")
    print("=" * 60)
    print("\nТеперь вы можете использовать search.py для поиска по документам.")


if __name__ == "__main__":
    main()
