"""
Единый файл для запуска всего пайплайна RAG с FAISS.

Этот скрипт:
1. Проверяет наличие OpenAI API ключа
2. Загружает документы из data/ в FAISS
3. Запускает интерактивный поиск

Использование:
    python run_all.py
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from loader.txt_loader import load_txt
from loader.html_loader import load_html
from loader.chunker import create_chunks_with_metadata
from faiss_store.faiss_client import FAISSClient


def check_openai_key():
    """Проверяет наличие OpenAI API ключа."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ ОШИБКА: OPENAI_API_KEY не найден!")
        print("\n💡 Установите ключ одним из способов:")
        print("   1. Переменная окружения:")
        print("      PowerShell: $env:OPENAI_API_KEY = 'sk-ваш_ключ'")
        print("      CMD: set OPENAI_API_KEY=sk-ваш_ключ")
        print("\n   2. Файл .env:")
        print("      Создайте файл .env и добавьте:")
        print("      OPENAI_API_KEY=sk-ваш_ключ")
        return False
    
    masked_key = api_key[:7] + "..." + api_key[-4:] if len(api_key) > 11 else "***"
    print(f"✅ OpenAI API ключ загружен ({masked_key})")
    return True


def load_document(file_path: str):
    """Загружает документ в зависимости от его типа."""
    file_ext = Path(file_path).suffix.lower()
    
    if file_ext == '.txt':
        return load_txt(file_path), 'txt'
    elif file_ext in ['.html', '.htm']:
        return load_html(file_path), 'html'
    else:
        raise ValueError(f"Неподдерживаемый формат: {file_ext}")


def ingest_documents():
    """Загружает документы в FAISS."""
    print("\n" + "=" * 70)
    print("ШАГ 1: ЗАГРУЗКА ДОКУМЕНТОВ В FAISS")
    print("=" * 70)
    
    data_dir = Path("data")
    if not data_dir.exists():
        print(f"❌ Директория {data_dir} не найдена!")
        return None
    
    file_paths = [
        str(f) for f in data_dir.iterdir() 
        if f.suffix.lower() in ['.txt', '.html', '.htm']
    ]
    
    if not file_paths:
        print("❌ Не найдено файлов для обработки!")
        return None
    
    print(f"Найдено файлов: {len(file_paths)}")
    print()
    
    all_chunks = []
    all_metadatas = []
    all_ids = []
    doc_counter = 0
    
    for file_path in file_paths:
        try:
            print(f"📄 Обработка: {Path(file_path).name}")
            text, doc_type = load_document(file_path)
            print(f"   Загружено: {len(text)} символов")
            
            chunks_with_meta = create_chunks_with_metadata(
                text=text,
                chunk_size=500,
                overlap=100,
                source=Path(file_path).name,
                doc_type=doc_type
            )
            print(f"   Чанков: {len(chunks_with_meta)}")
            
            for chunk_data in chunks_with_meta:
                all_chunks.append(chunk_data['text'])
                all_metadatas.append(chunk_data['metadata'])
                chunk_id = f"doc_{doc_counter}_chunk_{chunk_data['metadata']['chunk_id']}"
                all_ids.append(chunk_id)
            
            doc_counter += 1
            print(f"   ✅ Готово!\n")
            
        except Exception as e:
            print(f"   ❌ Ошибка: {str(e)}\n")
            continue
    
    if not all_chunks:
        print("❌ Не удалось обработать ни одного документа!")
        return None
    
    print(f"Всего чанков: {len(all_chunks)}")
    
    print("\n📦 Инициализация FAISS...")
    client = FAISSClient(
        persist_directory="./faiss_db",
        index_name="documents"
    )
    
    print("💾 Создание эмбеддингов и сохранение...")
    try:
        client.add_documents(
            texts=all_chunks,
            metadatas=all_metadatas,
            ids=all_ids
        )
        print("✅ Документы успешно загружены в FAISS!")
        return client
    except Exception as e:
        print(f"❌ Ошибка при сохранении: {str(e)}")
        return None


def display_results(results: dict, query: str):
    """Отображает результаты поиска."""
    print("\n" + "=" * 70)
    print(f"🔍 РЕЗУЛЬТАТЫ: {query}")
    print("=" * 70)
    
    if not results['documents'] or not results['documents'][0]:
        print("❌ Ничего не найдено")
        return
    
    for i, (doc, metadata, distance) in enumerate(zip(
        results['documents'][0],
        results['metadatas'][0],
        results['distances'][0]
    )):
        print(f"\n📄 Результат {i + 1}")
        print("-" * 70)
        print(f"Источник: {metadata.get('source', 'N/A')}")
        print(f"Тип: {metadata.get('type', 'N/A').upper()}")
        print(f"Distance: {distance:.4f}")
        print(f"\n📝 Текст:")
        display_text = doc if len(doc) <= 400 else doc[:400] + "..."
        print(display_text)
        print("-" * 70)


def interactive_search(client: FAISSClient):
    """Интерактивный режим поиска."""
    print("\n" + "=" * 70)
    print("ШАГ 2: ИНТЕРАКТИВНЫЙ ПОИСК")
    print("=" * 70)
    print("Введите 'exit' или 'quit' для выхода")
    print("Введите 'help' для справки")
    print("=" * 70)
    
    stats = client.get_index_stats()
    print(f"\n📊 В базе данных: {stats['document_count']} документов")
    
    while True:
        try:
            query = input("\n🔍 Введите запрос: ").strip()
            
            if not query:
                continue
            
            if query.lower() in ['exit', 'quit', 'q']:
                print("\n👋 До свидания!")
                break
            
            if query.lower() == 'help':
                print("\n📖 Справка:")
                print("  - Просто введите вопрос на естественном языке")
                print("  - Примеры:")
                print("    • Что такое корпоративная культура?")
                print("    • Какие инструменты использует команда?")
                print("    • Расскажи про удаленную работу")
                print("  - 'exit' или 'quit' - выход")
                continue
            
            try:
                results = client.search(query=query, n_results=3)
                display_results(results, query)
            except Exception as e:
                print(f"\n❌ Ошибка при поиске: {str(e)}")
                continue
            
        except KeyboardInterrupt:
            print("\n\n👋 Прервано пользователем. До свидания!")
            break
        except Exception as e:
            print(f"\n❌ Ошибка: {str(e)}")
            continue


def main():
    """Основная функция - запускает весь пайплайн."""
    print("=" * 70)
    print("🚀 RAG FAISS DEMO - ПОЛНЫЙ ПАЙПЛАЙН")
    print("=" * 70)
    
    import argparse
    parser = argparse.ArgumentParser(description="Запуск полного пайплайна RAG")
    parser.add_argument(
        '--openai-key',
        type=str,
        help='API ключ OpenAI'
    )
    
    args = parser.parse_args()
    
    if args.openai_key:
        os.environ['OPENAI_API_KEY'] = args.openai_key
    
    if not check_openai_key():
        sys.exit(1)
    
    faiss_db_path = Path("./faiss_db")
    if faiss_db_path.exists():
        print("\n💡 База данных уже существует!")
        response = input("Загрузить документы заново? (y/N): ").strip().lower()
        
        if response == 'y':
            print("🔄 Пересоздаем базу данных...")
            client = ingest_documents()
        else:
            print("📂 Используем существующую базу данных...")
            client = FAISSClient(
                persist_directory="./faiss_db",
                index_name="documents"
            )
            if not client.load_index():
                print("❌ Не удалось загрузить индекс!")
                sys.exit(1)
    else:
        client = ingest_documents()
    
    if not client:
        print("\n❌ Не удалось инициализировать систему!")
        sys.exit(1)
    
    interactive_search(client)
    
    print("\n" + "=" * 70)
    print("🎉 СПАСИБО ЗА ИСПОЛЬЗОВАНИЕ RAG FAISS DEMO!")
    print("=" * 70)


if __name__ == "__main__":
    main()
