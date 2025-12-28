"""
Скрипт для поиска по документам в FAISS.

Использование:
    python search.py "Ваш поисковый запрос"
    
    # С дополнительными параметрами:
    python search.py "запрос" --n-results 5
    
    # Интерактивный режим:
    python search.py --interactive
"""

import sys
import argparse
from typing import Optional

from faiss_store.faiss_client import FAISSClient


def display_results(results: dict, query: str):
    """Отображает результаты поиска в удобном формате."""
    print("\n" + "=" * 80)
    print(f"РЕЗУЛЬТАТЫ ПОИСКА")
    print("=" * 80)
    print(f"Запрос: {query}")
    print("=" * 80)
    
    if not results['documents'] or not results['documents'][0]:
        print("\n❌ Ничего не найдено")
        return
    
    for i, (doc, metadata, distance) in enumerate(zip(
        results['documents'][0],
        results['metadatas'][0],
        results['distances'][0]
    )):
        print(f"\n📄 Результат {i + 1}")
        print("-" * 80)
        
        print(f"Источник: {metadata.get('source', 'N/A')}")
        print(f"Тип: {metadata.get('type', 'N/A').upper()}")
        print(f"Чанк: {metadata.get('chunk_id', 'N/A')} из {metadata.get('total_chunks', 'N/A')}")
        print(f"Distance: {distance:.4f}")
        
        print(f"\n📝 Текст:")
        print("-" * 80)
        display_text = doc if len(doc) <= 500 else doc[:500] + "..."
        print(display_text)
        print("-" * 80)
    
    print()


def search_documents(
    query: str,
    n_results: int = 5,
    openai_api_key: Optional[str] = None,
    index_name: str = "documents",
    filter_type: Optional[str] = None
):
    """Выполняет поиск по документам."""
    client = FAISSClient(
        persist_directory="./faiss_db",
        index_name=index_name
    )
    
    # Загружаем индекс
    if not client.load_index():
        print("❌ Индекс не найден!")
        print("\n💡 Подсказка: Убедитесь, что вы запустили ingest.py перед поиском!")
        sys.exit(1)
    
    stats = client.get_index_stats()
    if stats.get('document_count', 0) == 0:
        print("❌ В индексе нет документов!")
        print("\n💡 Подсказка: Сначала загрузите документы с помощью ingest.py")
        sys.exit(1)
    
    print(f"\n📊 В индексе '{index_name}': {stats['document_count']} документов")
    
    where = None
    if filter_type:
        where = {"type": filter_type}
        print(f"🔍 Фильтр: только документы типа '{filter_type}'")
    
    try:
        print(f"\n🔍 Выполняется поиск...")
        
        results = client.search(
            query=query,
            n_results=n_results,
            where=where,
            openai_api_key=openai_api_key
        )
        
        display_results(results, query)
        
    except Exception as e:
        print(f"❌ Ошибка при поиске: {str(e)}")
        sys.exit(1)


def interactive_mode(
    openai_api_key: Optional[str] = None,
    index_name: str = "documents"
):
    """Интерактивный режим поиска."""
    print("\n" + "=" * 80)
    print("ИНТЕРАКТИВНЫЙ РЕЖИМ ПОИСКА")
    print("=" * 80)
    print("Введите 'exit' или 'quit' для выхода")
    print("Введите 'help' для справки")
    print("=" * 80)
    
    client = FAISSClient(
        persist_directory="./faiss_db",
        index_name=index_name
    )
    
    if not client.load_index():
        print("❌ Индекс не найден! Сначала запустите ingest.py")
        return
    
    stats = client.get_index_stats()
    print(f"\n📊 Документов в индексе: {stats['document_count']}")
    
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
                print("  - Просто введите ваш вопрос на естественном языке")
                print("  - Примеры: 'Что такое RAG?', 'корпоративная культура'")
                print("  - 'exit' или 'quit' - выход из программы")
                continue
            
            results = client.search(
                query=query,
                n_results=3,
                openai_api_key=openai_api_key
            )
            display_results(results, query)
            
        except KeyboardInterrupt:
            print("\n\n👋 Прервано пользователем. До свидания!")
            break
        except Exception as e:
            print(f"\n❌ Ошибка: {str(e)}")
            continue


def main():
    """Основная функция скрипта."""
    parser = argparse.ArgumentParser(
        description="Поиск по документам в FAISS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python search.py "Что такое корпоративная культура?"
  python search.py "удаленная работа" --n-results 3
  python search.py "инструменты разработки" --filter-type txt
  python search.py --interactive
        """
    )
    
    parser.add_argument(
        'query',
        nargs='?',
        help='Поисковый запрос'
    )
    parser.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='Запустить в интерактивном режиме'
    )
    parser.add_argument(
        '--n-results', '-n',
        type=int,
        default=5,
        help='Количество результатов (по умолчанию: 5)'
    )
    parser.add_argument(
        '--openai-key',
        type=str,
        help='API ключ OpenAI'
    )
    parser.add_argument(
        '--index',
        type=str,
        default='documents',
        help='Имя индекса FAISS (по умолчанию: documents)'
    )
    parser.add_argument(
        '--filter-type',
        choices=['txt', 'html'],
        help='Фильтровать результаты по типу документа'
    )
    
    args = parser.parse_args()
    
    if args.interactive:
        interactive_mode(
            openai_api_key=args.openai_key,
            index_name=args.index
        )
        return
    
    if not args.query:
        parser.print_help()
        print("\n❌ Ошибка: Укажите поисковый запрос или используйте --interactive")
        sys.exit(1)
    
    search_documents(
        query=args.query,
        n_results=args.n_results,
        openai_api_key=args.openai_key,
        index_name=args.index,
        filter_type=args.filter_type
    )


if __name__ == "__main__":
    main()
