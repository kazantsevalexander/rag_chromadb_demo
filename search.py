"""
Скрипт для поиска по документам в ChromaDB.

Этот скрипт позволяет выполнять семантический поиск по документам,
которые были загружены с помощью ingest.py.

Использование:
    python search.py "Ваш поисковый запрос"
    
    # С дополнительными параметрами:
    python search.py "запрос" --n-results 5 --use-openai
    
    # Интерактивный режим:
    python search.py --interactive
"""

import sys
import argparse
from typing import Optional
import os

from chroma.chroma_client import ChromaDBClient


def display_results(results: dict, query: str):
    """
    Отображает результаты поиска в удобном формате.
    
    Args:
        results: Результаты поиска от ChromaDB
        query: Исходный поисковый запрос
    """
    print("\n" + "=" * 80)
    print(f"РЕЗУЛЬТАТЫ ПОИСКА")
    print("=" * 80)
    print(f"Запрос: {query}")
    print("=" * 80)
    
    # Проверяем, есть ли результаты
    if not results['documents'] or not results['documents'][0]:
        print("\n❌ Ничего не найдено")
        return
    
    # Выводим каждый результат
    for i, (doc, metadata, distance) in enumerate(zip(
        results['documents'][0],
        results['metadatas'][0],
        results['distances'][0]
    )):
        print(f"\n📄 Результат {i + 1}")
        print("-" * 80)
        
        # Информация о документе
        print(f"Источник: {metadata.get('source', 'N/A')}")
        print(f"Тип: {metadata.get('type', 'N/A').upper()}")
        print(f"Чанк: {metadata.get('chunk_id', 'N/A')} из {metadata.get('total_chunks', 'N/A')}")
        print(f"Релевантность: {1 - distance:.4f}")  # Преобразуем distance в оценку релевантности
        print(f"Distance: {distance:.4f}")
        
        # Текст результата
        print(f"\n📝 Текст:")
        print("-" * 80)
        # Обрезаем длинный текст для читаемости
        display_text = doc if len(doc) <= 500 else doc[:500] + "..."
        print(display_text)
        print("-" * 80)
    
    print()


def search_documents(
    query: str,
    n_results: int = 5,
    use_openai: bool = False,
    openai_api_key: Optional[str] = None,
    collection_name: str = "documents",
    filter_type: Optional[str] = None
):
    """
    Выполняет поиск по документам.
    
    Args:
        query: Поисковый запрос
        n_results: Количество результатов
        use_openai: Использовать ли OpenAI для эмбеддингов
        openai_api_key: API ключ OpenAI
        collection_name: Имя коллекции
        filter_type: Фильтр по типу документа (txt, html или None)
    """
    # Создаем клиента ChromaDB
    client = ChromaDBClient(
        persist_directory="./chroma_db",
        collection_name=collection_name
    )
    
    # Получаем коллекцию
    try:
        client.get_or_create_collection()
    except Exception as e:
        print(f"❌ Ошибка при подключении к ChromaDB: {str(e)}")
        print("\n💡 Подсказка: Убедитесь, что вы запустили ingest.py перед поиском!")
        sys.exit(1)
    
    # Проверяем, есть ли документы в коллекции
    stats = client.get_collection_stats()
    if stats.get('document_count', 0) == 0:
        print("❌ В коллекции нет документов!")
        print("\n💡 Подсказка: Сначала загрузите документы с помощью ingest.py")
        sys.exit(1)
    
    print(f"\n📊 В коллекции '{collection_name}': {stats['document_count']} документов")
    
    # Подготавливаем фильтр
    where = None
    if filter_type:
        where = {"type": filter_type}
        print(f"🔍 Фильтр: только документы типа '{filter_type}'")
    
    # Выполняем поиск
    try:
        print(f"\n🔍 Выполняется поиск...")
        
        if use_openai:
            results = client.search_with_openai_embedding(
                query=query,
                n_results=n_results,
                where=where,
                openai_api_key=openai_api_key
            )
        else:
            results = client.search(
                query=query,
                n_results=n_results,
                where=where
            )
        
        # Отображаем результаты
        display_results(results, query)
        
    except Exception as e:
        print(f"❌ Ошибка при поиске: {str(e)}")
        sys.exit(1)


def interactive_mode(
    use_openai: bool = False,
    openai_api_key: Optional[str] = None,
    collection_name: str = "documents"
):
    """
    Интерактивный режим поиска.
    Позволяет выполнять несколько поисковых запросов подряд.
    """
    print("\n" + "=" * 80)
    print("ИНТЕРАКТИВНЫЙ РЕЖИМ ПОИСКА")
    print("=" * 80)
    print("Введите 'exit' или 'quit' для выхода")
    print("Введите 'help' для справки")
    print("=" * 80)
    
    # Создаем клиента один раз
    client = ChromaDBClient(
        persist_directory="./chroma_db",
        collection_name=collection_name
    )
    
    try:
        client.get_or_create_collection()
        stats = client.get_collection_stats()
        print(f"\n📊 Документов в коллекции: {stats['document_count']}")
    except Exception as e:
        print(f"❌ Ошибка: {str(e)}")
        return
    
    while True:
        try:
            # Получаем запрос от пользователя
            query = input("\n🔍 Введите запрос: ").strip()
            
            if not query:
                continue
            
            # Обработка специальных команд
            if query.lower() in ['exit', 'quit', 'q']:
                print("\n👋 До свидания!")
                break
            
            if query.lower() == 'help':
                print("\n📖 Справка:")
                print("  - Просто введите ваш вопрос на естественном языке")
                print("  - Примеры: 'Что такое RAG?', 'корпоративная культура'")
                print("  - 'exit' или 'quit' - выход из программы")
                continue
            
            # Выполняем поиск
            search_documents(
                query=query,
                n_results=3,  # Меньше результатов в интерактивном режиме
                use_openai=use_openai,
                openai_api_key=openai_api_key,
                collection_name=collection_name
            )
            
        except KeyboardInterrupt:
            print("\n\n👋 Прервано пользователем. До свидания!")
            break
        except Exception as e:
            print(f"\n❌ Ошибка: {str(e)}")
            continue


def main():
    """
    Основная функция скрипта.
    """
    parser = argparse.ArgumentParser(
        description="Поиск по документам в ChromaDB",
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
        '--no-openai',
        action='store_true',
        help='НЕ использовать OpenAI (использовать встроенные эмбеддинги ChromaDB)'
    )
    parser.add_argument(
        '--openai-key',
        type=str,
        help='API ключ OpenAI'
    )
    parser.add_argument(
        '--collection',
        type=str,
        default='documents',
        help='Имя коллекции ChromaDB (по умолчанию: documents)'
    )
    parser.add_argument(
        '--filter-type',
        choices=['txt', 'html'],
        help='Фильтровать результаты по типу документа'
    )
    
    args = parser.parse_args()
    
    # Определяем, использовать ли OpenAI (по умолчанию ДА)
    use_openai = not args.no_openai
    
    # Интерактивный режим
    if args.interactive:
        interactive_mode(
            use_openai=use_openai,
            openai_api_key=args.openai_key,
            collection_name=args.collection
        )
        return
    
    # Обычный режим - требуется запрос
    if not args.query:
        parser.print_help()
        print("\n❌ Ошибка: Укажите поисковый запрос или используйте --interactive")
        sys.exit(1)
    
    # Выполняем поиск
    search_documents(
        query=args.query,
        n_results=args.n_results,
        use_openai=use_openai,
        openai_api_key=args.openai_key,
        collection_name=args.collection,
        filter_type=args.filter_type
    )


if __name__ == "__main__":
    main()

