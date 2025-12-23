"""
Клиент для работы с ChromaDB.

ChromaDB - это векторная база данных для хранения эмбеддингов.
Она позволяет:
1. Сохранять текстовые чанки с их векторными представлениями
2. Выполнять семантический поиск (находить похожие по смыслу тексты)
3. Хранить метаданные вместе с текстом
"""

import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional
import openai
import os


class ChromaDBClient:
    """
    Класс для работы с ChromaDB.
    Инкапсулирует логику создания коллекций, добавления документов и поиска.
    """
    
    def __init__(
        self, 
        persist_directory: str = "./chroma_db",
        collection_name: str = "documents"
    ):
        """
        Инициализирует клиент ChromaDB.
        
        Args:
            persist_directory: Директория для хранения данных ChromaDB
            collection_name: Имя коллекции для хранения документов
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        # Инициализируем ChromaDB клиент
        # persist_directory означает, что данные будут сохраняться на диск
        self.client = chromadb.Client(Settings(
            persist_directory=persist_directory,
            anonymized_telemetry=False  # Отключаем телеметрию
        ))
        
        # Получаем или создаем коллекцию
        self.collection = None
        
        print(f"ChromaDB инициализирован. Директория: {persist_directory}")
    
    def get_or_create_collection(self) -> chromadb.Collection:
        """
        Получает существующую коллекцию или создает новую.
        
        Returns:
            Объект коллекции ChromaDB
        """
        try:
            # Пытаемся получить существующую коллекцию
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "RAG документы для демонстрации"}
            )
            print(f"Коллекция '{self.collection_name}' готова к использованию")
            return self.collection
        except Exception as e:
            raise Exception(f"Ошибка при создании коллекции: {str(e)}")
    
    def delete_collection(self):
        """
        Удаляет коллекцию (полезно для очистки данных).
        """
        try:
            self.client.delete_collection(name=self.collection_name)
            print(f"Коллекция '{self.collection_name}' удалена")
        except Exception as e:
            print(f"Ошибка при удалении коллекции: {str(e)}")
    
    def add_documents(
        self,
        texts: List[str],
        metadatas: List[Dict],
        ids: Optional[List[str]] = None
    ):
        """
        Добавляет документы в коллекцию.
        
        ChromaDB автоматически создает эмбеддинги, если не указаны явно.
        
        Args:
            texts: Список текстовых чанков
            metadatas: Список метаданных для каждого чанка
            ids: Список уникальных идентификаторов (если None, генерируются автоматически)
        """
        if not self.collection:
            self.get_or_create_collection()
        
        # Генерируем ID, если они не предоставлены
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(texts))]
        
        try:
            # Добавляем документы в коллекцию
            # ChromaDB автоматически создаст эмбеддинги с помощью встроенной модели
            self.collection.add(
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            print(f"Добавлено {len(texts)} документов в коллекцию")
        except Exception as e:
            raise Exception(f"Ошибка при добавлении документов: {str(e)}")
    
    def add_documents_with_openai_embeddings(
        self,
        texts: List[str],
        metadatas: List[Dict],
        ids: Optional[List[str]] = None,
        openai_api_key: Optional[str] = None
    ):
        """
        Добавляет документы с эмбеддингами от OpenAI.
        
        Args:
            texts: Список текстовых чанков
            metadatas: Список метаданных
            ids: Список ID
            openai_api_key: API ключ OpenAI (если None, берется из переменной окружения)
        """
        if not self.collection:
            self.get_or_create_collection()
        
        # Настраиваем OpenAI API
        if openai_api_key:
            openai.api_key = openai_api_key
        elif os.getenv("OPENAI_API_KEY"):
            openai.api_key = os.getenv("OPENAI_API_KEY")
        else:
            raise ValueError(
                "OpenAI API key не найден. "
                "Установите переменную окружения OPENAI_API_KEY "
                "или передайте ключ явно"
            )
        
        # Генерируем ID, если они не предоставлены
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(texts))]
        
        try:
            # Создаем эмбеддинги через OpenAI
            print("Создание эмбеддингов через OpenAI...")
            embeddings = self._create_openai_embeddings(texts)
            
            # Добавляем документы с эмбеддингами
            self.collection.add(
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            print(f"Добавлено {len(texts)} документов с OpenAI эмбеддингами")
        except Exception as e:
            raise Exception(f"Ошибка при создании эмбеддингов: {str(e)}")
    
    def _create_openai_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Создает эмбеддинги с помощью OpenAI API.
        
        Args:
            texts: Список текстов
            
        Returns:
            Список векторов эмбеддингов
        """
        embeddings = []
        
        # Обрабатываем тексты батчами для эффективности
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            try:
                # Используем модель text-embedding-ada-002 (или text-embedding-3-small для более новой версии)
                response = openai.embeddings.create(
                    input=batch,
                    model="text-embedding-3-small"  # Или "text-embedding-ada-002"
                )
                
                # Извлекаем эмбеддинги из ответа
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)
                
                print(f"Обработано {len(embeddings)}/{len(texts)} текстов...")
                
            except Exception as e:
                print(f"Ошибка при создании эмбеддингов для батча: {str(e)}")
                raise
        
        return embeddings
    
    def search(
        self,
        query: str,
        n_results: int = 5,
        where: Optional[Dict] = None
    ) -> Dict:
        """
        Выполняет семантический поиск по коллекции.
        
        Args:
            query: Поисковый запрос
            n_results: Количество результатов для возврата
            where: Фильтр по метаданным (например, {"type": "txt"})
            
        Returns:
            Словарь с результатами поиска
        """
        if not self.collection:
            self.get_or_create_collection()
        
        try:
            # Выполняем поиск
            # ChromaDB автоматически создаст эмбеддинг для запроса
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where
            )
            
            return results
        except Exception as e:
            raise Exception(f"Ошибка при поиске: {str(e)}")
    
    def search_with_openai_embedding(
        self,
        query: str,
        n_results: int = 5,
        where: Optional[Dict] = None,
        openai_api_key: Optional[str] = None
    ) -> Dict:
        """
        Выполняет поиск с эмбеддингом от OpenAI.
        
        Args:
            query: Поисковый запрос
            n_results: Количество результатов
            where: Фильтр метаданных
            openai_api_key: API ключ OpenAI
            
        Returns:
            Словарь с результатами
        """
        if not self.collection:
            self.get_or_create_collection()
        
        # Настраиваем OpenAI API
        if openai_api_key:
            openai.api_key = openai_api_key
        elif os.getenv("OPENAI_API_KEY"):
            openai.api_key = os.getenv("OPENAI_API_KEY")
        else:
            raise ValueError("OpenAI API key не найден")
        
        try:
            # Создаем эмбеддинг для запроса
            query_embedding = self._create_openai_embeddings([query])[0]
            
            # Выполняем поиск с явным эмбеддингом
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where
            )
            
            return results
        except Exception as e:
            raise Exception(f"Ошибка при поиске: {str(e)}")
    
    def get_collection_stats(self) -> Dict:
        """
        Получает статистику коллекции.
        
        Returns:
            Словарь со статистикой
        """
        if not self.collection:
            self.get_or_create_collection()
        
        try:
            count = self.collection.count()
            return {
                "name": self.collection_name,
                "document_count": count
            }
        except Exception as e:
            return {"error": str(e)}


if __name__ == "__main__":
    # Пример использования
    print("=== Демонстрация ChromaDB клиента ===\n")
    
    # Создаем клиента
    client = ChromaDBClient(
        persist_directory="./test_chroma_db",
        collection_name="test_collection"
    )
    
    # Создаем коллекцию
    client.get_or_create_collection()
    
    # Пример данных
    sample_texts = [
        "Python - это высокоуровневый язык программирования.",
        "ChromaDB используется для хранения векторных эмбеддингов.",
        "RAG означает Retrieval-Augmented Generation."
    ]
    
    sample_metadatas = [
        {"source": "python_doc.txt", "type": "txt"},
        {"source": "chroma_doc.txt", "type": "txt"},
        {"source": "rag_doc.txt", "type": "txt"}
    ]
    
    # Добавляем документы (с встроенными эмбеддингами ChromaDB)
    print("Добавление документов...")
    client.add_documents(
        texts=sample_texts,
        metadatas=sample_metadatas
    )
    
    # Получаем статистику
    print("\nСтатистика коллекции:")
    stats = client.get_collection_stats()
    print(stats)
    
    # Выполняем поиск
    print("\n=== Поиск ===")
    query = "Что такое векторная база данных?"
    results = client.search(query, n_results=2)
    
    print(f"\nЗапрос: {query}")
    print(f"Найдено результатов: {len(results['documents'][0])}\n")
    
    for i, (doc, metadata, distance) in enumerate(zip(
        results['documents'][0],
        results['metadatas'][0],
        results['distances'][0]
    )):
        print(f"Результат {i+1}:")
        print(f"  Текст: {doc}")
        print(f"  Источник: {metadata['source']}")
        print(f"  Расстояние: {distance:.4f}")
        print()

