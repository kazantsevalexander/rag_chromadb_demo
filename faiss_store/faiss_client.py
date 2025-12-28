"""
Клиент для работы с FAISS.

FAISS - это библиотека от Facebook для эффективного поиска по векторам.
Она позволяет:
1. Сохранять текстовые чанки с их векторными представлениями
2. Выполнять семантический поиск (находить похожие по смыслу тексты)
3. Хранить метаданные вместе с текстом
"""

import faiss
import numpy as np
import pickle
import os
from pathlib import Path
from typing import List, Dict, Optional
import openai


class FAISSClient:
    """
    Класс для работы с FAISS.
    Инкапсулирует логику создания индекса, добавления документов и поиска.
    """
    
    def __init__(
        self, 
        persist_directory: str = "./faiss_db",
        index_name: str = "documents"
    ):
        """
        Инициализирует клиент FAISS.
        
        Args:
            persist_directory: Директория для хранения данных FAISS
            index_name: Имя индекса для хранения документов
        """
        self.persist_directory = Path(persist_directory)
        self.index_name = index_name
        self.index = None
        self.documents = []  # Список текстов
        self.metadatas = []  # Список метаданных
        self.ids = []  # Список ID
        self.dimension = 1536  # Размерность для OpenAI embeddings
        
        # Создаем директорию если не существует
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        print(f"FAISS инициализирован. Директория: {persist_directory}")
    
    def _get_index_path(self) -> Path:
        return self.persist_directory / f"{self.index_name}.index"
    
    def _get_data_path(self) -> Path:
        return self.persist_directory / f"{self.index_name}.pkl"
    
    def create_index(self, dimension: int = 1536):
        """
        Создает новый FAISS индекс.
        
        Args:
            dimension: Размерность векторов
        """
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.documents = []
        self.metadatas = []
        self.ids = []
        print(f"Создан новый индекс с размерностью {dimension}")

    def load_index(self) -> bool:
        """
        Загружает существующий индекс с диска.
        
        Returns:
            True если индекс загружен, False если не найден
        """
        index_path = self._get_index_path()
        data_path = self._get_data_path()
        
        if index_path.exists() and data_path.exists():
            try:
                self.index = faiss.read_index(str(index_path))
                with open(data_path, 'rb') as f:
                    data = pickle.load(f)
                    self.documents = data['documents']
                    self.metadatas = data['metadatas']
                    self.ids = data['ids']
                    self.dimension = data.get('dimension', 1536)
                print(f"Индекс '{self.index_name}' загружен. Документов: {len(self.documents)}")
                return True
            except Exception as e:
                print(f"Ошибка при загрузке индекса: {str(e)}")
                return False
        return False
    
    def save_index(self):
        """Сохраняет индекс на диск."""
        if self.index is None:
            raise Exception("Индекс не создан")
        
        index_path = self._get_index_path()
        data_path = self._get_data_path()
        
        faiss.write_index(self.index, str(index_path))
        with open(data_path, 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'metadatas': self.metadatas,
                'ids': self.ids,
                'dimension': self.dimension
            }, f)
        print(f"Индекс сохранен в {self.persist_directory}")
    
    def delete_index(self):
        """Удаляет индекс (полезно для очистки данных)."""
        try:
            index_path = self._get_index_path()
            data_path = self._get_data_path()
            if index_path.exists():
                os.remove(index_path)
            if data_path.exists():
                os.remove(data_path)
            self.index = None
            self.documents = []
            self.metadatas = []
            self.ids = []
            print(f"Индекс '{self.index_name}' удален")
        except Exception as e:
            print(f"Ошибка при удалении индекса: {str(e)}")
    
    def _create_openai_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Создает эмбеддинги с помощью OpenAI API.
        
        Args:
            texts: Список текстов
            
        Returns:
            Массив векторов эмбеддингов
        """
        embeddings = []
        batch_size = 100
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            try:
                response = openai.embeddings.create(
                    input=batch,
                    model="text-embedding-3-small"
                )
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)
                print(f"Обработано {len(embeddings)}/{len(texts)} текстов...")
            except Exception as e:
                print(f"Ошибка при создании эмбеддингов: {str(e)}")
                raise
        
        return np.array(embeddings, dtype='float32')

    def add_documents(
        self,
        texts: List[str],
        metadatas: List[Dict],
        ids: Optional[List[str]] = None,
        openai_api_key: Optional[str] = None
    ):
        """
        Добавляет документы в индекс с эмбеддингами от OpenAI.
        
        Args:
            texts: Список текстовых чанков
            metadatas: Список метаданных для каждого чанка
            ids: Список уникальных идентификаторов
            openai_api_key: API ключ OpenAI
        """
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
            
            # Создаем индекс если не существует
            if self.index is None:
                self.create_index(dimension=embeddings.shape[1])
            
            # Добавляем векторы в индекс
            self.index.add(embeddings)
            
            # Сохраняем документы и метаданные
            self.documents.extend(texts)
            self.metadatas.extend(metadatas)
            self.ids.extend(ids)
            
            # Сохраняем на диск
            self.save_index()
            
            print(f"Добавлено {len(texts)} документов в индекс")
        except Exception as e:
            raise Exception(f"Ошибка при добавлении документов: {str(e)}")
    
    def search(
        self,
        query: str,
        n_results: int = 5,
        where: Optional[Dict] = None,
        openai_api_key: Optional[str] = None
    ) -> Dict:
        """
        Выполняет семантический поиск по индексу.
        
        Args:
            query: Поисковый запрос
            n_results: Количество результатов для возврата
            where: Фильтр по метаданным (например, {"type": "txt"})
            openai_api_key: API ключ OpenAI
            
        Returns:
            Словарь с результатами поиска
        """
        if self.index is None or self.index.ntotal == 0:
            raise Exception("Индекс пуст или не загружен")
        
        # Настраиваем OpenAI API
        if openai_api_key:
            openai.api_key = openai_api_key
        elif os.getenv("OPENAI_API_KEY"):
            openai.api_key = os.getenv("OPENAI_API_KEY")
        else:
            raise ValueError("OpenAI API key не найден")
        
        try:
            # Создаем эмбеддинг для запроса
            query_embedding = self._create_openai_embeddings([query])
            
            # Выполняем поиск
            distances, indices = self.index.search(query_embedding, min(n_results * 2, self.index.ntotal))
            
            # Фильтруем результаты
            documents = []
            metadatas_result = []
            distances_result = []
            
            for dist, idx in zip(distances[0], indices[0]):
                if idx == -1:
                    continue
                
                metadata = self.metadatas[idx]
                
                # Применяем фильтр если указан
                if where:
                    match = all(metadata.get(k) == v for k, v in where.items())
                    if not match:
                        continue
                
                documents.append(self.documents[idx])
                metadatas_result.append(metadata)
                distances_result.append(float(dist))
                
                if len(documents) >= n_results:
                    break
            
            return {
                'documents': [documents],
                'metadatas': [metadatas_result],
                'distances': [distances_result]
            }
        except Exception as e:
            raise Exception(f"Ошибка при поиске: {str(e)}")
    
    def get_index_stats(self) -> Dict:
        """
        Получает статистику индекса.
        
        Returns:
            Словарь со статистикой
        """
        if self.index is None:
            return {"name": self.index_name, "document_count": 0}
        
        return {
            "name": self.index_name,
            "document_count": len(self.documents),
            "vector_count": self.index.ntotal,
            "dimension": self.dimension
        }


if __name__ == "__main__":
    # Пример использования
    print("=== Демонстрация FAISS клиента ===\n")
    
    # Создаем клиента
    client = FAISSClient(
        persist_directory="./test_faiss_db",
        index_name="test_index"
    )
    
    # Пример данных
    sample_texts = [
        "Python - это высокоуровневый язык программирования.",
        "FAISS используется для хранения векторных эмбеддингов.",
        "RAG означает Retrieval-Augmented Generation."
    ]
    
    sample_metadatas = [
        {"source": "python_doc.txt", "type": "txt"},
        {"source": "faiss_doc.txt", "type": "txt"},
        {"source": "rag_doc.txt", "type": "txt"}
    ]
    
    # Добавляем документы
    print("Добавление документов...")
    client.add_documents(
        texts=sample_texts,
        metadatas=sample_metadatas
    )
    
    # Получаем статистику
    print("\nСтатистика индекса:")
    stats = client.get_index_stats()
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
