#!/usr/bin/env python
# coding: utf-8

# !pip install langchain
# !pip install langchain-gigachat
# !pip install langchain-community
# !pip install faiss-gpu
# !pip install streamlit
# !pip install streamlit-chat

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from streamlit_chat import message
from typing import List, Dict

import faiss
import json
import os
import re
import streamlit as st
import shutil

# Путь к PDF файлу
# PDF_FILE_PATH = '/kaggle/input/karamazovy/dostoevskiy_bratya_karamazovy.pdf'
PDF_FILE_PATH = 'book/dostoevskiy_bratya_karamazovy.pdf'

# Streamlit

# Инициализация приложения Streamlit
st.set_page_config(page_title="Чат-бот: Братья Карамазовы", layout="centered")
st.title("Чат-бот: Братья Карамазовы")
st.write("Задайте любой вопрос о книге 'Братья Карамазовы'. Введите 'выход' для завершения чата.")

# Поле для ввода OpenAI API ключа
st.sidebar.header("Настройки API")
api_key = st.sidebar.text_input("Введите ваш OpenAI API ключ:", type="password", placeholder="Введите ваш API ключ здесь...")

# Проверка на наличие ключа
if not api_key:
    st.sidebar.warning("Пожалуйста, введите API ключ, чтобы продолжить.")
    st.stop()

# Функция для разбиения на чанки
@st.cache_data(show_spinner=False)
def load_and_process_text(pdf_path, chunk_size=500):
    # Выгрузка текста книги
    loader = PyPDFLoader(pdf_path)
    
    # Удаленим содержание
    pages = loader.load()[7:]

    current_book = None
    current_chapter = None
    current_chapter_title = None
    is_next_title = True

    chunks = []
    for page in pages:
        text = page.page_content
        page_number = page.metadata['page'] + 1

        # Поиск текущей книги, главы и названия
        for line in text.splitlines():
            book_match = re.match(r"^\s*Книга\s+(\w+)", line)
            chapter_match = re.match(r"^\s*([IVXLCDM]+)\s*$", line)
            
            if current_book and is_next_title:
                current_chapter_title = line.strip()
                is_next_title = False
                
            if book_match:
                current_book = f"Книга {book_match.group(1)}"
                
            if chapter_match:
                current_chapter = f"Глава {chapter_match.group(1)}"
                is_next_title = True 

        # Разбиваем текст на чанки
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i + chunk_size]
    
            # Убираем переносы слов (например, "припоминае-\nмого" -> "припоминаемого")
            chunk_text = re.sub(r'-\n', '', chunk)
            
            # Заменяем разрывы строк на пробелы (например, "\n" -> " ")
            chunk_text = re.sub(r'\n', ' ', chunk_text)
            
            # Удаляем лишние пробелы
            chunk_text = re.sub(r'\s+', ' ', chunk_text).strip()
            
            # Удаляем подстроку с названием книги и автором
            chunk_text = re.sub(r'Ф. М. Достоевский. «Братья Карамазовы»', '', chunk_text, flags=re.IGNORECASE)
    
            chunks.append({
                'chunk': f"Текст чанка {chunk_text}. | Источник: номер страницы {page_number}, {current_book}, {current_chapter}",
                'page': page_number,
                'book': current_book,
                'chapter': current_chapter,
            })
    return chunks

# Загрузка текста с анимацией
with st.spinner('Загрузка книги... Пожалуйста, подождите...'):
    chunks = load_and_process_text(PDF_FILE_PATH)

st.success('Текст книги успешно загружен!')

# Создание списка текстов для индексации
texts = [chunk['chunk'] for chunk in chunks]
metadatas = [chunk for chunk in chunks]

# Что будет происходить дальше: проверка, есть ли уже построенные эмбеддинги по конкретным чанкам
# Если нет, то построим их, если да, зачем их перестраивать

# Название директории
directory = "openai_vector_store_with_metainfo"

# Пути полные к чанкам и индексу
index_path = os.path.join(directory, "faiss_index")
chunks_path = os.path.join(directory, "chunks.json")

# Функции для работы с файлами
# Загрузка чанков
def load_chunk_texts(path):
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

# Сохранение чанков
def save_chunk_texts(path, chunk_texts):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(chunk_texts, f, ensure_ascii=False, indent=4)
        
# Функция для создания векторного хранилища
def create_vector_store(chunk_texts, chunk_metadatas):
    os.makedirs(directory, exist_ok=True)
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=api_key)

    vector_store = FAISS.from_texts(chunk_texts, embeddings, metadatas = chunk_metadatas)
    vector_store.save_local(index_path)
    
    save_chunk_texts(chunks_path, chunk_texts)
    return vector_store
        
# Основной код
saved_chunk_texts = load_chunk_texts(chunks_path)

if saved_chunk_texts == texts:
    vector_store = FAISS.load_local(
        index_path, 
        OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=api_key),
        allow_dangerous_deserialization = True
    )
else:
    if os.path.exists(directory):
        shutil.rmtree(directory)
        
    with st.spinner('Еще немного... Пожалуйста, подождите...'):
        vector_store = create_vector_store(texts, metadatas)
    
    st.success('Теперь точно все! Добро пожаловать в наш чат-бот!')

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.00,
    openai_api_key=api_key
)

template = """Вы — ассистент, специализирующийся на предоставлении точных и эстетически оформленных ответов на вопросы по книге "Братья Карамазовы" Фёдора Достоевского.  

Используйте предоставленные отрывки из книги для формирования ответов. Обязательно:

1. Приводите полный и развернутый ответ на вопрос.  
2. Затем приводите **точные и грамматически корректные цитаты** из текста книги, подтверждающие ваш ответ, извлекая их из контекста.  
3. Если цитата обрывается или выглядит неестественно (например, заканчивается на половине предложения), **исправьте её для красоты и логичности**, добавляя **троеточия ("...")** в начале или конце, чтобы обозначить пропуск текста.  
4. Указывайте источник каждой цитаты.  
5. Если контекст не содержит достаточной информации для ответа, сообщите об этом прямо, **не добавляя домыслов или предположений**.  

---

### **Шаблон ответа:**  

**Ответ:**  
Полный и развернутый ответ на вопрос.  

**Цитаты:**  
- "...Цитата 1..." (Источник)  
- "...Цитата 2..." (Источник)  

---

**Контекст:**  
{context}

**Вопрос:**  
{question}

---

**Ответ:**
"""

prompt = PromptTemplate(
    template=template,
    input_variables=["context", "question"]
)

# Создание цепочки Вопрос-Ответ с использованием RetrievalQA
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # Использование простого объединения
    retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
    return_source_documents=True,  # Добавляем возврат источников
    chain_type_kwargs={"prompt": prompt}
)

# Состояние сессии для хранения истории сообщений
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Отображение существующих сообщений
for i, chat in enumerate(st.session_state.messages):
    if chat['role'] == 'user':
        message(chat['content'], is_user=True, key=f"user_{i}")
    else:
        message(chat['content'], key=f"bot_{i}")

# Поле ввода для вопроса пользователя
user_input = st.text_input("Ваш вопрос:", placeholder="Введите ваш вопрос здесь и нажмите Enter...")

if user_input:
    if user_input.lower() in ['выход', 'exit', 'quit']:
        st.write("Чат завершён. Обновите страницу для начала нового диалога.")
    else:
        # Добавление сообщения пользователя в историю
        st.session_state.messages.append({"role": "user", "content": user_input})

        try:
            # Генерация ответа с использованием qa_chain
            response = qa_chain.invoke(user_input)['result']
        except Exception as e:
            response = f"Извините, произошла ошибка: {e}"

        # Добавление ответа бота в историю
        st.session_state.messages.append({"role": "bot", "content": response})

        # Отображение новых сообщений
        message(user_input, is_user=True, key=f"user_{len(st.session_state.messages) - 2}")
        message(response, key=f"bot_{len(st.session_state.messages) - 1}")
