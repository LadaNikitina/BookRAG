#!/usr/bin/env python
# coding: utf-8

# !pip install streamlit streamlit-chat
# !pip install langchain
# !pip install -U langchain-community
# !pip3 install sentence-transformers
# !pip3 install faiss-gpu
# !pip install openai
# !pip3 install langchain_openai
# !pip3 install pypdf
# !pip3 install tiktoken

from langchain_community.document_loaders import PyPDFLoader
import re
from typing import List, Dict

# Путь к PDF файлу
# PDF_FILE_PATH = '/kaggle/input/karamazovy/dostoevskiy_bratya_karamazovy.pdf'
PDF_FILE_PATH = 'book/dostoevskiy_bratya_karamazovy.pdf'

# Загрузка страниц из PDF
loader = PyPDFLoader(PDF_FILE_PATH)
pages = loader.load()[7:]

# # Обработка текста для каждой страницы
# for i in range(len(pages)):
#     # Текущий текст страницы
#     text = pages[i].page_content
    
#     # Убираем переносы слов (например, "припоминае-\nмого" -> "припоминаемого")
#     text = re.sub(r'-\n', '', text)
    
#     # Заменяем разрывы строк на пробелы (например, "\n" -> " ")
#     text = re.sub(r'\n', ' ', text)
    
#     # Удаляем лишние пробелы
#     text = re.sub(r'\s+', ' ', text).strip()
    
#     # Удаляем подстроку с названием книги и автором
#     text = re.sub(r'Ф\.?\s*М\.?\s*Достоевский\.?\s*«Братья Карамазовы»', '', text, flags=re.IGNORECASE)
    
#     # Перезаписываем текст страницы
#     pages[i].page_content = text

def split_text_into_chunks_with_metadata(pages, chunk_size=500):
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
                # 'chapter_title': current_chapter_title
            })
    return chunks

chunks = split_text_into_chunks_with_metadata(pages)

import faiss
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# Streamlit

import streamlit as st
from streamlit_chat import message

# Инициализация приложения Streamlit
st.set_page_config(page_title="Чат-бот: Братья Карамазовы", layout="centered")
st.title("Чат-бот: Братья Карамазовы")
st.write("Задайте любой вопрос о книге 'Братья Карамазовы'. Введите 'выход' для завершения чата.")

# Поле для ввода OpenAI API ключа
st.sidebar.header("Настройки API")
api_key = st.sidebar.text_input("Введите ваш OpenAI API ключ:", type="password", placeholder="Введите ваш API ключ здесь...")
st.sidebar.write("Пожалуйста, предоставьте ваш API ключ перед началом чата. Если ключ не предоставлен, бот вернёт стандартный ответ.")

# Инициализация модели эмбеддингов
embeddings = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=api_key)

# Создание списка текстов для индексации
texts = [chunk['chunk'] for chunk in chunks]
metadatas = [chunk for chunk in chunks]

# Создание векторного хранилища FAISS
vector_store = FAISS.from_texts(texts, embeddings, metadatas=metadatas)

# Сохранение индекса (опционально)
vector_store.save_local("faiss_index")

llm = ChatOpenAI(
    model="gpt-4o",           # Specify the GPT-4o model
    temperature=0.00,         # Low creativity for precise answers
    openai_api_key=api_key,   # Your OpenAI API key
    max_tokens=2048           # Adjust as needed
)

from langchain.prompts import PromptTemplate

template = """Вы — ассистент, специализирующийся на предоставлении точных ответов на вопросы по книге "Братья Карамазовы" Фёдора Достоевского.

Используйте предоставленные отрывки из книги для формирования ответов. Обязательно:

1. Приводите полный и развернутый ответ на вопрос.
2. Затем приводите точные цитаты из текста книги, подтверждающие ваш ответ, извлекая их из контекста.
3. Указывайте источник информации.


Контекст:
{context}

Вопрос: {question}

Ответ:
1. [Развернутый и полный ответ на вопрос]
2. [Цитата из текста: "..."]
3. [Источник: страница page, book, chapter]
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
        st.write("Чат завершён. Обновите страницу для начала нового разговора.")
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
