# app.py

import streamlit as st
from streamlit_chat import message
import json

from utils.config import (
    PAGE_TITLE,
    PAGE_LAYOUT,
    PDF_FILE_PATH,
)

from utils.data_processing import load_and_process_text
from utils.vector_store import get_vector_store
from utils.qa_chain import create_qa_chain

def main():
    # Инициализация приложения Streamlit
    st.set_page_config(page_title=PAGE_TITLE, layout=PAGE_LAYOUT)
    st.title(PAGE_TITLE)
    st.write("Задайте любой вопрос о книге 'Братья Карамазовы'. Введите 'выход' для завершения чата.")
    
    # Поле для ввода OpenAI API ключа
    st.sidebar.header("Настройки API")
    api_key = st.sidebar.text_input("Введите ваш OpenAI API ключ:", type="password", placeholder="Введите ваш API ключ здесь...")
    
    # Проверка на наличие ключа
    if not api_key:
        st.sidebar.warning("Пожалуйста, введите API ключ, чтобы продолжить.")
        st.stop()
    
    # Загрузка и обработка текста
    with st.spinner('Загрузка книги... Пожалуйста, подождите...'):
        chunks = load_and_process_text(PDF_FILE_PATH)
    
    st.success('Книга успешно загружена!')
    
    # Создание списка текстов для индексации
    texts = [chunk['chunk'] for chunk in chunks]
    metadatas = [chunk for chunk in chunks]
    
    # Получение или создание векторного хранилища
    vector_store = get_vector_store(texts, metadatas, api_key)
    
    # Создание цепочки Вопрос-Ответ
    qa_chain = create_qa_chain(vector_store, api_key)
    
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
    user_input = st.text_input("Ваш вопрос:", placeholder="Введите ваш вопрос здесь и нажмите Enter...", key='input')

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

if __name__ == "__main__":
    main()