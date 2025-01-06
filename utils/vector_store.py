# vector_store.py

import os
import json
import shutil
from langchain_gigachat.embeddings import GigaChatEmbeddings
from langchain.vectorstores import FAISS
import streamlit as st
from .config import VECTOR_STORE_DIR, INDEX_PATH, CHUNKS_PATH

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
def create_vector_store(chunk_texts, chunk_metadatas, api_key):
    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
    
    embeddings = GigaChatEmbeddings(
        credentials = api_key,
        scope = "GIGACHAT_API_PERS",
        verify_ssl_certs = False
    )
    
    vector_store = FAISS.from_texts(chunk_texts, embeddings, metadatas = chunk_metadatas)
    vector_store.save_local(INDEX_PATH)
    
    save_chunk_texts(CHUNKS_PATH, chunk_texts)
    return vector_store

# Функция для получения векторного хранилища
def get_vector_store(texts, metadatas, api_key):
    saved_chunk_texts = load_chunk_texts(CHUNKS_PATH)

    if saved_chunk_texts == texts:
        vector_store = FAISS.load_local(
            INDEX_PATH, 
            GigaChatEmbeddings(
                credentials = api_key,
                scope="GIGACHAT_API_PERS", 
                verify_ssl_certs = False
            ),
            allow_dangerous_deserialization = True
        )
    else:
        if os.path.exists(VECTOR_STORE_DIR):
            shutil.rmtree(VECTOR_STORE_DIR)
            
        with st.spinner('Еще немного... Пожалуйста, подождите...'):
            vector_store = create_vector_store(texts, metadatas, api_key)

        st.success('Теперь точно все! Добро пожаловать в наш чат-бот!')
    
    return vector_store
