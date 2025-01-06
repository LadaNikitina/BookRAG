# data_processing.py

import re
from langchain_community.document_loaders import PyPDFLoader
import streamlit as st

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
