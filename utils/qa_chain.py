# qa_chain.py

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_gigachat.chat_models import GigaChat
from .config import PROMPT_TEMPLATE

def create_qa_chain(vector_store, api_key):
    llm = GigaChat(
        credentials = api_key,
        scope = "GIGACHAT_API_PERS",
        model = "GigaChat-Pro",
        temperature = 0.01,
        verify_ssl_certs = False,
        streaming = False
    )
    
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
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
    
    return qa_chain
