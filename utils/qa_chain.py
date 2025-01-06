# qa_chain.py

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from .config import PROMPT_TEMPLATE

def create_qa_chain(vector_store, api_key):
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0.00,
        openai_api_key=api_key
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
