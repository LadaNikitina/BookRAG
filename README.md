# Проект BookRAG

Команда "Bookworms"

Участники: Леднева Дарья, Федотова Евгения, Клименченко Дмитрий

## Описание проекта

Чат-бот для ответов на вопросы по книге Федора Михайловича Достоевского "Братья Карамазовы". Пользователь
задает вопросы по сюжету/персонажам книги, ассистент предоставляет ответ на основе базы знаний по книге, с указанием
глав/частей книги, на основе которых был сформирован ответ.

### Использованные технологии

FAISS, LangChain, Streamlit

## Важное замечание

Мы понимаем, что далеко не у всех есть возможность запустить VPN или оплатить API ключ OpenAI и потестировать наш BookRAG. В случае, если у вас такой возможности нет, мы подготовили для вас реализацию на основе GigaChat. Нужно всего лишь перейти по [ссылке](https://github.com/LadaNikitina/BookRAG/tree/giga-main) (она же ветка giga-main) и продолжить оценивать наш проект :) 

P.S. Тем более, что реализация на GigaChat быстрее работает ^_^

## Демо проекта

Записано для Giga-BookRAG, реализованного на основе GigaChat.

<video width="640" height="360" controls>
  <source src="video.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

### Конечный результат

Развернуто локальное решение с frontend'ом, реализованным с помощью технологии Streamlit.

## Запуск приложения

### OpenAI API key

Для генерации ответов на вопросы наше решение использует модель от OpenAI, поэтому для использования чат-бота требуется
иметь собственный OpenAI API ключ, который может быть сгенерирован и оплачен
здесь: https://platform.openai.com/docs/overview

### Установка зависимостей

`pip install streamlit streamlit-chat`

`pip install langchain`

`pip install -U langchain-community`

`pip install sentence-transformers`

`pip install faiss-gpu`

`pip install openai`

`pip install langchain_openai`

`pip install pypdf`

`pip install tiktoken`

### Запуск и setup

Для запуска приложения воспользуйтесь командой:

`streamlit run rag-solution-with-streamlit.py
`

Steamlit-приложение автоматически развернется в вашем браузере на localhost.

Введите ваш OpenAI API ключ в соответствующее поле в интерфейсе веб-приложения:

![image](https://github.com/user-attachments/assets/5202ed3c-2544-45a6-86a0-bf61630def07)

После ввода корректного ключа будет произведена индексация книги и построена векторная база, необходимая для работы RAG.
После этого можно пользоваться чат-ботом и задавать интересующие вопросы.

## Работа приложения

В соответствующем поле введите ваш вопрос по книге Ф.М. Достоевского "Братья Карамазовы". Чатбот вернет ответ, состоящий
из непосредственно ответа на поставленный вопрос, цитаты из текста книги и источника цитаты.

![image](https://github.com/user-attachments/assets/208760a1-9746-469f-8bcb-84a67c437d01)

Наше приложение позволяет задавать неограниченное количество вопросов, поддерживая с ботом длительный диалог.

![image](https://github.com/user-attachments/assets/5c36df10-5e66-444a-a642-0fe3a821f600)

Для завершения диалога с чат-ботом введите "exit"/"выход"/"quit".

![image](https://github.com/user-attachments/assets/db83adb6-632b-487f-9fbd-404ff3282352)
