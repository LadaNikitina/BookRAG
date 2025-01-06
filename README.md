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

Мы понимаем, что далеко не у всех есть возможность запустить VPN или оплатить API ключ OpenAI и потестировать наш BookRAG. В случае, если у вас такой возможности нет, мы подготовили для вас реализацию на основе GigaChat. Нужно всего лишь перейти по [ссылке](https://github.com/LadaNikitina/BookRAG/tree/gigachat-main) (она же ветка gigachat-main) и продолжить оценивать наш проект :) 

А если у вас нет возможности развернуть локально, мы подняли для вас сервер по [ссылке](https://bookrag-oayoayaoaoay.streamlit.app/).

P.S. Тем более, что реализация на GigaChat быстрее работает ^_^

## Демо проекта

Демо проекта можно посмотреть по [ссылке](https://drive.google.com/file/d/1vo0ggH95uGgEKx4H_Fntg8SOLOCva6tk/view?usp=sharing). Записано для Giga-BookRAG, реализованного на основе GigaChat.

## Конечный результат

Развернуто локальное решение с frontend'ом, реализованным с помощью технологии Streamlit. В рамках решения реализован RAG в 2 версиях - на эмбеддингах и модели OpenAI, а также на эмбеддингах и модели GigaChat. Также, реализовано кеширование векторной базы - она будет построена лишь один раз и сохранена локально. В случае, если запуск будет произведен повторно и при этом чанки будут теми же, векторная база будет подгружена, что позволит сэкономить и время, и токены :)

## Метрики проекта (валидация RAG)

Подробнее о том, как вычислялись метрики, можно посмотреть по [ссылке](https://github.com/LadaNikitina/BookRAG/tree/metrics).

### **1. Faithfulness (достоверность)**  
**Faithfulness** — метрика, которая измеряет, насколько точно и достоверно ответ (или генерация) основан на предоставленном контексте. Она проверяет соответствие информации в ответе исходным данным, выявляя неточные или вымышленные утверждения.  

**Задачи Faithfulness:**  
- Сравнение ответа с контекстом.  
- Выявление галлюцинаций.  
- Проверка точности цитирования.  

### **2. Answer Relevancy (релевантность ответа)**  
**Answer relevancy** — метрика, оценивающая, насколько ответ модели релевантен заданному вопросу. Она определяет, отвечает ли модель на вопрос по существу, избегая отклонений от темы.  

**Задачи Answer Relevancy:**  
- Соответствие вопросу.  
- Отсутствие лишней информации в ответе.  
- Полнота ответа.  

### **Результаты тестирования**  
При использовании модели от OpenAI удалось достичь следующих показателей:  
- **Answer relevancy** — **0.91**  
- **Faithfulness** — **0.71**  

Эти результаты демонстрируют высокую релевантность ответов при умеренном уровне достоверности, что указывает на необходимость дальнейших улучшений в точности цитирования и предотвращении галлюцинаций.  

При использовании же модели от Gigachat удалось достичь следующих показателей:  
- **Answer relevancy** — **0.79**  
- **Faithfulness** — **0.52**

Модель GigaChat демонстрирует умеренную релевантность при низкой достоверности, что также указывает на необходимость доработки в точности цитирования и предотвращении галлюцинаций.

### **Примеры ответов моделей на один из вопросов** 
**Какое значение имеет сцена покаяния Дмитрия перед Алёшей?**
- **Ответ модели от OpenAI**:
1. Сцена покаяния Дмитрия перед Алёшей имеет глубокое значение в контексте романа "Братья Карамазовы". Она демонстрирует внутреннюю борьбу Дмитрия и его стремление к искуплению. Дмитрий, несмотря на свои пороки и ошибки, испытывает искреннее раскаяние и надежду на чудо, которое может изменить его судьбу. Это покаяние также подчеркивает его веру в божественное вмешательство и надежду на прощение. Взаимоотношения между братьями, особенно между Дмитрием и Алёшей, показывают, как духовная поддержка и понимание могут помочь человеку в его моральных и духовных поисках.

2. Цитата из текста: «Что ты? Я не помешан в уме, – пристально и даже как-то торжественно смотря, произнес Дмитрий Федорович. – Небось я тебя посылаю к отцу и знаю, что говорю: я чуду верю. – Чуду? – Чуду промысла Божьего. Богу известно мое сердце, он видит все мое отчаяние. Он всю эту картину видит. Неужели он попустит совершиться ужасу? Алеша, я чуду верю, иди!»

3. Источник: номер страницы 84, Книга третья, Глава V

- **Ответ модели Gigachat**:
В контексте сцены покаяния Дмитрия перед Алешей важно отметить несколько ключевых моментов. Во-первых, поклон старца Зосимы Дмитрию является символическим актом признания вины и раскаяния. Этот жест подчеркивает глубину внутреннего конфликта Дмитрия, который борется с собственными грехами и внутренними противоречиями. Поклон символизирует осознание своей ошибки и готовность принять наказание. Во-вторых, эта сцена показывает важность братских отношений между персонажами романа. Алеша, будучи духовным учеником старца Зосимы, глубоко переживает за судьбу своего брата Дмитрия. Его беспокойство отражает христианскую идею милосердия и прощения.В-третьих, поклон также демонстрирует смирение Дмитрия перед высшими силами и обществом. Он понимает, что должен признать свои ошибки и исправить их, чтобы обрести душевный покой.**Цитата из текста**:"Став на колени, старец поклонился Дмитрию Федоровичу в ноги полным, отчетливым, сознательным поклоном и даже лбом своим коснулся земли."*(Источник: стр. 53, Книга вторая, Глава VI)Таким образом, сцена покаяния Дмитрия перед Алешей имеет глубокий символический и нравственный смысл, подчеркивая темы вины, раскаяния и прощения в романе.

## Запуск приложения

### OpenAI API key

Для генерации ответов на вопросы наше решение использует модель от OpenAI, поэтому для использования чат-бота требуется
иметь собственный OpenAI API ключ, который может быть сгенерирован и оплачен
здесь: https://platform.openai.com/docs/overview

### Клонирование репозитория

```
git clone https://github.com/LadaNikitina/BookRAG.git
cd BookRAG
```

### Установка зависимостей

```
python3 -m venv venv
source venv/bin/activate  # Для macOS/Linux
# или
venv\Scripts\activate     # Для Windows

pip install -r requirements.txt
```

### Запуск и setup

Включите VPN, если он вам требуется для использования OpenAI.

Далее, для запуска приложения воспользуйтесь командой:

`streamlit run app.py
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
