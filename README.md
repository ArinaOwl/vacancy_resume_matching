# Матчинг описания вакансий и резюме (Курс "Глубокое обучение на практике", Команда 10)

**Цель проекта:** 
Разработка и реализация программного решения для сопоставления текстовых описаний вакансии и резюме.

**Описание проекта:** 
Сервис для платформы по поиску работы и сотрудников, позволяющий измерить релевантность резюме соискателя и  описания вакансии. Задача приложения - выдавать численную оценку, насколько конкретный соискатель отвечает требованиям вакансии.

## Формализация задачи

**Задача:**
Разработка модели машинного обучения для оценки релевантности между текстовыми описаниями вакансий и резюме соискателей.

**Входные данные:** 
Текстовые описания вакансии и резюме, предоставляемые пользователем.

**Выходные данные:**
Численная оценка релевантности между конкретным резюме и вакансией.

**Формулировка проблемы в терминах ML:**
Это задача бинарной классификации, где целью является определение, соответствует ли резюме требованиям вакансии.

**Подход к решению:**
- Использование методов обработки естественного языка (NLP) для векторизации текстов;
- Обучение модели бинарной классификации.

**Метрики:**
- Precision, Recall;
- Устойчивость к изменениям в оформлении описания вакансии и текста резюме;
- Скорость предсказания.

**Архитектуры:**
- BERT
- Siamese Networks

## Архитектура решения
![Архитектура решения](https://github.com/ArinaOwl/vacancy_resume_matching/blob/main/architecture.png)

## Получение эмбедингов
Чтобы применять модели к вакансиям и резюме, нужно перевести их из текстовой формы в числовую. Для этого мы использовали модель cointegrated/rubert-tiny для берта из библиотеки huggingface. Именно эту модель мы используем для того, чтобы она распознавала и русские и английские символы, при этом, время работы было бы не слишком большим.

## Предобработка данных
Для обучения моделей были взяты публичные датасеты [Открытые данные "Работа России"](https://trudvsem.ru/opendata/datasets).

[Предобработка датасета](https://github.com/ArinaOwl/vacancy_resume_matching/blob/main/data_preprocessing.ipynb) включила в себя:
- скачивание и приведение к единому формату датасетов матча, резюме, вакансий,
- фильтрация по IT вакансиям,
- преобразование данных в датасетах в удобный для использования формат,
- объединение полученных датасетов в один.
