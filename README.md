# Матчинг описания вакансий и резюме (Курс "Глубокое обучение на практике", Команда 10)

## Описание

**Цель проекта:** 
Разработка и реализация программного решения для сопоставления текстовых описаний вакансии и резюме.

**Описание проекта:** 
Сервис для платформы по поиску работы и сотрудников, позволяющий измерить релевантность резюме соискателя и  описания вакансии. Задача приложения - выдавать численную оценку, насколько конкретный соискатель отвечает требованиям вакансии.

## Пример работы приложения
![](demo/test.gif)
## Локальный запуск приложения
1. Установить библиотеку **Streamlit** согласно [официальной инструкции по установке](https://docs.streamlit.io/library/get-started/installation).

2. Установить дополнительные зависимости:

```
pip install -r requirements.txt
```

3. Запустить приложение из корня репозитория

```
streamlit run app.py
```

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
- Accuracy;
- AUC-ROC;
- AUC-PR.


## Архитектура решения
![Архитектура решения](https://github.com/ArinaOwl/vacancy_resume_matching/blob/main/architecture.png)

## Предобработка данных
Для обучения моделей были взяты публичные датасеты [Открытые данные "Работа России"](https://trudvsem.ru/opendata/datasets).

[Предобработка датасета](https://github.com/ArinaOwl/vacancy_resume_matching/blob/main/data_preprocessing.ipynb) включила в себя:
- скачивание и приведение к единому формату датасетов матча, резюме, вакансий,
- фильтрация по IT вакансиям,
- преобразование данных в датасетах в удобный для использования формат,
- объединение полученных датасетов в один,
- разметка датасета (добавление нерелевантных пар резюме-вакансия) на основании косинусной схожести описаний вакансий.

  
## Эксперименты

В нашем решении мы используем следующий алгоритм получения оценки для пары вакансия-резюме:
1. Текст резюме и вакансий преобразуется в векторное представление (embeddings) моделью Bert.
2. Векторные представления для резюме и вакансии конкатенируются и подаются на вход модели-классификатору.

Для получения векторных представлений из текста мы использовали модель cointegrated/rubert-tiny из huggingface. Модель распознавает как русские, так и английские символы, при этом время работы достаточно мало. 

Для получения similarity-score были рассмотрены следующие подходы:
- Cosine similarity
- kNN
- Ridge regression
- XGBoost
- SVM

Был произведен поиск гиперпараметров для моделей SVM и XGBoost. В результате подбора гиперпараметров получены следующие конфигурации моделей:
- **XGBoost** (colsample_bytree=0.28, learning_rate=0.09, max_depth=5, n_estimators=662, subsample=0.74)
- **SVM** (C=10, class_weight='balanced', gamma=1, max_iter=10000, tol=1e-06, kernel= 'rbf')

### Сравнение моделей
| Модель       | Accuracy| AUC-ROC |AUC-PR|
| -------------|:-------:| :------:|:----:|
| SVM   TUNED |0.97|0.97|0.89|
| XGBoost TUNED|0.98|0.90|0.90|
| Ridge regression NOT TUNED |-|0.90|-|
| kNN |-|0.70|-|
| Cosine similarity |-|0.70|-|

### Подбор гиперпараметров для XGBoost

Гиперпараметры подбирались с помощью библиотеки **hyperopt**:

![](demo/xgboost_hyperopt.png)

### Демонстрационное приложение

Написано с помощью библиотеки **streamlit**.
