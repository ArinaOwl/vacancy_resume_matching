import streamlit as st
import pandas as pd
import torch
import hydralit_components as hc
from xgboost import XGBClassifier
from transformers import AutoTokenizer, AutoModel

from model import svm

def load_models():
    with hc.HyLoader('Инициализация...', hc.Loaders.standard_loaders, index=[1]):
        tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny")
        model = AutoModel.from_pretrained("cointegrated/rubert-tiny")
        # model.cuda()  # uncomment it if you have a GPU

        classifier = XGBClassifier()
        classifier.load_model('./trained_models/xgboost_best.json')
    
    return tokenizer, model, classifier

if 'init' not in st.session_state:
    st.session_state['init'] = load_models()

tokenizer, model, classifier = st.session_state['init']

def embed_bert_cls(text, model, tokenizer):
    t = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**{k: v.to(model.device) for k, v in t.items()})
    embeddings = model_output.last_hidden_state[:, 0, :]
    embeddings = torch.nn.functional.normalize(embeddings)
    return embeddings[0].cpu().numpy()

def predict_similarity(vacancy, cv, classifier_type):
    columns = [f'vacancy_{i + 1}' for i in range(312)] + [f'cv_{i + 1}' for i in range(312)]

    vacancy_emb = embed_bert_cls(vacancy, model, tokenizer)
    cv_emb = embed_bert_cls(cv, model, tokenizer)

    df_embed = pd.DataFrame([vacancy_emb.tolist() + cv_emb.tolist()], columns=columns)

    if classifier_type == 'SVM':
        return svm.predict_proba(svm.model, df_embed)[0]
    elif classifier_type == 'XGBoost':
        return classifier.predict_proba(df_embed)[0, 1]
    else:
        return (svm.predict_proba(svm.model, df_embed)[0] + classifier.predict_proba(df_embed)[0, 1]) / 2

st.markdown(
        f"<div style='text-align: left;'><h1>Опредление степени соответствия вакансии и резюме</h1>\n<h2>Входные данные:</h2></div>",
        unsafe_allow_html=True)

classifier_type = st.selectbox(
    'Выберите классификатор',
    ('SVM', 'XGBoost', 'SVM + XGBoost average'))

vacancy = st.text_area("Текст вакансии")

cv = st.text_area("Текст резюме")

run_model = st.button("Определить степень соответствия", use_container_width=True)
if run_model:
    if not vacancy or not cv:
        st.error('Поля со входными данными должны быть не пустые', icon="🚨")
    else:
        with st.container():
            with hc.HyLoader('Определяем...', hc.Loaders.standard_loaders, index=[1]):
                pred = predict_similarity(vacancy, cv, classifier_type)

        st.markdown(
            f"<div style='text-align: left;'><h2>Вакансия и резюме соответствуют на {pred * 100:.2f}%</h2></div>",
            unsafe_allow_html=True)
