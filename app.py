import streamlit as st
import os
import subprocess
import time
import hydralit_components as hc
from ultralytics import YOLO
import moviepy.editor as moviepy    

def predict_similarity(vacancy, cv):
    time.sleep(5)
    return 0.92342334

st.markdown(
        f"<div style='text-align: left;'><h1>Опредление степени соответствия вакансии и резюме</h1>\n<h2>Входные данные:</h2></div>",
        unsafe_allow_html=True)

vacancy = st.text_area("Текст вакансии")

cv = st.text_area("Текст резюме")

run_model = st.button("Определить степень соответствия", use_container_width=True)
if run_model:
    if not vacancy or not cv:
        st.error('Поля со входными данными должны быть не пустые', icon="🚨")
    else:
        with st.container():
            with hc.HyLoader('Определяем...', hc.Loaders.standard_loaders, index=[1]):
                pred = predict_similarity(vacancy, cv)

        st.markdown(
            f"<div style='text-align: left;'><h2>Вакансия и резюме соответствуют на {pred * 100:.2f}%</h2></div>",
            unsafe_allow_html=True)
