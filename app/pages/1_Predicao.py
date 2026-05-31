"""
app/pages/1_Predicao.py
=======================
Página de predição — formulário clínico → predição via pipeline serializada.
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Permite importar app_utils (um nível acima de pages/)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app_utils import (  # noqa: E402
    CAEC_CALC_OPTIONS, CH2O_LABELS, CLASS_COLORS, CLASS_LABELS_PT, CLASS_ORDER,
    FAF_LABELS, FCVC_LABELS, GENDER_OPTIONS, MTRANS_OPTIONS, NCP_LABELS,
    TUE_LABELS, YESNO_OPTIONS, get_recommendation, inject_css, load_model_a,
    load_model_b, load_target_encoder, page_header,
)

st.set_page_config(page_title='Predição | Obesidade', page_icon='🎯', layout='wide')
inject_css()

page_header('Predição de Nível de Peso',
            'Preencha os dados do paciente para obter a estimativa')

# Seleção do modelo
model_choice = st.radio(
    'Modelo a utilizar:',
    options=['Triagem comportamental (Modelo B)', 'Diagnóstico (Modelo A)'],
    horizontal=True,
    help='O Modelo B prevê a partir de comportamento (sem peso/IMC). '
         'O Modelo A inclui peso e IMC (alta acurácia, mas valor clínico limitado).',
)
use_model_a = model_choice.startswith('Diagnóstico')

st.markdown('---')

# -----------------------------------------------------------------------------
# Formulário
# -----------------------------------------------------------------------------
with st.form('prediction_form'):
    st.markdown('#### Dados demográficos e antropométricos')
    c1, c2, c3 = st.columns(3)
    with c1:
        gender = st.selectbox('Gênero', list(GENDER_OPTIONS))
        age = st.number_input('Idade (anos)', min_value=14, max_value=100, value=30, step=1)
    with c2:
        height = st.number_input('Altura (m)', min_value=1.30, max_value=2.20,
                                 value=1.70, step=0.01, format='%.2f')
    with c3:
        weight = st.number_input('Peso (kg)', min_value=30.0, max_value=250.0,
                                 value=70.0, step=0.5, format='%.1f')
        if use_model_a:
            st.caption('Atenção: Peso usado apenas pelo Modelo A')

    st.markdown('#### Histórico e hábitos alimentares')
    c4, c5, c6 = st.columns(3)
    with c4:
        family = st.selectbox('Histórico familiar de excesso de peso', list(YESNO_OPTIONS))
        favc = st.selectbox('Consome alimentos muito calóricos com frequência?',
                            list(YESNO_OPTIONS))
    with c5:
        fcvc = st.select_slider('Frequência de consumo de vegetais',
                               options=[1, 2, 3],
                               format_func=lambda x: FCVC_LABELS[x], value=2)
        ncp = st.select_slider('Número de refeições principais/dia',
                              options=[1, 2, 3, 4],
                              format_func=lambda x: NCP_LABELS[x], value=3)
    with c6:
        caec = st.selectbox('Consumo de alimentos entre refeições',
                           list(CAEC_CALC_OPTIONS))
        scc = st.selectbox('Monitora calorias ingeridas?', list(YESNO_OPTIONS))

    st.markdown('#### Estilo de vida')
    c7, c8, c9 = st.columns(3)
    with c7:
        ch2o = st.select_slider('Consumo diário de água',
                               options=[1, 2, 3],
                               format_func=lambda x: CH2O_LABELS[x], value=2)
        smoke = st.selectbox('Fuma?', list(YESNO_OPTIONS))
    with c8:
        faf = st.select_slider('Frequência de atividade física',
                              options=[0, 1, 2, 3],
                              format_func=lambda x: FAF_LABELS[x], value=1)
        tue = st.select_slider('Tempo em dispositivos eletrônicos',
                              options=[0, 1, 2],
                              format_func=lambda x: TUE_LABELS[x], value=1)
    with c9:
        calc = st.selectbox('Consumo de álcool', list(CAEC_CALC_OPTIONS))
        mtrans = st.selectbox('Meio de transporte habitual', list(MTRANS_OPTIONS))

    submitted = st.form_submit_button('🔍 Prever nível de peso', width='stretch')

# -----------------------------------------------------------------------------
# Predição
# -----------------------------------------------------------------------------
if submitted:
    # Monta DataFrame raw com EXATAMENTE o schema do treino
    raw_input = pd.DataFrame([{
        'Gender':         GENDER_OPTIONS[gender],
        'Age':            float(age),
        'Height':         float(height),
        'Weight':         float(weight),
        'family_history': YESNO_OPTIONS[family],
        'FAVC':           YESNO_OPTIONS[favc],
        'FCVC':           float(fcvc),
        'NCP':            float(ncp),
        'CAEC':           CAEC_CALC_OPTIONS[caec],
        'SMOKE':          YESNO_OPTIONS[smoke],
        'CH2O':           float(ch2o),
        'SCC':            YESNO_OPTIONS[scc],
        'FAF':            float(faf),
        'TUE':            float(tue),
        'CALC':           CAEC_CALC_OPTIONS[calc],
        'MTRANS':         MTRANS_OPTIONS[mtrans],
    }])

    try:
        model = load_model_a() if use_model_a else load_model_b()
        target_encoder = load_target_encoder()

        pred_int = model.predict(raw_input)[0]
        pred_class = target_encoder.inverse_transform([pred_int])[0]
        proba = model.predict_proba(raw_input)[0]
    except FileNotFoundError:
        st.error(
            '❌ Modelo não encontrado. Treine os modelos primeiro executando os '
            'notebooks 02 e 03, que geram os artefatos em `models/artifacts/`.'
        )
        st.stop()
    except Exception as e:
        st.error(f'❌ Erro ao gerar predição: {e}')
        st.stop()

    st.markdown('---')
    st.markdown('## Resultado')

    color = CLASS_COLORS[pred_class]
    label_pt = CLASS_LABELS_PT[pred_class]
    confidence = proba[pred_int] * 100

    res_col1, res_col2 = st.columns([1, 1.3])
    with res_col1:
        st.markdown(
            f"<div style='background:{color}; color:white; padding:1.8rem; "
            f"border-radius:14px; text-align:center;'>"
            f"<p style='margin:0; opacity:0.9; font-size:0.95rem'>Classe predita</p>"
            f"<h2 style='color:white; margin:0.3rem 0; font-size:1.8rem'>{label_pt}</h2>"
            f"<p style='margin:0; opacity:0.9'>Confiança: {confidence:.1f}%</p></div>",
            unsafe_allow_html=True,
        )
        # IMC informativo (sempre calculável a partir do form)
        imc = weight / (height ** 2)
        st.markdown(
            f"<div class='info-box' style='margin-top:1rem'>"
            f"<b>IMC informado:</b> {imc:.1f} kg/m²<br>"
            f"<span style='font-size:0.85rem; color:#5a6c73'>"
            f"(Calculado a partir de peso e altura; mostrado apenas para referência)"
            f"</span></div>",
            unsafe_allow_html=True,
        )

    with res_col2:
        # Gráfico de probabilidades por classe
        proba_df = pd.DataFrame({
            'classe': [CLASS_LABELS_PT[c] for c in target_encoder.classes_],
            'prob': proba * 100,
            'raw_class': list(target_encoder.classes_),
        })
        # Reordena em ordem clínica
        proba_df['order'] = proba_df['raw_class'].map(
            {c: i for i, c in enumerate(CLASS_ORDER)})
        proba_df = proba_df.sort_values('order')

        fig = go.Figure(go.Bar(
            x=proba_df['prob'], y=proba_df['classe'], orientation='h',
            marker_color=[CLASS_COLORS[c] for c in proba_df['raw_class']],
            text=[f'{p:.1f}%' for p in proba_df['prob']],
            textposition='auto',
        ))
        fig.update_layout(
            title='Probabilidade por classe',
            xaxis_title='Probabilidade (%)', yaxis_title='',
            height=340, margin=dict(l=10, r=10, t=40, b=10),
            plot_bgcolor='rgba(0,0,0,0)',
        )
        st.plotly_chart(fig, width='stretch')

    # Recomendação textual
    st.markdown('### Recomendação')
    st.markdown(
        f"<div class='info-box'>{get_recommendation(pred_class)}</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div class='disclaimer'>Resultado gerado por modelo estatístico de apoio.</div>",
        unsafe_allow_html=True,
    )
else:
    st.info('Preencha o formulário acima e clique em **Prever nível de peso**.')
