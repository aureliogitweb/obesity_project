"""
app/pages/2_Dashboard.py
========================
Dashboard analítico — insights epidemiológicos do estudo sobre obesidade.

Todos os gráficos são Plotly (interativos). Filtros na sidebar reagem em todas
as visualizações. Inclui feature importance do modelo treinado.
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app_utils import (  # noqa: E402
    CLASS_COLORS, CLASS_LABELS_PT, CLASS_ORDER, inject_css, load_dataset,
    load_model_b, page_header,
)

st.set_page_config(page_title='Dashboard | Obesidade', page_icon='📊', layout='wide')
inject_css()

page_header('📊 Dashboard Analítico',
            'Insights epidemiológicos do estudo sobre obesidade')

# -----------------------------------------------------------------------------
# Carregamento + filtros
# -----------------------------------------------------------------------------
try:
    df = load_dataset()
except FileNotFoundError:
    st.error('Dataset não encontrado em `data/raw/obesity.csv`.')
    st.stop()

st.sidebar.markdown('### Filtros')
gender_filter = st.sidebar.multiselect(
    'Gênero', options=['Female', 'Male'], default=['Female', 'Male'],
    format_func=lambda x: 'Feminino' if x == 'Female' else 'Masculino',
)
age_range = st.sidebar.slider(
    'Faixa etária', int(df['Age'].min()), int(df['Age'].max()),
    (int(df['Age'].min()), int(df['Age'].max())),
)
family_filter = st.sidebar.multiselect(
    'Histórico familiar', options=['yes', 'no'], default=['yes', 'no'],
    format_func=lambda x: 'Com histórico' if x == 'yes' else 'Sem histórico',
)

# Aplica filtros
mask = (
    df['Gender'].isin(gender_filter)
    & df['Age'].between(age_range[0], age_range[1])
    & df['family_history'].isin(family_filter)
)
dff = df[mask]

if len(dff) == 0:
    st.warning('Nenhum registro com os filtros selecionados.')
    st.stop()

# -----------------------------------------------------------------------------
# KPIs
# -----------------------------------------------------------------------------
k1, k2, k3, k4 = st.columns(4)
k1.metric('Registros', f'{len(dff):,}')
k2.metric('IMC médio', f"{dff['IMC'].mean():.1f}")
k3.metric('Idade média', f"{dff['Age'].mean():.0f} anos")
obese_pct = dff['Obesity'].isin(
    ['Obesity_Type_I', 'Obesity_Type_II', 'Obesity_Type_III']).mean() * 100
k4.metric('% Obesidade', f'{obese_pct:.0f}%')

st.markdown('---')

# -----------------------------------------------------------------------------
# Linha 1: distribuição de classes + IMC por classe
# -----------------------------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    st.markdown('#### Distribuição das classes')
    counts = dff['Obesity'].value_counts().reindex(CLASS_ORDER).fillna(0)
    fig = go.Figure(go.Bar(
        x=[CLASS_LABELS_PT[c] for c in CLASS_ORDER],
        y=counts.values,
        marker_color=[CLASS_COLORS[c] for c in CLASS_ORDER],
        text=counts.values.astype(int), textposition='auto',
    ))
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10),
                      plot_bgcolor='rgba(0,0,0,0)', yaxis_title='Registros')
    fig.update_xaxes(tickangle=-30)
    st.plotly_chart(fig, width='stretch')

with col2:
    st.markdown('#### IMC por classe')
    fig = px.box(dff, x='Obesity', y='IMC', category_orders={'Obesity': CLASS_ORDER},
                 color='Obesity', color_discrete_map=CLASS_COLORS)
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10),
                      plot_bgcolor='rgba(0,0,0,0)', showlegend=False,
                      xaxis_title='')
    fig.update_xaxes(ticktext=[CLASS_LABELS_PT[c] for c in CLASS_ORDER],
                     tickvals=CLASS_ORDER, tickangle=-30)
    st.plotly_chart(fig, width='stretch')

st.markdown(
    "<div class='info-box'>💡 O IMC separa as classes de forma quase perfeita — "
    "esperado, já que as classes são definidas pelo IMC. Por isso o Modelo B "
    "exclui peso e IMC, buscando sinal comportamental.</div>",
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Linha 2: histórico familiar + atividade física
# -----------------------------------------------------------------------------
col3, col4 = st.columns(2)

with col3:
    st.markdown('#### Histórico familiar × obesidade')
    ct = pd.crosstab(dff['family_history'], dff['Obesity'], normalize='index') * 100
    ct = ct.reindex(columns=CLASS_ORDER).fillna(0)
    fig = go.Figure()
    for cls in CLASS_ORDER:
        fig.add_trace(go.Bar(
            name=CLASS_LABELS_PT[cls], x=['Com histórico', 'Sem histórico'],
            y=[ct.loc['yes', cls] if 'yes' in ct.index else 0,
               ct.loc['no', cls] if 'no' in ct.index else 0],
            marker_color=CLASS_COLORS[cls],
        ))
    fig.update_layout(barmode='stack', height=380,
                      margin=dict(l=10, r=10, t=10, b=10),
                      plot_bgcolor='rgba(0,0,0,0)', yaxis_title='%',
                      legend=dict(font=dict(size=9)))
    st.plotly_chart(fig, width='stretch')

with col4:
    st.markdown('#### Atividade física × obesidade')
    dff_faf = dff.copy()
    dff_faf['FAF_round'] = dff_faf['FAF'].round().astype(int)
    faf_labels = {0: 'Nenhuma', 1: '1-2x/sem', 2: '3-4x/sem', 3: '5x+/sem'}
    dff_faf['FAF_label'] = dff_faf['FAF_round'].map(faf_labels)
    ct = pd.crosstab(dff_faf['FAF_label'], dff_faf['Obesity'], normalize='index') * 100
    ct = ct.reindex(columns=CLASS_ORDER).fillna(0)
    ct = ct.reindex(index=[v for v in faf_labels.values() if v in ct.index])
    fig = go.Figure()
    for cls in CLASS_ORDER:
        fig.add_trace(go.Bar(
            name=CLASS_LABELS_PT[cls], x=ct.index, y=ct[cls],
            marker_color=CLASS_COLORS[cls],
        ))
    fig.update_layout(barmode='stack', height=380,
                      margin=dict(l=10, r=10, t=10, b=10),
                      plot_bgcolor='rgba(0,0,0,0)', yaxis_title='%',
                      legend=dict(font=dict(size=9)))
    st.plotly_chart(fig, width='stretch')

# -----------------------------------------------------------------------------
# Linha 3: hábitos alimentares + transporte
# -----------------------------------------------------------------------------
col5, col6 = st.columns(2)

with col5:
    st.markdown('#### Alimento calórico (FAVC) × obesidade')
    ct = pd.crosstab(dff['FAVC'], dff['Obesity'], normalize='index') * 100
    ct = ct.reindex(columns=CLASS_ORDER).fillna(0)
    fig = go.Figure()
    labels_map = {'yes': 'Consome', 'no': 'Não consome'}
    for cls in CLASS_ORDER:
        fig.add_trace(go.Bar(
            name=CLASS_LABELS_PT[cls],
            x=[labels_map.get(i, i) for i in ct.index], y=ct[cls],
            marker_color=CLASS_COLORS[cls],
        ))
    fig.update_layout(barmode='stack', height=380,
                      margin=dict(l=10, r=10, t=10, b=10),
                      plot_bgcolor='rgba(0,0,0,0)', yaxis_title='%',
                      legend=dict(font=dict(size=9)))
    st.plotly_chart(fig, width='stretch')

with col6:
    st.markdown('#### Meio de transporte × obesidade')
    mtrans_labels = {
        'Public_Transportation': 'Transp. público', 'Automobile': 'Automóvel',
        'Walking': 'Caminhada', 'Motorbike': 'Moto', 'Bike': 'Bicicleta',
    }
    ct = pd.crosstab(dff['MTRANS'], dff['Obesity'], normalize='index') * 100
    ct = ct.reindex(columns=CLASS_ORDER).fillna(0)
    fig = go.Figure()
    for cls in CLASS_ORDER:
        fig.add_trace(go.Bar(
            name=CLASS_LABELS_PT[cls],
            x=[mtrans_labels.get(i, i) for i in ct.index], y=ct[cls],
            marker_color=CLASS_COLORS[cls],
        ))
    fig.update_layout(barmode='stack', height=380,
                      margin=dict(l=10, r=10, t=10, b=10),
                      plot_bgcolor='rgba(0,0,0,0)', yaxis_title='%',
                      legend=dict(font=dict(size=9)))
    fig.update_xaxes(tickangle=-20)
    st.plotly_chart(fig, width='stretch')

# -----------------------------------------------------------------------------
# Feature importance do Modelo B
# -----------------------------------------------------------------------------
st.markdown('---')
st.markdown('#### Importância das variáveis — Modelo B (triagem comportamental)')

# Dicionário de tradução: nome técnico → rótulo amigável em português
# (afeta APENAS a exibição; o modelo continua usando os nomes originais)
FEATURE_LABELS_PT = {
    'Age':             'Idade',
    'Height':          'Altura',
    'Gender':          'Gênero',
    'family_history':  'Histórico familiar',
    'FAVC':            'Consumo de calorias',
    'FCVC':            'Consumo de vegetais',
    'NCP':             'Refeições por dia',
    'CAEC':            'Lanches entre refeições',
    'SMOKE':           'Fumante',
    'CH2O':            'Consumo de água',
    'SCC':             'Monitora calorias',
    'FAF':             'Atividade física',
    'TUE':             'Tempo em telas',
    'CALC':            'Consumo de álcool',
    # MTRANS, após one-hot, vira MTRANS_<categoria>
    'MTRANS_Automobile':            'Transporte: automóvel',
    'MTRANS_Bike':                  'Transporte: bicicleta',
    'MTRANS_Motorbike':             'Transporte: moto',
    'MTRANS_Public_Transportation': 'Transporte: público',
    'MTRANS_Walking':               'Transporte: caminhada',
}

try:
    model_b = load_model_b()
    xgb = model_b.named_steps['model']
    preprocessor = model_b.named_steps['preprocessor']
    feat_names = list(preprocessor.get_feature_names_out())
    importances = xgb.feature_importances_

    fi = (pd.DataFrame({'feature': feat_names, 'importance': importances})
          .sort_values('importance', ascending=True).tail(12))

    # Aplica a tradução para exibição (fallback no nome original se faltar mapa)
    fi['feature_pt'] = fi['feature'].map(FEATURE_LABELS_PT).fillna(fi['feature'])

    fig = go.Figure(go.Bar(
        x=fi['importance'], y=fi['feature_pt'], orientation='h',   # ← usa feature_pt
        marker_color='#0d7377',
    ))
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10),
                      plot_bgcolor='rgba(0,0,0,0)', xaxis_title='Importância (gain)')
    st.plotly_chart(fig, width='stretch')

    st.markdown(
        "<div class='info-box'>Estas são as características comportamentais e demográficas "
        "que mais influenciam a previsão do nível de peso, indicando oportunidades para intervenções preventivas."
        "</div>"
        "",
        unsafe_allow_html=True,
    )
except Exception as e:
    st.info(f'Feature importance indisponível: treine o Modelo B primeiro. ({e})')

st.markdown('---')
st.caption('Dados: estudo sobre obesidade.'
           'Dataset balanceado via SMOTE.')
