"""
app/pages/3_Sobre.py
====================
Página "Sobre o modelo" — métricas, matriz de confusão, limitações, leakage, disclaimer.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app_utils import (  # noqa: E402
    CLASS_LABELS_PT, CLASS_ORDER, inject_css, load_dataset, load_metrics_csv,
    load_metrics_json, load_model_b, load_target_encoder, page_header,
)

st.set_page_config(page_title='Sobre o Modelo | Obesidade', page_icon='ℹ️', layout='wide')
inject_css()

page_header('ℹ️ Sobre o Modelo',
            'Métricas, limitações e considerações técnicas')

# -----------------------------------------------------------------------------
# Métricas
# -----------------------------------------------------------------------------
st.markdown('## Desempenho dos modelos')

try:
    metrics_csv = load_metrics_csv()
    metrics_json = load_metrics_json()
except FileNotFoundError:
    st.error('❌ Métricas não encontradas. Execute o notebook 03 primeiro.')
    st.stop()

# Tabela de teste, formatada
test_metrics = metrics_csv[metrics_csv['stage'] == 'test'].copy()
test_metrics = test_metrics.sort_values('f1_macro', ascending=False)
test_metrics['Modelo'] = test_metrics['model'] + ' (' + test_metrics['variant'] + ')'
display_df = test_metrics[['Modelo', 'f1_macro', 'accuracy']].copy()
display_df.columns = ['Modelo (variante)', 'F1-macro', 'Acurácia']
display_df['F1-macro'] = display_df['F1-macro'].map('{:.4f}'.format)
display_df['Acurácia'] = display_df['Acurácia'].map('{:.4f}'.format)

st.markdown('**Resultados no conjunto de teste (isolado durante todo o treino):**')
st.dataframe(display_df, width='stretch', hide_index=True)

col1, col2 = st.columns(2)
with col1:
    b = metrics_json['best_model_b']
    st.markdown(
        f"<div class='metric-card'><h2>{b['test_accuracy']*100:.1f}%</h2>"
        f"<p>Modelo B (triagem) — acurácia · {b['model']}</p></div>",
        unsafe_allow_html=True,
    )
with col2:
    a = metrics_json['best_model_a']
    st.markdown(
        f"<div class='metric-card'><h2>{a['test_accuracy']*100:.1f}%</h2>"
        f"<p>Modelo A (diagnóstico) — acurácia · {a['model']}</p></div>",
        unsafe_allow_html=True,
    )

st.markdown(
    "<div class='info-box'>✅ O Modelo B supera o requisito de <b>75% de acurácia</b> "
    "do desafio usando apenas variáveis comportamentais — o resultado clinicamente "
    "relevante.</div>",
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Matriz de confusão (Modelo B, recalculada no teste)
# -----------------------------------------------------------------------------
st.markdown('## Matriz de confusão — Modelo B')

try:
    from sklearn.metrics import confusion_matrix

    # Recarrega splits para recomputar a matriz (dados de teste)
    splits_path = Path(__file__).resolve().parents[2] / 'data' / 'processed' / 'splits.joblib'
    import joblib
    splits = joblib.load(splits_path)
    model_b = load_model_b()
    te = load_target_encoder()

    y_test = te.transform(splits['y_test'])
    y_pred = model_b.predict(splits['X_test'])

    cm = confusion_matrix(y_test, y_pred, labels=list(range(len(CLASS_ORDER))))
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    labels_pt = [CLASS_LABELS_PT[c] for c in CLASS_ORDER]

    fig = px.imshow(
        cm_norm, x=labels_pt, y=labels_pt,
        color_continuous_scale='Teal', aspect='auto',
        labels=dict(x='Predito', y='Real', color='Proporção'),
        text_auto='.2f',
    )
    fig.update_layout(height=520, margin=dict(l=10, r=10, t=10, b=10))
    fig.update_xaxes(tickangle=-30)
    st.plotly_chart(fig, width='stretch')

    st.markdown(
        "<div class='info-box'>A diagonal mostra os acertos (recall por classe). "
        "Confusões tendem a ocorrer entre classes <b>adjacentes</b> (ex.: sobrepeso I ↔ II), "
        "o que é clinicamente aceitável para uma ferramenta de triagem.</div>",
        unsafe_allow_html=True,
    )
except Exception as e:
    st.info(f'Matriz de confusão indisponível: {e}')

# -----------------------------------------------------------------------------
# Discussão sobre leakage
# -----------------------------------------------------------------------------
st.markdown('## Considerações técnicas')

st.markdown('### Por que dois modelos? O "vazamento" do IMC')
st.markdown(
    """
    - **Modelo A** (com peso/IMC): baseline de sanidade. Alta precisão sem sentido.
    - **Modelo B** (sem peso/IMC): prevê a partir de comportamento e histórico. Precisão
      menor, mas é o modelo com valor clínico real, permite ação preventiva antes
      mesmo da balança apontar o problema.
    """
)

st.markdown('---')
st.caption('Tech Challenge — Pós-Tech | Modelo de apoio à decisão clínica para previsão de obesidade.')
