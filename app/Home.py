"""
app/Home.py
===========
Página inicial do app — contexto, modelos, instruções, links.

Executar localmente:
    streamlit run app/Home.py
"""

import sys
from pathlib import Path

import streamlit as st

# Garante que app_utils seja importável quando Home.py é o entry-point
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app_utils import GITHUB_URL, inject_css, load_metrics_json, page_header

st.set_page_config(
    page_title='Previsão de Obesidade | Apoio à Decisão Clínica',
    page_icon='🩺',
    layout='wide',
    initial_sidebar_state='expanded',
)

inject_css()

page_header(
    '🩺 Sistema Preditivo de Obesidade',
    'Ferramenta de apoio à decisão clínica baseada em Machine Learning',
)

# Métricas de destaque
try:
    metrics = load_metrics_json()
    acc_b = metrics['best_model_b']['test_accuracy']
    f1_b = metrics['best_model_b']['test_f1_macro']
    model_b_name = metrics['best_model_b']['model']
except Exception:
    acc_b, f1_b, model_b_name = None, None, 'XGBoost'

st.markdown('### ')
col1, col2, col3 = st.columns(3)
with col1:
    val = f'{acc_b*100:.1f}%' if acc_b else '—'
    st.markdown(
        f"<div class='metric-card'><h2>{val}</h2>"
        f"<p>Acurácia do modelo de triagem (teste)</p></div>",
        unsafe_allow_html=True,
    )
with col2:
    val = f'{f1_b:.3f}' if f1_b else '—'
    st.markdown(
        f"<div class='metric-card'><h2>{val}</h2>"
        f"<p>F1-macro (métrica de seleção)</p></div>",
        unsafe_allow_html=True,
    )
with col3:
    st.markdown(
        f"<div class='metric-card'><h2>7</h2>"
        f"<p>Classes de peso corporal preditas</p></div>",
        unsafe_allow_html=True,
    )

st.markdown('---')

# Contexto clínico
st.markdown('## O problema')
st.markdown(
    """
    A obesidade é uma condição médica caracterizada pelo acúmulo excessivo de gordura
    corporal, com causas multifatoriais (genéticas, ambientais e comportamentais). Sua
    prevalência cresce globalmente, atingindo todas as idades e classes sociais.

    Este sistema auxilia a equipe médica a estimar o nível de peso corporal de um paciente
    a partir de variáveis clínicas e comportamentais, oferecendo um ponto de partida
    objetivo para a tomada de decisão.
    """
)

# Descrição dos modelos
st.markdown('## Os dois modelos')
col_a, col_b = st.columns(2)
with col_a:
    st.markdown(
        """
        #### Modelo A — Diagnóstico
        Utiliza **peso e IMC** entre as variáveis. Atinge altíssima precisão
        (~97-100%), mas isso é esperado: as classes de obesidade são *definidas*
        pelo IMC. Serve como **linha base de sanidade**, não como ferramenta de valor
        clínico — o médico já calcula o IMC diretamente.
        """
    )
with col_b:
    st.markdown(
        """
        #### Modelo B — Triagem comportamental
        **Exclui peso e IMC.** Prediz o risco a partir de hábitos alimentares,
        atividade física, histórico familiar e estilo de vida — **antes da balança**.
        É o modelo de maior valor clínico, pois permite **ação preventiva**.
        É o modelo usado por padrão na página de Predição.
        """
    )

st.markdown(
    "<div class='info-box'>💡 <b>Por que dois modelos?</b> A separação evidencia "
    "que o sinal preditivo do Modelo B vem do comportamento, não da reprodução trivial "
    "da fórmula do IMC. Essa é a contribuição científica do projeto.</div>",
    unsafe_allow_html=True,
)

# Instruções de uso
st.markdown('## Como usar')
st.markdown(
    """
    1. **Predição** — preencha o formulário clínico do paciente e obtenha a estimativa
       de nível de peso, com probabilidades por classe e recomendação textual.
    2. **Dashboard** — explore os insights epidemiológicos do estudo: distribuição de
       classes, fatores de risco, hábitos e importância das variáveis.
    3. **Sobre o modelo** — consulte métricas, matriz de confusão, limitações e o
       disclaimer médico.

    Use o menu lateral à esquerda para navegar entre as páginas.
    """
)

# Links
st.markdown('## Recursos')
st.markdown(f"- **Repositório GitHub:** [{GITHUB_URL}]({GITHUB_URL})")

st.markdown('---')
st.caption('Tech Challenge — Pós-Tech | Modelo de apoio à decisão clínica para previsão de obesidade.')
