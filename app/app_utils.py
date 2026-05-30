"""
app/app_utils.py
================
"""

from pathlib import Path
import sys

import joblib
import pandas as pd
import streamlit as st

# -----------------------------------------------------------------------------
# Paths — resolvidos a partir deste arquivo (robusto no Streamlit Cloud)
# -----------------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent

# CRÍTICO: a pipeline serializada referencia funções de `src.features` (via
# FunctionTransformer). Para o joblib.load conseguir resolver esses imports,
# a raiz do projeto precisa estar no sys.path. Sem isto, o deserialize falha
# com ModuleNotFoundError: No module named 'src'.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ARTIFACTS_DIR = PROJECT_ROOT / 'models' / 'artifacts'
RAW_DATA_PATH = PROJECT_ROOT / 'data' / 'raw' / 'obesity.csv'

BEST_MODEL_PATH = ARTIFACTS_DIR / 'best_pipeline.joblib'          # Modelo B (triagem)
MODEL_A_PATH = ARTIFACTS_DIR / 'pipeline_model_a.joblib'          # Modelo A (diagnóstico)
TARGET_ENCODER_PATH = ARTIFACTS_DIR / 'target_encoder.joblib'
METRICS_CSV_PATH = ARTIFACTS_DIR / 'metrics_comparison.csv'
METRICS_JSON_PATH = ARTIFACTS_DIR / 'metrics_detailed.json'

GITHUB_URL = 'https://github.com/aureliogitweb/obesity_project'  # ← ajustar no deploy

# -----------------------------------------------------------------------------
# Constantes clínicas / UI
# -----------------------------------------------------------------------------
# Ordem clínica do mais magro ao mais obeso
CLASS_ORDER = [
    'Insufficient_Weight', 'Normal_Weight',
    'Overweight_Level_I', 'Overweight_Level_II',
    'Obesity_Type_I', 'Obesity_Type_II', 'Obesity_Type_III',
]

# Labels amigáveis em português para exibição
CLASS_LABELS_PT = {
    'Insufficient_Weight': 'Abaixo do peso',
    'Normal_Weight':       'Peso normal',
    'Overweight_Level_I':  'Sobrepeso I',
    'Overweight_Level_II': 'Sobrepeso II',
    'Obesity_Type_I':      'Obesidade I',
    'Obesity_Type_II':     'Obesidade II',
    'Obesity_Type_III':    'Obesidade III',
}

# Paleta progressiva (azul → verde → laranja → vermelho)
CLASS_COLORS = {
    'Insufficient_Weight': '#3b82f6',
    'Normal_Weight':       '#10b981',
    'Overweight_Level_I':  '#fbbf24',
    'Overweight_Level_II': '#f97316',
    'Obesity_Type_I':      '#ef4444',
    'Obesity_Type_II':     '#dc2626',
    'Obesity_Type_III':    '#7f1d1d',
}

# Opções dos dropdowns — devem casar com o domínio do treino
GENDER_OPTIONS = {'Feminino': 'Female', 'Masculino': 'Male'}
YESNO_OPTIONS = {'Sim': 'yes', 'Não': 'no'}
CAEC_CALC_OPTIONS = {
    'Não': 'no', 'Às vezes': 'Sometimes',
    'Frequentemente': 'Frequently', 'Sempre': 'Always',
}
MTRANS_OPTIONS = {
    'Transporte público': 'Public_Transportation',
    'Automóvel': 'Automobile',
    'Caminhada': 'Walking',
    'Moto': 'Motorbike',
    'Bicicleta': 'Bike',
}

# Escalas ordinais com descrição
FCVC_LABELS = {1: '1 — Raramente', 2: '2 — Às vezes', 3: '3 — Sempre'}
NCP_LABELS = {1: '1 refeição', 2: '2 refeições', 3: '3 refeições', 4: '4+ refeições'}
CH2O_LABELS = {1: '1 — Menos de 1L', 2: '2 — Entre 1 e 2L', 3: '3 — Mais de 2L'}
FAF_LABELS = {0: '0 — Nenhuma', 1: '1 — 1-2x/semana', 2: '2 — 3-4x/semana', 3: '3 — 5x+/semana'}
TUE_LABELS = {0: '0 — 0-2h/dia', 1: '1 — 3-5h/dia', 2: '2 — Mais de 5h/dia'}


# -----------------------------------------------------------------------------
# Carregamento cacheado
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model_b():
    """Carrega o Modelo B (triagem comportamental) — destaque do app."""
    return joblib.load(BEST_MODEL_PATH)


@st.cache_resource(show_spinner=False)
def load_model_a():
    """Carrega o Modelo A (diagnóstico, usa Weight + IMC)."""
    return joblib.load(MODEL_A_PATH)


@st.cache_resource(show_spinner=False)
def load_target_encoder():
    """Carrega o encoder do target (int ↔ nome de classe)."""
    return joblib.load(TARGET_ENCODER_PATH)


@st.cache_data(show_spinner=False)
def load_metrics_csv() -> pd.DataFrame:
    """Carrega a tabela consolidada de métricas."""
    return pd.read_csv(METRICS_CSV_PATH)


@st.cache_data(show_spinner=False)
def load_metrics_json() -> dict:
    """Carrega métricas detalhadas (recall por classe, best_params)."""
    import json
    with open(METRICS_JSON_PATH, encoding='utf-8') as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_dataset() -> pd.DataFrame:
    """Carrega o dataset bruto para o dashboard. Adiciona IMC para análise."""
    df = pd.read_csv(RAW_DATA_PATH)
    df['IMC'] = df['Weight'] / (df['Height'] ** 2)
    return df


# -----------------------------------------------------------------------------
# Helpers de UI
# -----------------------------------------------------------------------------
def inject_css() -> None:
    """Injeta CSS customizado para identidade visual clínica."""
    st.markdown(
        """
        <style>
        /* Tipografia e respiro */
        .main .block-container { padding-top: 2.5rem; max-width: 1100px; }
        h1, h2, h3 { color: #0d4f52; font-weight: 700; }

        /* Cards de destaque */
        .metric-card {
            background: linear-gradient(135deg, #0d7377 0%, #14a085 100%);
            color: white; padding: 1.4rem 1.6rem; border-radius: 14px;
            box-shadow: 0 4px 14px rgba(13,115,119,0.18);
        }
        .metric-card h2 { color: white; margin: 0; font-size: 2.1rem; }
        .metric-card p { margin: 0.2rem 0 0; opacity: 0.92; font-size: 0.9rem; }

        .info-box {
            background: #f0f4f5; border-left: 4px solid #0d7377;
            padding: 1rem 1.2rem; border-radius: 8px; margin: 0.8rem 0;
        }
        .disclaimer {
            background: #fef3f2; border-left: 4px solid #dc2626;
            padding: 1rem 1.2rem; border-radius: 8px; margin: 1rem 0;
            font-size: 0.9rem; color: #7f1d1d;
        }
        /* Botão primário mais sólido */
        .stButton > button {
            background: #0d7377; color: white; border: none;
            border-radius: 10px; padding: 0.6rem 1.4rem; font-weight: 600;
        }
        .stButton > button:hover { background: #0a5d60; color: white; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_header(title: str, subtitle: str = '') -> None:
    """Cabeçalho padrão de página."""
    st.title(title)
    if subtitle:
        st.markdown(f"<p style='color:#5a6c73; font-size:1.05rem; margin-top:-0.5rem'>"
                    f"{subtitle}</p>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Recomendação textual a partir da predição
# -----------------------------------------------------------------------------
RECOMMENDATIONS = {
    'Insufficient_Weight': (
        "Resultado sugere peso abaixo do ideal. Considerar avaliação nutricional "
        "para ganho de peso saudável e investigação de causas subjacentes."
    ),
    'Normal_Weight': (
        "Resultado sugere peso dentro da faixa normal. Reforçar manutenção de "
        "hábitos saudáveis de alimentação e atividade física."
    ),
    'Overweight_Level_I': (
        "Resultado sugere sobrepeso inicial. Momento oportuno para intervenção "
        "preventiva: revisão de dieta, aumento de atividade física e hidratação."
    ),
    'Overweight_Level_II': (
        "Resultado sugere sobrepeso. Recomenda-se acompanhamento nutricional e "
        "plano estruturado de atividade física."
    ),
    'Obesity_Type_I': (
        "Resultado sugere obesidade grau I. Indicado acompanhamento "
        "multiprofissional (nutrição, atividade física, avaliação clínica)."
    ),
    'Obesity_Type_II': (
        "Resultado sugere obesidade grau II. Recomenda-se avaliação clínica "
        "aprofundada e plano terapêutico individualizado."
    ),
    'Obesity_Type_III': (
        "Resultado sugere obesidade grau III (severa). Indicada avaliação clínica "
        "prioritária e possível abordagem multidisciplinar especializada."
    ),
}


def get_recommendation(class_name: str) -> str:
    """Retorna texto de recomendação para a classe predita."""
    return RECOMMENDATIONS.get(class_name, "Recomenda-se avaliação clínica individualizada.")
