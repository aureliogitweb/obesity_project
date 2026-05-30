"""
src/utils/config.py
===================
Configurações centralizadas do projeto Tech Challenge de previsão de obesidade.

Esta é a ÚNICA fonte de verdade para:
- paths do projeto (resolvidos a partir da raiz, independente do CWD)
- random state global
- listas de colunas por tipo (binárias, ordinais, contínuas, etc.)
- mapeamentos determinísticos (binary, ordinal-with-order, target)
- ordem clínica das classes do target

Qualquer outro módulo que precise dessas constantes IMPORTA daqui — nunca redefine.
Isso garante que mudanças (ex: adicionar uma feature) propaguem em um único lugar.
"""

from pathlib import Path

# -----------------------------------------------------------------------------
# Reprodutibilidade
# -----------------------------------------------------------------------------
RANDOM_STATE = 42

# -----------------------------------------------------------------------------
# Paths do projeto (resolvidos a partir da localização do arquivo)
# -----------------------------------------------------------------------------
# src/utils/config.py → sobe 2 níveis → raiz do projeto
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
RAW_DATA_PATH = RAW_DATA_DIR / 'obesity.csv'

MODELS_DIR = PROJECT_ROOT / 'models'
ARTIFACTS_DIR = MODELS_DIR / 'artifacts'

# -----------------------------------------------------------------------------
# Target
# -----------------------------------------------------------------------------
TARGET_COL = 'Obesity'

# Ordem clínica natural — do mais magro ao mais obeso.
# Usada para encoding ordinal do target (LabelEncoder default seria alfabético
# e quebraria a ordem clínica — não usamos LabelEncoder por isso).
CLASS_ORDER = [
    'Insufficient_Weight',
    'Normal_Weight',
    'Overweight_Level_I',
    'Overweight_Level_II',
    'Obesity_Type_I',
    'Obesity_Type_II',
    'Obesity_Type_III',
]
N_CLASSES = len(CLASS_ORDER)

# -----------------------------------------------------------------------------
# Schema esperado das colunas brutas (para validação)
# -----------------------------------------------------------------------------
EXPECTED_COLUMNS = [
    'Gender', 'Age', 'Height', 'Weight',
    'family_history', 'FAVC', 'FCVC', 'NCP',
    'CAEC', 'SMOKE', 'CH2O', 'SCC',
    'FAF', 'TUE', 'CALC', 'MTRANS',
    'Obesity',
]

# -----------------------------------------------------------------------------
# Listas de colunas por tipo (organizam toda a pipeline)
# -----------------------------------------------------------------------------

# Contínuas "puras" — recebem StandardScaler se modelo for linear (LogReg).
# Weight só entra no Modelo A. IMC é engineered, também só no Modelo A.
CONTINUOUS_COLS_BASE = ['Age', 'Height']
CONTINUOUS_COLS_MODEL_A = CONTINUOUS_COLS_BASE + ['Weight', 'IMC']
CONTINUOUS_COLS_MODEL_B = CONTINUOUS_COLS_BASE  # Height fica (sozinha não vaza target)

# Ordinais numéricas (após arredondamento) — passthrough.
# A EDA confirmou que arredondar não destrói sinal (ΔF1 = +0.005 a favor do contínuo,
# dentro do desvio dos folds) e garante coerência treino-inferência.
NUMERIC_ORDINAL_COLS = ['FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']

# Binárias yes/no e Gender — viram 0/1 no cleaning.
BINARY_COLS = ['Gender', 'family_history', 'FAVC', 'SMOKE', 'SCC']

# Ordinais string com ordem natural — mapeadas para 0..3 no cleaning.
# Tratamos como ordinais (não one-hot) porque a ordem importa para o modelo.
ORDINAL_STRING_COLS = ['CAEC', 'CALC']

# Nominal puro — sem ordem, vai pra OneHotEncoder dentro do ColumnTransformer.
NOMINAL_COLS = ['MTRANS']

# -----------------------------------------------------------------------------
# Mapeamentos determinísticos (não dependem de fit — são universais)
# -----------------------------------------------------------------------------
BINARY_MAP = {
    'yes': 1, 'no': 0,
    'Male': 1, 'Female': 0,
}

# CAEC e CALC têm a mesma escala ordinal — um mapa serve para ambas.
CAEC_CALC_MAP = {
    'no': 0,
    'Sometimes': 1,
    'Frequently': 2,
    'Always': 3,
}

# -----------------------------------------------------------------------------
# Split sizes (estratificados)
# -----------------------------------------------------------------------------
TEST_SIZE = 0.15        # 15% para teste final (nunca tocado até o fim)
VAL_SIZE = 0.15         # 15% para validação durante tuning
# Treino = 70%

# -----------------------------------------------------------------------------
# Nomes dos artefatos persistidos
# -----------------------------------------------------------------------------
TARGET_ENCODER_FILENAME = 'target_encoder.joblib'
PIPELINE_MODEL_A_FILENAME = 'pipeline_model_a.joblib'
PIPELINE_MODEL_B_FILENAME = 'pipeline_model_b.joblib'

# Artefatos de treino (Etapa 4)
SPLITS_FILENAME = 'splits.joblib'                       # em data/processed/
BEST_MODEL_FILENAME = 'best_pipeline.joblib'            # melhor pipeline completa
METRICS_FILENAME = 'metrics_comparison.csv'             # tabela consolidada
METRICS_JSON_FILENAME = 'metrics_detailed.json'         # métricas detalhadas

# -----------------------------------------------------------------------------
# Configuração de treino / CV
# -----------------------------------------------------------------------------
CV_FOLDS = 5
SCORING_PRIMARY = 'f1_macro'        # métrica de seleção
N_ITER_RANDOM_SEARCH = 50           # teto de iterações do RandomizedSearchCV

# Classes clinicamente críticas — falso negativo aqui é o erro mais caro
CRITICAL_CLASSES = ['Obesity_Type_II', 'Obesity_Type_III']


def ensure_dirs() -> None:
    """Cria diretórios necessários se não existirem. Idempotente."""
    for d in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, ARTIFACTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
