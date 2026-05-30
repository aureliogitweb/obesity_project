# Tech Challenge — Previsão de Obesidade

Projeto de Machine Learning para auxiliar a equipe médica no diagnóstico de obesidade a partir de variáveis clínicas e comportamentais.

Estrutura do projeto
project/
├── data/
│   ├── raw/
│   │   └── obesity.csv               ← input (coloque aqui antes de rodar)
│   └── processed/
│       └── splits.joblib             ← splits estratificados (gerado pelo NB02)
├── models/
│   └── artifacts/
│       └── target\_encoder.joblib     ← gerado pelo NB02
├── notebooks/
│   ├── 01\_eda.ipynb                  ← EDA com ablação contínuo vs. arredondado
│   ├── 02\_preprocessing.ipynb        ← orquestração da pipeline
│   └── 03\_model\_training.ipynb       ← treino, comparação e avaliação
├── src/
│   ├── data/
│   │   ├── loader.py                 ← load\_raw\_data, validate\_schema
│   │   └── splits.py                 ← stratified\_split → DataSplits
│   ├── features/
│   │   ├── cleaning.py               ← dedup, round, map (determinístico)
│   │   ├── build\_features.py         ← add\_bmi (apenas Modelo A)
│   │   ├── pipeline.py               ← build\_pipeline (sklearn Pipeline)
│   │   └── target\_encoder.py         ← OrdinalTargetEncoder
│   ├── models/
│   │   ├── registry.py               ← estimators + espaço de busca XGB
│   │   ├── train.py                  ← cross\_validate, tune\_xgboost
│   │   ├── evaluate.py               ← métricas, confusion matrix, feature importance
│   │   └── persistence.py            ← save/load modelos e métricas
│   └── utils/
│       └── config.py                 ← paths, constantes, listas de colunas
├── requirements.txt
└── README.md

\---

## Como rodar

### 1\. Setup

```bash
pip install -r requirements.txt
```

### 2\. Dados

Coloque `obesity.csv` em `data/raw/`. O `loader.py` resolve esse caminho automaticamente a partir da raiz do projeto, então o CWD ao rodar não importa.

### 3\. Execução

```bash
# A partir da raiz do projeto
jupyter notebook notebooks/02\_preprocessing.ipynb
```

O notebook 02 importa de `src/` via `sys.path` (configurado na primeira célula). Para executar via terminal:

```bash
jupyter nbconvert --to notebook --execute notebooks/02\_preprocessing.ipynb
```

\---

## Decisões de arquitetura

### Modelos A e B

|Modelo|Features|Acurácia esperada|Valor clínico|
|-|-|-|-|
|**A** (Diagnóstico)|Todas, inclui Weight + IMC|\~97-98%|Baixo — é um calculador de IMC. Existe como baseline de sanidade.|
|**B** (Triagem comportamental)|Exclui Weight e IMC|\~78-85%|**Alto** — prediz risco a partir de comportamento, antes da balança.|

**Por quê dois modelos?** A EDA mostrou que um classificador trivial baseado apenas em IMC (sem ML) atinge 91.95% de acurácia no dataset. Modelo A herda essa "trivialidade". Modelo B é onde está a entrega científica real do projeto.

### Pipeline sklearn

```
raw DataFrame
    │
    ├─► \[prepare]        FunctionTransformer
    │     - arredondamento de Age + ordinais SMOTE-interpoladas
    │     - mapeamento binário (yes/no → 0/1)
    │     - mapeamento ordinal (CAEC, CALC → 0..3)
    │
    ├─► \[add\_bmi]        FunctionTransformer (somente Modelo A)
    │
    ├─► \[preprocessor]   ColumnTransformer
    │     - continuas    → StandardScaler (opcional, para LogReg)
    │     - binárias     → passthrough (já 0/1)
    │     - ordinais     → passthrough (já 0..3)
    │     - MTRANS       → OneHotEncoder(drop\_first=True)
    │
    └─► \[model]          estimator (opcional, próxima etapa)
```

**Vantagem:** o app Streamlit envia raw input do formulário, a pipeline aplica tudo internamente. Zero divergência entre treino e inferência.

