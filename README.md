# 🩺 SIADO: Sistema de Apoio ao Diagnóstico de Obesidade

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://tech-challenge-siado-fiap.streamlit.app/)

## 🚀 Sobre o Projeto

**SIADO** é uma solução de Machine Learning de ponta a ponta, desenvolvida como parte do Tech Challenge da Pós-Graduação em Data Analytics da FIAP. O sistema utiliza um modelo preditivo (XGBoost com 97.8% de acurácia) para auxiliar profissionais de saúde a diagnosticar o nível de obesidade de um paciente com base em seus dados demográficos e hábitos de vida.

A principal inovação do projeto é a integração de **IA Explicável (XAI)** com a biblioteca SHAP, que não apenas prevê o diagnóstico, mas também revela quais fatores mais contribuíram para essa decisão, transformando uma "caixa preta" em uma ferramenta de insight clínico.

### Entregáveis do Projeto
*   **[Aplicação Preditiva (Streamlit)](URL_DA_SUA_APP_STREAMLIT_AQUI)**: Ferramenta interativa para diagnóstico de pacientes individuais.
*   **[Painel Analítico (Tableau)](URL_DO_SEU_DASHBOARD_TABLEAU_AQUI)**: Dashboard com insights estratégicos sobre os fatores de risco na população estudada.
*   **[Vídeo de Apresentação](URL_DO_SEU_VIDEO_AQUI)**: Demonstração da solução e seus benefícios.

---

## 🛠️ Tecnologias Utilizadas

- **Linguagem:** Python 3.11
- **Análise de Dados:** Pandas, NumPy
- **Visualização:** Matplotlib, Seaborn, Plotly
- **Machine Learning:** Scikit-learn, XGBoost
- **IA Explicável (XAI):** SHAP
- **Aplicação Web:** Streamlit
- **Dashboard:** Tableau Public
- **Containerização:** Docker
- **Versionamento:** Git & GitHub

---

## 📂 Estrutura do Repositório

O projeto segue uma estrutura modular para garantir organização e escalabilidade:

├── app/
│ └── app.py # Código da aplicação Streamlit
├── data/
│ └── raw/obesity.csv # Dados brutos
├── models/
│ ├── pipeline_obesidade.joblib # Pipeline de pré-processamento + modelo
│ └── label_encoder.joblib # Encoder dos rótulos
├── notebooks/
│ ├── 1-EDA.ipynb # Análise Exploratória de Dados
│ └── 2-Model-Training.ipynb # Treinamento e avaliação do modelo
├── .gitignore
├── Dockerfile # Configuração para containerizar a aplicação
├── README.md # Este arquivo
└── requirements.txt # Dependências do projeto



---

## 🏁 Como Executar Localmente

### Pré-requisitos
- Python 3.11+
- Git

### Opção 1: Usando um Ambiente Virtual (Recomendado)

1.  **Clone o repositório:**
    ```bash
    git clone https://github.com/rosimotasp/postech.git
    cd postech
    ```
2.  **Crie e ative um ambiente virtual:**
    ```bash
    # Windows
    python -m venv venv
    .\venv\Scripts\activate

    # macOS / Linux
    python3 -m venv venv
    source venv/bin/activate
    ```
3.  **Instale as dependências:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Execute a aplicação Streamlit:**
    ```bash
    streamlit run app/app.py
    ```

### Opção 2: Usando Docker

1.  **Clone o repositório.**
2.  **Construa a imagem Docker:**
    ```bash
    docker build -t siado-app .
    ```
3.  **Execute o contêiner:**
    ```bash
    docker run -p 8501:8501 siado-app
    ```
Acesse a aplicação em `http://localhost:8501`.

---

## 👥 Equipe de Desenvolvimento

- **Rosicléia Cavalcante Mota**
- **Guillermo Jesus Camahuali Privat**
- **Kelly Priscilla Matos Campos**
