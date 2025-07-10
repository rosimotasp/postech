# =============================================================================
# Dockerfile para a Aplicação SIADO (Streamlit)
# =============================================================================

# --- Estágio 1: Base ---
# Usamos uma imagem oficial do Python. Usar uma versão específica (ex: 3.11)
# garante a reprodutibilidade. A versão "slim" é mais leve.
FROM python:3.11-slim

# --- Configurações do Ambiente ---
# Define o diretório de trabalho dentro do contêiner.
# Todos os comandos a seguir serão executados a partir daqui.
WORKDIR /app

# Evita que o Python gere arquivos .pyc, mantendo o contêiner limpo.
ENV PYTHONDONTWRITEBYTECODE 1
# Garante que a saída do Python seja exibida imediatamente nos logs.
ENV PYTHONUNBUFFERED 1

# --- Cópia dos Arquivos do Projeto ---
# Copia apenas o arquivo de dependências primeiro.
# Isso aproveita o cache do Docker: se o requirements.txt não mudar,
# a longa etapa de instalação não será refeita a cada build.
COPY requirements.txt .

# --- Instalação das Dependências ---
# Atualiza o pip e instala as bibliotecas do requirements.txt.
# --no-cache-dir reduz o tamanho final da imagem.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copia todo o resto do projeto para dentro do diretório de trabalho /app.
COPY . .

# --- Configuração de Rede e Execução ---
# Expõe a porta 8501, que é a porta padrão do Streamlit.
EXPOSE 8501

# Define o "ponto de entrada" de saúde para o contêiner.
# O Docker pode verificar se a aplicação está rodando nesta porta.
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# O comando que será executado quando o contêiner iniciar.
# --server.port=8501: Define a porta.
# --server.address=0.0.0.0: Permite que a aplicação seja acessada de fora do contêiner.
ENTRYPOINT ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]