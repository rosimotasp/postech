
import streamlit as st
import pandas as pd
import pickle
import requests

st.set_page_config(page_title="Previsão de Obesidade", layout="centered")

# Carregar modelo treinado do GitHub
url_modelo = 'https://github.com/rosimotasp/postech/raw/main/modelo_obesidade.pkl'
modelo = pickle.loads(requests.get(url_modelo).content)

st.title("🔍 Previsão de Nível de Obesidade")

# Interface do usuário - entradas em português
genero = st.selectbox("Gênero", ["Feminino", "Masculino"])
idade = st.slider("Idade", 10, 100, 25)
altura = st.number_input("Altura (m)", 1.3, 2.2, 1.70)
peso = st.number_input("Peso (kg)", 30.0, 200.0, 70.0)
historico_familiar = st.selectbox("Histórico familiar de obesidade", ["sim", "não"])
consumo_calorico = st.selectbox("Costuma consumir comida calórica com frequência?", ["sim", "não"])
consumo_vegetais = st.slider("Frequência de consumo de vegetais (1 = baixa, 3 = alta)", 1.0, 3.0, 2.0)
refeicoes_dia = st.slider("Número de refeições principais por dia", 1.0, 4.0, 3.0)
lanches = st.selectbox("Costuma lanchar entre as refeições?", ["não", "às vezes", "frequentemente", "sempre"])
fuma = st.selectbox("Fuma?", ["sim", "não"])
consumo_agua = st.slider("Consumo de água (litros por dia)", 1.0, 3.0, 2.0)
controle_calorias = st.selectbox("Monitora o consumo de calorias?", ["sim", "não"])
atividade_fisica = st.slider("Atividade física por semana (horas)", 0.0, 4.0, 2.0)
uso_tecnologia = st.slider("Tempo usando dispositivos por dia (horas)", 0.0, 3.0, 1.0)
consumo_alcool = st.selectbox("Consome bebida alcoólica?", ["não", "às vezes", "frequentemente", "sempre"])
meio_transporte = st.selectbox("Meio de transporte predominante", [
    "Transporte Público", "Caminhada", "Automóvel", "Motocicleta", "Bicicleta"
])

# Mapeamento reverso: português → inglês
mapa_valores = {
    "Feminino": "Female", "Masculino": "Male",
    "sim": "yes", "não": "no",
    "às vezes": "Sometimes", "frequentemente": "Frequently", "sempre": "Always",
    "Transporte Público": "Public_Transportation",
    "Caminhada": "Walking", "Automóvel": "Automobile",
    "Motocicleta": "Motorbike", "Bicicleta": "Bike"
}

# Botão de previsão
if st.button("Prever"):
    # Traduzir para o formato usado no modelo
    dados = pd.DataFrame({
        'Gender': [mapa_valores[genero]],
        'Age': [idade],
        'Height': [altura],
        'Weight': [peso],
        'family_history': [mapa_valores[historico_familiar]],
        'FAVC': [mapa_valores[consumo_calorico]],
        'FCVC': [consumo_vegetais],
        'NCP': [refeicoes_dia],
        'CAEC': [mapa_valores[lanches]],
        'SMOKE': [mapa_valores[fuma]],
        'CH2O': [consumo_agua],
        'SCC': [mapa_valores[controle_calorias]],
        'FAF': [atividade_fisica],
        'TUE': [uso_tecnologia],
        'CALC': [mapa_valores[consumo_alcool]],
        'MTRANS': [mapa_valores[meio_transporte]]
    })

    # Aplicar get_dummies e alinhar colunas
    dados_dummies = pd.get_dummies(dados)
    colunas_esperadas = modelo.feature_names_in_

    # Adicionar colunas ausentes com 0
    for col in colunas_esperadas:
        if col not in dados_dummies.columns:
            dados_dummies[col] = 0
    dados_dummies = dados_dummies[colunas_esperadas]

    # Previsão
    pred = modelo.predict(dados_dummies)[0]

    # Traduzir resultado final
    mapa_saida = {
        'Insufficient_Weight': 'Peso insuficiente',
        'Normal_Weight': 'Peso normal',
        'Overweight_Level_I': 'Sobrepeso - Nível I',
        'Overweight_Level_II': 'Sobrepeso - Nível II',
        'Obesity_Type_I': 'Obesidade Tipo I',
        'Obesity_Type_II': 'Obesidade Tipo II',
        'Obesity_Type_III': 'Obesidade Tipo III'
    }

    resultado = mapa_saida.get(pred, pred)
    st.success(f"Nível de obesidade previsto: **{resultado}**")
