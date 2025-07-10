# app/app.py
# =============================================================================
# SIADO – Sistema de Apoio ao Diagnóstico de Obesidade
# =============================================================================
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt
from streamlit_shap import st_shap

# -----------------------------------------------------------------------------
# CONFIGURAÇÃO DA PÁGINA
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="SIADO - Diagnóstico de Obesidade",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# FUNÇÕES AUXILIARES
# -----------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    """Carrega o pipeline e o label-encoder salvos em disco."""
    try:
        pipeline = joblib.load("models/pipeline_obesidade.joblib")
        encoder  = joblib.load("models/label_encoder.joblib")
        return pipeline, encoder
    except FileNotFoundError:
        st.error("Arquivos do modelo não encontrados em 'models/'.")
        return None, None


@st.cache_data
def get_shap_explanation(_pipeline, input_df):
    """Gera valores SHAP para uma única instância."""
    try:
        preproc   = _pipeline.named_steps["preprocessor"]
        clf       = _pipeline.named_steps["classifier"]
        
        # Transformar os dados
        processed = preproc.transform(input_df)
        
        # Garantir que processed seja um array 2D
        if processed.ndim == 1:
            processed = processed.reshape(1, -1)
        
        # Criar o explicador SHAP
        explainer = shap.TreeExplainer(clf)
        
        # Obter valores SHAP
        shap_vals = explainer.shap_values(processed)
        
        # Normalizar shap_vals para garantir consistência
        if isinstance(shap_vals, list):
            # Para classificação multiclasse
            normalized_shap = []
            for sv in shap_vals:
                if sv.ndim == 1:
                    normalized_shap.append(sv.reshape(1, -1))
                else:
                    normalized_shap.append(sv)
            shap_vals = normalized_shap
        else:
            # Para classificação binária
            if shap_vals.ndim == 1:
                shap_vals = shap_vals.reshape(1, -1)
        
        return shap_vals, explainer, processed
    except Exception as e:
        st.error(f"Erro na função get_shap_explanation: {str(e)}")
        return None, None, None


def show_simplified_analysis(pred):
    """Mostra análise simplificada baseada em conhecimento médico."""
    st.subheader("Análise Simplificada dos Dados")
    patient_data = pred["input_df"].iloc[0].to_dict()
    
    # Destacar fatores de risco conhecidos
    risk_factors = []
    protective_factors = []
    
    # Análise baseada em conhecimento médico
    imc = patient_data.get('imc', 0)
    if imc > 30:
        risk_factors.append(f"IMC indica obesidade: {imc:.1f}")
    elif imc > 25:
        risk_factors.append(f"IMC indica sobrepeso: {imc:.1f}")
    elif imc < 18.5:
        risk_factors.append(f"IMC abaixo do normal: {imc:.1f}")
    else:
        protective_factors.append(f"IMC normal: {imc:.1f}")
    
    if patient_data.get('come_comida_calorica_freq') == 'yes':
        risk_factors.append("Consome alimentos calóricos frequentemente")
    
    freq_atividade = patient_data.get('freq_atividade_fisica', 0)
    if freq_atividade < 1:
        risk_factors.append("Sedentarismo (sem atividade física)")
    elif freq_atividade < 2:
        risk_factors.append("Baixa frequência de atividade física")
    else:
        protective_factors.append("Atividade física regular")
    
    if patient_data.get('historia_familiar_sobrepeso') == 'yes':
        risk_factors.append("Histórico familiar de sobrepeso")
    
    if patient_data.get('fumante') == 'yes':
        risk_factors.append("Tabagismo")
    
    consumo_agua = patient_data.get('consumo_agua_litros', 0)
    if consumo_agua < 2:
        risk_factors.append("Baixo consumo de água")
    else:
        protective_factors.append("Consumo adequado de água")
    
    # Exibir fatores
    col1, col2 = st.columns(2)
    
    with col1:
        if risk_factors:
            st.error("**Fatores de Risco Identificados:**")
            for factor in risk_factors:
                st.write(f"• {factor}")
    
    with col2:
        if protective_factors:
            st.success("**Fatores Protetivos Identificados:**")
            for factor in protective_factors:
                st.write(f"• {factor}")
    
    # Recomendações gerais
    st.subheader("Recomendações Gerais")
    st.info(
        """
        **Baseado nos dados inseridos, considere:**
        - Manter ou adotar uma dieta balanceada
        - Praticar atividade física regularmente
        - Manter hidratação adequada
        - Monitorar o peso regularmente
        - Consultar um profissional de saúde para orientação personalizada
        """
    )


# -----------------------------------------------------------------------------
# CARREGA MODELO
# -----------------------------------------------------------------------------
pipeline, encoder = load_artifacts()

# -----------------------------------------------------------------------------
# INTERFACE
# -----------------------------------------------------------------------------
if pipeline and encoder:

    # ---------- Sidebar -------------------------------------------------------
    st.sidebar.image(
        "https://images.unsplash.com/photo-1579684385127-1ef15d508118?q=80&w=1780&auto=format&fit=crop",
        use_column_width=True
    )
    st.sidebar.title("Parâmetros do Paciente")
    st.sidebar.write("Insira os dados para obter a previsão do modelo.")

    mapa_genero  = {"Feminino": "Female", "Masculino": "Male"}
    mapa_sim_nao = {"Sim": "yes", "Não": "no"}
    mapa_refeicoes = {
        "Não come": "no", "Às vezes": "Sometimes",
        "Frequentemente": "Frequently", "Sempre": "Always"
    }
    mapa_alcool = {
        "Não bebe": "no", "Às vezes": "Sometimes",
        "Frequentemente": "Frequently", "Sempre": "Always"
    }
    mapa_transporte = {
        "Transporte Público": "Public_Transportation",
        "Automóvel": "Automobile", "Caminhando": "Walking",
        "Motocicleta": "Motorbike", "Bicicleta": "Bike"
    }
    mapa_tempo_disp = {"0-2 horas": 0, "3-5 horas": 1, "Mais de 5 horas": 2}

    # ---- Entradas básicas
    genero   = st.sidebar.selectbox("Gênero", list(mapa_genero))
    idade    = st.sidebar.number_input("Idade", 1, 120, 25)
    altura_m = st.sidebar.number_input("Altura (m)", 0.5, 2.5, 1.70, format="%.2f")
    peso_kg  = st.sidebar.number_input("Peso (kg)", 10.0, 250.0, 70.0, format="%.1f")

    # ---- Hábitos
    st.sidebar.subheader("Hábitos de Vida e Comportamento")
    hist_fam = st.sidebar.radio("Histórico familiar de sobrepeso?", list(mapa_sim_nao), horizontal=True)
    favc     = st.sidebar.radio("Alimentos calóricos frequentemente (FAVC)?", list(mapa_sim_nao), horizontal=True)
    fcvc     = st.sidebar.slider("Frequência de consumo de vegetais (FCVC)", 1, 3, 2)
    ncp      = st.sidebar.slider("Número de refeições principais (NCP)", 1, 4, 3)
    caec     = st.sidebar.select_slider("Consome algo entre as refeições (CAEC)?", list(mapa_refeicoes))
    smoke    = st.sidebar.radio("É fumante?", list(mapa_sim_nao), horizontal=True, index=1)
    ch2o     = st.sidebar.slider("Consumo diário de água (CH2O)", 1, 3, 2)
    scc      = st.sidebar.radio("Monitora calorias (SCC)?", list(mapa_sim_nao), horizontal=True, index=1)

    faf_dias = st.sidebar.slider("Atividade física (dias/semana)", 0, 7, 2)
    faf      = 0 if faf_dias == 0 else (1 if faf_dias <= 2 else (2 if faf_dias <= 4 else 3))

    tue_str  = st.sidebar.selectbox("Tempo em dispositivos (TUE)", list(mapa_tempo_disp))
    tue      = mapa_tempo_disp[tue_str]

    calc     = st.sidebar.select_slider("Frequência de álcool (CALC)?", list(mapa_alcool))
    mtrans   = st.sidebar.selectbox("Meio de transporte (MTRANS)", list(mapa_transporte))

    # ---------- DataFrame de entrada -----------------------------------------
    dados = {
        "genero": mapa_genero[genero],
        "idade": idade,
        "altura_m": altura_m,
        "peso_kg": peso_kg,
        "historia_familiar_sobrepeso": mapa_sim_nao[hist_fam],
        "come_comida_calorica_freq": mapa_sim_nao[favc],
        "freq_consumo_vegetais": fcvc,
        "num_refeicoes_principais": ncp,
        "come_entre_refeicoes": mapa_refeicoes[caec],
        "fumante": mapa_sim_nao[smoke],
        "consumo_agua_litros": ch2o,
        "monitora_calorias": mapa_sim_nao[scc],
        "freq_atividade_fisica": faf,
        "tempo_uso_dispositivos": tue,
        "freq_consumo_alcool": mapa_alcool[calc],
        "meio_transporte": mapa_transporte[mtrans],
    }
    input_df = pd.DataFrame([dados])
    input_df["imc"] = input_df["peso_kg"] / (input_df["altura_m"] ** 2)

    # ---------- Predição ------------------------------------------------------
    if "last_prediction" not in st.session_state:
        st.session_state.last_prediction = None

    if st.sidebar.button("Analisar Paciente", type="primary"):
        y_pred  = pipeline.predict(input_df)
        y_proba = pipeline.predict_proba(input_df)
        st.session_state.last_prediction = {
            "encoded": int(y_pred[0]),
            "decoded": encoder.inverse_transform(y_pred)[0],
            "proba":   float(np.max(y_proba) * 100),
            "all_probas": y_proba,
            "input_df": input_df,
        }

    # ---------- Exibição dos resultados --------------------------------------
    if st.session_state.last_prediction:
        pred = st.session_state.last_prediction
        resultado_fmt = pred["decoded"].replace("_", " ")

        st.title("Resultado da Análise Preditiva")
        st.markdown("---")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Diagnóstico Previsto")
            if "Obesity" in pred["decoded"] or "Overweight" in pred["decoded"]:
                st.error(f"## {resultado_fmt}")
            elif "Normal" in pred["decoded"]:
                st.success(f"## {resultado_fmt}")
            else:
                st.warning(f"## {resultado_fmt}")

        with col2:
            st.subheader("Confiança do Modelo")
            st.metric("Probabilidade", f"{pred['proba']:.1f}%")
            st.progress(int(pred["proba"]))

        # ---------- Análise Simplificada dos Dados ----------------------------
        show_simplified_analysis(pred)

        # ---------- Explicabilidade (SHAP) - Contribuição das Variáveis ------
        with st.expander("🔬 Entenda a Previsão (XAI) - Contribuição das Variáveis", expanded=False):
            st.info(
                "Fatores em **vermelho** aumentam o risco; "
                "fatores em **azul** reduzem o risco."
            )

            # Tentar análise SHAP
            shap_vals, explainer, processed_features = get_shap_explanation(pipeline, pred["input_df"])
            
            if shap_vals is not None and explainer is not None:
                try:
                    # Determinar se é classificação multiclasse ou binária
                    if isinstance(shap_vals, list) and len(shap_vals) > 1:  # multiclasse
                        base_value = explainer.expected_value[pred["encoded"]]
                        shap_1d = shap_vals[pred["encoded"]][0]
                    else:  # binário
                        if isinstance(shap_vals, list):
                            shap_vals = shap_vals[0]
                        base_value = explainer.expected_value
                        shap_1d = shap_vals[0]

                    # Garantir que base_value seja escalar
                    if isinstance(base_value, (list, np.ndarray)):
                        base_value = float(base_value[0])
                    else:
                        base_value = float(base_value)

                    # CORREÇÃO: Garantir que shap_1d seja 1D
                    if isinstance(shap_1d, np.ndarray):
                        # Achatar o array para garantir que seja 1D
                        shap_1d = shap_1d.flatten()
                    
                    # Verificar se ainda há problemas de dimensão
                    if shap_1d.ndim > 1:
                        shap_1d = shap_1d.ravel()

                    # Obter nomes das features
                    try:
                        feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
                    except:
                        feature_names = [f"feature_{i}" for i in range(len(shap_1d))]

                    # Verificar se o número de features corresponde
                    if len(feature_names) != len(shap_1d):
                        feature_names = [f"feature_{i}" for i in range(len(shap_1d))]

                    # -------- Gráfico de Barras com Contribuição SHAP -----------
                    st.subheader("Contribuição das Variáveis")
                    
                    # CORREÇÃO: Criar DataFrame com verificação de dimensões
                    try:
                        feature_importance = pd.DataFrame({
                            'feature': feature_names,
                            'importance': shap_1d
                        }).sort_values('importance', key=abs, ascending=False)
                    except ValueError as e:
                        # Fallback: truncar ou expandir arrays conforme necessário
                        min_len = min(len(feature_names), len(shap_1d))
                        feature_importance = pd.DataFrame({
                            'feature': feature_names[:min_len],
                            'importance': shap_1d[:min_len]
                        }).sort_values('importance', key=abs, ascending=False)
                    
                    # Limitar a 15 features mais importantes
                    top_features = feature_importance.head(15)
                    
                    fig, ax = plt.subplots(figsize=(12, 8))
                    colors = ['#FF6B6B' if x > 0 else '#4ECDC4' for x in top_features['importance']]
                    bars = ax.barh(range(len(top_features)), top_features['importance'], color=colors)
                    
                    # Configurar eixos
                    ax.set_yticks(range(len(top_features)))
                    ax.set_yticklabels(top_features['feature'], fontsize=10)
                    ax.set_xlabel('Contribuição SHAP', fontsize=12)
                    ax.set_title('Top 15 - Importância das Variáveis na Predição', fontsize=14, fontweight='bold')
                    ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
                    
                    # Adicionar valores nas barras
                    for i, (bar, value) in enumerate(zip(bars, top_features['importance'])):
                        offset = max(abs(top_features['importance'])) * 0.02
                        ax.text(value + (offset if value > 0 else -offset), 
                               bar.get_y() + bar.get_height()/2, 
                               f'{value:.3f}', ha='left' if value > 0 else 'right', va='center', fontsize=9)
                    
                    # Adicionar legenda
                    ax.text(0.02, 0.98, 'Vermelho: Aumenta risco\nAzul: Reduz risco', 
                           transform=ax.transAxes, fontsize=10, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                    
                    plt.tight_layout()
                    st.pyplot(fig)

                    # -------- Interpretação Textual -----------
                    st.subheader("Interpretação dos Resultados")
                    
                    # Separar fatores de risco e proteção
                    risk_factors = top_features[top_features['importance'] > 0].head(5)
                    protective_factors = top_features[top_features['importance'] < 0].head(5)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if not risk_factors.empty:
                            st.error("**Top 5 Fatores de Risco:**")
                            for _, row in risk_factors.iterrows():
                                st.write(f"• {row['feature']}: +{row['importance']:.3f}")
                    
                    with col2:
                        if not protective_factors.empty:
                            st.success("**Top 5 Fatores Protetivos:**")
                            for _, row in protective_factors.iterrows():
                                st.write(f"• {row['feature']}: {row['importance']:.3f}")

                except Exception as e:
                    st.warning(f"Erro ao processar valores SHAP: {str(e)}")
                    st.info("Análise SHAP não disponível. Consulte a análise simplificada acima.")
            else:
                st.info("Análise SHAP não disponível. Consulte a análise simplificada acima.")

        # ---------- Dados Adicionais --------------------------------------------
        with st.expander("📊 Dados Adicionais", expanded=False):
            # -------- Tabela com Valores Originais -----------
            st.subheader("Valores dos Parâmetros do Paciente")
            
            # Criar tabela com valores originais
            patient_data = pred["input_df"].iloc[0].to_dict()
            patient_df = pd.DataFrame(list(patient_data.items()), columns=['Parâmetro', 'Valor'])
            
            # Formatação dos valores
            patient_df['Valor'] = patient_df['Valor'].apply(lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else str(x))
            
            # Exibir tabela
            st.dataframe(patient_df, use_container_width=True)

            # Probabilidades de cada classe
            st.markdown("##### Probabilidade por Classe")
            prob_df = (
                pd.DataFrame(pred["all_probas"], columns=encoder.classes_)
                .T.rename(columns={0: "Probabilidade"})
            )
            prob_df["Probabilidade"] *= 100
            st.bar_chart(prob_df)

    else:
        # ------------------- Tela inicial ------------------------------------
        st.title("🩺 SIADO: Sistema de Apoio ao Diagnóstico de Obesidade")
        st.markdown("---")
        st.subheader("Ferramenta de Data Science para Auxiliar a Prática Clínica")
        st.markdown(
            """
            Bem-vindo(a) ao **SIADO**! O modelo possui **97,8 % de acurácia** e prevê
            o nível de obesidade a partir de dados clínicos e comportamentais.
            
            **Como usar:**  
            1. Preencha os dados na barra lateral.  
            2. Clique em **"Analisar Paciente"**.  
            3. Veja o diagnóstico e a explicação gerada pelo modelo.
            
            *Protótipo acadêmico — não substitui avaliação médica profissional.*
            """
        )
        st.info("Modelo treinado no dataset Kaggle 'Obesity or CVD risk'.", icon="ℹ️")

    # ------------------- Créditos --------------------------------------------
    st.sidebar.markdown("---")
    with st.sidebar.expander("Sobre a Equipe"):
        st.write(
            """
            **Integrantes:**  
            • Rosicléia Cavalcante Mota  
            • Guillermo Jesus Camahuali Privat  
            • Kelly Priscilla Matos Campos  
            
            *Tech Challenge – Pós-Graduação FIAP (Data Analytics).*
            """
        )
else:
    st.error("Não foi possível carregar o modelo. Verifique se os arquivos estão na pasta 'models/'.")
    st.info("Arquivos necessários: 'pipeline_obesidade.joblib' e 'label_encoder.joblib'")