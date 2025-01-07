import streamlit as st
# Configuración de la página - DEBE SER LA PRIMERA INSTRUCCIÓN DE STREAMLIT
st.set_page_config(page_title="AlzCare AI", page_icon="🧠", layout="wide")

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os
from datetime import datetime
import sys

# Añadir el directorio padre al path para poder importar los módulos
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chatbot.chatbot_interface import ChatbotInterface

# Configuración inicial
MODEL_PATH = os.path.join(os.getcwd(), "..", "models", "Bagging.pkl")
model = joblib.load(MODEL_PATH)

# Cargar datos relevantes para métricas
metrics_path = os.path.join(os.getcwd(), "..", "models", "metrics", "Bagging.csv")
metrics = pd.read_csv(metrics_path)

# Estilos CSS personalizados
st.markdown("""
<style>
.chat-container {
    background-color: #ffffff;
    border-radius: 10px;
    padding: 20px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    margin-top: 20px;
}

.chat-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 20px;
}

.chat-messages {
    height: 10px;
    overflow-y: auto;
    padding: 10px;
    border: 1px solid #eee;
    border-radius: 5px;
    margin-bottom: 20px;
}

.chat-button {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    padding: 0.5rem 1rem;
    background-color: #f0f2f6;
    border: none;
    border-radius: 5px;
    cursor: pointer;
    transition: background-color 0.3s;
}

.chat-button:hover {
    background-color: #e0e2e6;
}

.stButton>button {
    width: 100%;
}

.fixed-chat-button {
    position: fixed;
    bottom: 20px;
    right: 20px;
    z-index: 1000;
}

.chat-window {
    position: fixed;
    bottom: 80px;
    right: 20px;
    width: 400px;
    z-index: 1000;
    background: white;
    border-radius: 10px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

def predict_alzheimer(input_data):
    required_features = ['FunctionalAssessment', 'ADL', 'MMSE', 'BehavioralProblems', 'MemoryComplaints']
    
    feature_mapping = {
        'FunctionalAssessment': input_data['Age'] / 100,
        'ADL': -1 if input_data['Gender'] == 1 else 1,
        'MMSE': (input_data['PhysicalActivity'] + input_data['DietQuality']) / 30,
        'BehavioralProblems': 1 if input_data['Smoking'] == 1 else 0,
        'MemoryComplaints': 1 if input_data['MemoryComplaints'] == 1 else 0
    }
    
    df = pd.DataFrame([{feature: feature_mapping[feature] for feature in required_features}])
    probability = model.predict_proba(df)[:, 1]
    return probability[0]

def authenticate(username, password):
    if username == "admin" and password == "admin123":
        return "admin"
    elif username == "user" and password == "user123":
        return "user"
    else:
        return None

# Inicialización de variables de sesión
if 'chatbot' not in st.session_state:
    st.session_state.chatbot = ChatbotInterface()
    st.session_state.show_chat = False

# Interfaz Streamlit
st.title("Sistema de Predicción Temprana de Alzheimer")

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.role = None

if not st.session_state.authenticated:
    st.write("Bienvenido al Sistema de Predicción Temprana de Alzheimer. Por favor, introduzca sus credenciales.")
    
    username = st.text_input("Usuario")
    password = st.text_input("Contraseña", type="password")
    if st.button("Iniciar sesión"):
        role = authenticate(username, password)
        if role:
            st.session_state.authenticated = True
            st.session_state.role = role
            st.success("Inicio de sesión exitoso")
            st.rerun()
        else:
            st.error("Usuario o contraseña incorrectos")
else:
    if st.session_state.role == "admin":
        admin_option = st.sidebar.selectbox("¿Qué informe quiere consultar?", 
                                        ["Seleccione una opción",
                                        "Informe de Rendimiento", 
                                        "Informes de Rendimiento por Modelo", 
                                        "Comparativa Rendimiento Modelos"])
        
        if admin_option == "Informe de Rendimiento":
            tab1, tab2 = st.tabs(["Rendimiento del Modelo", "Gráficas de Rendimiento"])
            
            with tab1:
                st.subheader("Rendimiento del Modelo")
                st.dataframe(metrics)
            
            with tab2:
                st.subheader("Gráficas de Rendimiento")
                graph_path = os.path.join(os.getcwd(), "..", "models", "graph")
                ensemble_graphs = [f for f in os.listdir(graph_path) if f.startswith('Bagging') and f.endswith('.png')]

                col1, col2 = st.columns(2)
                for i, graph_file in enumerate(ensemble_graphs):
                    with col1 if i % 2 == 0 else col2:
                        st.image(os.path.join(graph_path, graph_file), caption=graph_file, use_container_width=True)

        elif admin_option == "Informes de Rendimiento por Modelo":
            metrics_folder = os.path.join(os.getcwd(), "..", "models", "metrics")
            model_files = [f for f in os.listdir(metrics_folder) if f.endswith('.csv') and not f.startswith('all_')]
    
            tabs = st.tabs([f.split('.')[0] for f in model_files])            
            for tab, model_file in zip(tabs, model_files):
                with tab:
                    model_name = model_file.split('.')[0]
                    st.subheader(f"Modelo: {model_name}")
                    metrics_tab, visualizations_tab = st.tabs(["Rendimiento del Modelo", "Gráficas de Rendimiento"])
                    with metrics_tab:
                        metrics = pd.read_csv(os.path.join(metrics_folder, model_file))
                        st.dataframe(metrics) 
                    with visualizations_tab:
                        graph_path = os.path.join(os.getcwd(), "..", "models", "graph")
                        model_graphs = [f for f in os.listdir(graph_path) if f.startswith(model_name) and f.endswith('.png')]

                        col1, col2 = st.columns(2)   
                        for i, graph_file in enumerate(model_graphs):
                            with col1 if i % 2 == 0 else col2:
                                st.image(os.path.join(graph_path, graph_file), caption=graph_file, use_container_width=True)

        elif admin_option == "Comparativa Rendimiento Modelos":
            st.subheader("Comparación de Rendimiento de Modelos")
            all_metrics_file = [f for f in os.listdir(os.path.join(os.getcwd(), "..", "models", "metrics")) if f.startswith('all_') and f.endswith('.csv')][0]
            all_metrics = pd.read_csv(os.path.join(os.getcwd(), "..", "models", "metrics", all_metrics_file))
            st.dataframe(all_metrics)       

    
    elif st.session_state.role == "user":
        # Crear dos columnas: una para el formulario y otra para la predicción
        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            st.subheader("Formulario de Datos")
            
            # Entradas del usuario
            age = st.number_input("Edad", min_value=40, max_value=100, value=60)
            gender = st.selectbox("Género", ["Masculino", "Femenino"])
            physical_activity = st.slider("Actividad física semanal (horas)", min_value=0, max_value=20, value=5)
            diet_quality = st.slider("Calidad de la dieta (1-10)", min_value=1, max_value=10, value=5)
            smoking = st.selectbox("¿Fuma?", ["Sí", "No"])
            memory_complaints = st.selectbox("¿Ha tenido problemas de memoria?", ["Sí", "No"])

            # Preparar entrada para el modelo
            gender = 1 if gender == "Masculino" else 0
            smoking = 1 if smoking == "Sí" else 0
            memory_complaints = 1 if memory_complaints == "Sí" else 0
        
            user_data = {
                "Age": age,
                "Gender": gender,
                "Smoking": smoking,
                "PhysicalActivity": physical_activity,
                "DietQuality": diet_quality,
                "MemoryComplaints": memory_complaints
            }

        with col2:
            st.subheader("Predicción y Recomendaciones")
            if st.button("Obtener Predicción"):
                probability = predict_alzheimer(user_data)
                st.session_state.chatbot.update_context(diagnosis=probability, user_data=user_data)
                st.success(f"La probabilidad de desarrollar Alzheimer es: {probability:.2%}")
                
                st.subheader("Recomendaciones Generales")
                if probability > 0.5:
                    st.write("- Consulte a un especialista para una evaluación clínica detallada.")
                    st.write("- Realice actividades que estimulen la cognición.")
                    st.write("- Mantenga un estilo de vida saludable.")
                else:
                    st.write("- Continúe con un estilo de vida saludable.")
                    st.write("- Manténgase activo física y mentalmente.")
                    
        with col3:

            # Botón de chat flotante
            st.markdown('<div class="fixed-chat-button">', unsafe_allow_html=True)
            if st.button("💬 Asistente Virtual", key="chat_toggle"):
                st.session_state.show_chat = not st.session_state.show_chat
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

            # Ventana de chat
            if st.session_state.show_chat:
                st.markdown('<div class="chat-window">', unsafe_allow_html=True)
                
                # Header del chat
                col_title, col_close = st.columns([5,1])
                with col_title:
                    st.markdown("### 🤖 AlzCare AI Assistant")
                with col_close:
                    if st.button("❌", key="close_chat"):
                        st.session_state.show_chat = False
                        st.rerun()

                # Mensaje inicial si no hay historial
                if not st.session_state.chatbot.get_history():
                    st.info("""
                    ¡Bienvenido! Puedo ayudarte con información sobre:
                    - ❓ Qué es el Alzheimer
                    - 🔍 Síntomas y señales de alerta
                    - 💊 Tratamientos disponibles
                    - 🤝 Cuidados necesarios
                    - 🎯 Medidas de prevención
                    
                    ¿Sobre qué tema te gustaría saber más?
                    """)

                st.markdown('</div>', unsafe_allow_html=True)
                # Área de mensajes
                st.markdown('<div class="chat-messages">', unsafe_allow_html=True)
                for message in st.session_state.chatbot.get_history():
                    if message["role"] == "user":
                        st.write(f"👤 Tú ({message['timestamp']}): {message['content']}")
                    else:
                        st.write(f"🤖 AlzCare AI ({message['timestamp']}): {message['content']}")
                st.markdown('</div>', unsafe_allow_html=True)

                # Input y botones
                user_message = st.text_input("Escribe tu mensaje aquí...", key="user_input")
                col_send, col_clear = st.columns([4,1])
                
                with col_send:
                    if st.button("📤 Enviar", key="send_message"):
                        if user_message:
                            st.session_state.chatbot.add_message(user_message, "user")
                            bot_response = st.session_state.chatbot.generate_response(user_message)
                            st.session_state.chatbot.add_message(bot_response, "bot")
                            st.rerun()
                
                with col_clear:
                    if st.button("🗑️", key="clear_chat"):
                        st.session_state.chatbot.clear_history()
                        st.rerun()

    if st.button("Cerrar sesión"):
        st.session_state.authenticated = False
        st.session_state.role = None
        st.rerun()