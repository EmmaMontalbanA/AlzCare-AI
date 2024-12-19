import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os

# Configuración inicial
MODEL_PATH = os.path.join(os.getcwd(), "..", "models", "Bagging.pkl")
model = joblib.load(MODEL_PATH)

# Cargar datos relevantes para métricas
metrics_path = os.path.join(os.getcwd(), "..", "models", "metrics", "Bagging.csv")
metrics = pd.read_csv(metrics_path)

def predict_alzheimer(input_data):
    # Mapear las características necesarias según el modelo entrenado
    required_features = ['FunctionalAssessment', 'ADL', 'MMSE', 'BehavioralProblems', 'MemoryComplaints']
    
    # Crear un mapeo entre las entradas del usuario y las características del modelo
    feature_mapping = {
        'FunctionalAssessment': input_data['Age'] / 100,  # Normalizar edad como aproximación
        'ADL': -1 if input_data['Gender'] == 1 else 1,  # Mapear género a ADL
        'MMSE': (input_data['PhysicalActivity'] + input_data['DietQuality']) / 30,  # Normalizar actividades
        'BehavioralProblems': 1 if input_data['Smoking'] == 1 else 0,  # Mapear comportamientos
        'MemoryComplaints': 1 if input_data['MemoryComplaints'] == 1 else 0  # Mantener quejas de memoria
    }
    
    # Crear DataFrame con las características requeridas
    df = pd.DataFrame([{feature: feature_mapping[feature] for feature in required_features}])
    probability = model.predict_proba(df)[:, 1]
    return probability[0]

# Función de autenticación
def authenticate(username, password):
    # Aquí deberías implementar tu lógica de autenticación real
    if username == "admin" and password == "admin123":
        return "admin"
    elif username == "user" and password == "user123":
        return "user"
    else:
        return None
    
# Interfaz Streamlit
st.title("Sistema de Predicción Temprana de Alzheimer")

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.role = None

if not st.session_state.authenticated:
    st.write("Bienvenido al Sistema de Predicción Temprana de Alzheimer. \
            Por favor, introduzca sus credenciales.")
    
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
        #st.header("Perfil Administrador")
        admin_option = st.sidebar.selectbox("Acceso a Consultas", 
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
            #st.header("Perfil Usuario")
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
    
            # Predicción y recomendaciones
            if st.button("Obtener Predicción"):
                probability = predict_alzheimer(user_data)
                st.success(f"La probabilidad de desarrollar Alzheimer es: {probability:.2%}")
        
                st.subheader("Recomendaciones")
                if probability > 0.5:
                    st.write("- Consulte a un especialista para una evaluación clínica detallada.")
                    st.write("- Realice actividades que estimulen la cognición, como lectura o juegos mentales.")
                    st.write("- Mantenga un estilo de vida saludable con dieta balanceada y ejercicio regular.")
                else:
                    st.write("- Continúe con un estilo de vida saludable.")
                    st.write("- Manténgase activo física y mentalmente.")


    if st.button("Cerrar sesión"):
        st.session_state.authenticated = False
        st.session_state.role = None
        st.rerun()