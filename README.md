# AlzCare-AI 🌐🧠

**AlzCare-AI** es un sistema avanzado de inteligencia artificial diseñado para la detección temprana del Alzheimer, combinado con recomendaciones personalizadas para cuidadores y familiares. Este proyecto integra Machine Learning, procesamiento de lenguaje natural y visualización interactiva para mejorar la calidad de vida de pacientes y sus familias.

## Objetivos del proyecto 🎯

- **Detección temprana del Alzheimer** mediante datos tabulares y de imágenes de tomografías.
- **Recomendaciones personalizadas** basadas en el grado de Alzheimer detectado.
- Proporcionar **interacciones inteligentes** a través de un chatbot impulsado por modelos de lenguaje grande (LLM).

## Funcionalidades principales 🔄

### Esenciales ✅
1. **Modelo de detección de Alzheimer**:
   - Entrenar un modelo con datos tabulares y de tomografías.
   - Evaluar el rendimiento asegurando que el overfitting sea menor al 5%.

2. **Chatbot interactivo**:
   - Crear una interfaz con Streamlit o Gradio.
   - Integrar el modelo de detección con la interfaz.

### Opcionales ⚙️
3. **Optimización del modelo**:
   - Ajuste de hiperparámetros y validación cruzada.
   - Manejo de datos desbalanceados con sobremuestreo o submuestreo.

4. **Implementación del LLM**:
   - Utilizar OpenAI GPT-4 o Hugging Face Transformers para mejorar la interacción del chatbot.
   - Ajustar el modelo de lenguaje para respuestas personalizadas.

### Avanzados 🚀
5. **Recomendaciones personalizadas**:
   - Sistema basado en el grado de Alzheimer detectado.
   - Consejos y recursos útiles para cuidadores y familiares.

6. **Dockerización de la aplicación**:
   - Crear un Dockerfile y configurar Docker Compose.

7. **Tracking con MLflow**:
   - Registrar experimentos, métricas y artefactos de modelos.

### Experto 🧠
8. **Estudio del rendimiento del modelo**:
   - Análisis exhaustivo del rendimiento.
   - Generar informes detallados con métricas de evaluación.

## Tecnologías utilizadas 💻

- **Lenguajes**: Python
- **Librerías de ML**: Scikit-learn, TensorFlow, Keras, OpenCV, Hugging Face Transformers
- **Análisis de datos**: Pandas, NumPy, Matplotlib, Seaborn
- **Infraestructura y desarrollo**: Git, GitHub, Streamlit, Gradio
- **Bases de datos**: MongoDB (imágenes), MySQL (datos tabulares)
- **Nube**: Google Colab para entrenamiento de modelos

## Datos a utilizar 📊

1. **Datasets**:
   - **Tabulares**: CSV con datos estructurados.
   - **Imágenes**: Tomografías en formatos JPEG/PNG, almacenadas en MongoDB.

2. **Privacidad**:
   - Cumplir normativas éticas y legales.
   - Anonimizar datos sensibles.

## Estructura del proyecto 🔍

```plaintext
AlzCare-AI/
├── data/         # Conjuntos de datos para entrenamiento y evaluación
├── notebooks/    # Jupyter Notebooks para exploración y pruebas
├── src/          # Código fuente
│   ├── models/       # Entrenamiento de modelos
│   ├── preprocessing/ # Limpieza y transformación de datos
│   ├── utils/        # Funciones auxiliares
├── tests/        # Pruebas unitarias e integradas
├── requirements.txt  # Dependencias
└── .gitignore    # Archivos ignorados por Git
```

## Instalación 🔧

### Prerrequisitos ⚡

- Python 3.8 o superior
- pip (Gestor de paquetes de Python)

### Pasos 🚀

1. Clona el repositorio:

   ```bash
   git clone https://github.com/usuario/AlzCare-AI.git
   cd AlzCare-AI
   ```

2. Crea y activa un entorno virtual:

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # En Windows: venv\Scripts\activate
   ```

3. Instala las dependencias:

   ```bash
   pip install -r requirements.txt
   ```

4. Coloca los datasets necesarios en `data/`.

## Guía de uso 🌐

### 1. Análisis exploratorio y preprocesamiento 🧪

Ejecuta los notebooks:

```bash
jupyter notebook notebooks/
```

### 2. Entrenamiento de modelos 🔢

Corre el script principal:

```bash
python src/main.py
```

### 3. Pruebas unitarias 🛠️

Ejecuta las pruebas:

```bash
pytest tests/
```

### 4. Despliegue 🚢

Consulta `src/deployment/` para configuraciones avanzadas.

## Contribuciones 📢

Las contribuciones son bienvenidas. Por favor, sigue estos pasos:

1. Haz un fork del repositorio.
2. Crea una rama:

   ```bash
   git checkout -b mi-funcion
   ```

3. Envía un pull request.

## Licencia 🔒

Este proyecto está bajo la licencia MIT. Consulta `LICENSE` para más información.

## Contacto 📧

Para preguntas o sugerencias:

- **Nombre:** Emma Montalbán
- **Correo:** [emma.montalban@example.com](mailto:emma.montalban@example.com)


