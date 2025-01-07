KNOWLEDGE_BASE = {
    "que_es": {
        "patterns": ["qué es el alzheimer", "que es alzheimer", "explica alzheimer", "información sobre alzheimer"],
        "responses": [
            """El Alzheimer es una enfermedad neurodegenerativa que afecta al cerebro y es la causa más común de demencia. 
            Según la Organización Mundial de la Salud (OMS):
            - Afecta principalmente a la memoria, el pensamiento y el comportamiento
            - Es progresiva, lo que significa que los síntomas empeoran gradualmente con el tiempo
            - Representa entre un 60% y un 70% de los casos de demencia
            - No es una parte normal del envejecimiento, aunque la edad es el principal factor de riesgo"""
        ]
    },
    "sintomas": {
        "patterns": ["síntomas", "signos", "señales", "como saber si tengo", "indicios"],
        "responses": [
            """Los principales síntomas del Alzheimer, según la Asociación de Alzheimer, incluyen:

            Síntomas tempranos:
            - Pérdida de memoria que afecta a la vida cotidiana
            - Dificultad para planificar o resolver problemas
            - Problemas para completar tareas familiares
            - Desorientación en tiempo y lugar
            
            Síntomas intermedios:
            - Problemas con el lenguaje (encontrar palabras)
            - Juicio disminuido
            - Cambios de humor y personalidad
            - Aislamiento social
            
            Síntomas avanzados:
            - Pérdida significativa de memoria
            - Dependencia total para actividades diarias
            - Problemas de movilidad
            - Cambios severos de comportamiento"""
        ]
    },
    "tratamiento": {
        "patterns": ["tratamiento", "cura", "medicamentos", "terapia", "como se trata"],
        "responses": [
            """Según la Sociedad Española de Neurología, el tratamiento del Alzheimer es multifacético:

            Tratamiento farmacológico:
            - Inhibidores de la colinesterasa
            - Antagonistas del receptor NMDA
            - Medicamentos para síntomas específicos
            
            Tratamiento no farmacológico:
            - Estimulación cognitiva
            - Terapia ocupacional
            - Ejercicio físico regular
            - Actividades sociales
            
            Es importante destacar que:
            - No existe cura definitiva actualmente
            - El tratamiento temprano puede ralentizar la progresión
            - Cada paciente requiere un plan personalizado
            - Es fundamental el seguimiento médico regular"""
        ]
    },
    "cuidados": {
        "patterns": ["cómo cuidar", "cuidados", "atención", "ayuda", "que hacer"],
        "responses": [
            """Recomendaciones de la Confederación Española de Alzheimer para el cuidado:

            Cuidados básicos:
            - Mantener rutinas diarias
            - Asegurar un ambiente seguro y familiar
            - Supervisar la medicación
            - Mantener una buena higiene
            
            Actividades recomendadas:
            - Ejercicios de memoria y estimulación
            - Actividad física adaptada
            - Socialización controlada
            - Tareas sencillas que fomenten la independencia
            
            Consejos para cuidadores:
            - Buscar apoyo profesional y familiar
            - Tomar descansos regulares
            - Unirse a grupos de apoyo
            - Mantener la paciencia y comprensión"""
        ]
    },
    "prevencion": {
        "patterns": ["prevenir", "prevención", "evitar", "reducir riesgo"],
        "responses": [
            """Según estudios recientes de la Academia Americana de Neurología:

            Factores de estilo de vida:
            - Actividad física regular (30 minutos diarios)
            - Dieta mediterránea
            - Estimulación mental continua
            - Socialización activa
            
            Factores de salud:
            - Control de presión arterial
            - Manejo de diabetes
            - Mantener peso saludable
            - No fumar
            
            Actividades recomendadas:
            - Lectura regular
            - Juegos de memoria
            - Aprendizaje de nuevas habilidades
            - Ejercicio físico adaptado a cada edad"""
        ]
    }
}

def get_most_relevant_response(user_input: str) -> str:
    """
    Encuentra la respuesta más relevante basada en el input del usuario
    """
    user_input = user_input.lower()
    
    # Buscar coincidencias en los patrones
    for category, content in KNOWLEDGE_BASE.items():
        if any(pattern in user_input for pattern in content["patterns"]):
            return content["responses"][0]
    
    # Respuesta por defecto si no se encuentra coincidencia
    return """Puedo ayudarte con información sobre:
    - Qué es el Alzheimer
    - Síntomas de la enfermedad
    - Tratamientos disponibles
    - Cuidados necesarios
    - Medidas de prevención
    
    ¿Sobre qué tema te gustaría saber más?"""