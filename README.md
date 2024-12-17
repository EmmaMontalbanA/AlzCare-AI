# alzheimer_detection
Detección temprana de Alzheimer
Acerca del conjunto de datos
Este conjunto de datos contiene información de salud extensa para 2149 pacientes, cada uno identificado de forma única con un ID que va desde 4751 a 6900. El conjunto de datos incluye detalles demográficos, factores de estilo de vida, historial médico, mediciones clínicas, evaluaciones cognitivas y funcionales, síntomas y un diagnóstico de la enfermedad de Alzheimer. Los datos son ideales para investigadores y científicos de datos que buscan explorar los factores asociados con el Alzheimer, desarrollar modelos predictivos y realizar análisis estadísticos.

Tabla de contenido
Información para el paciente
Identificación del paciente
Datos demográficos
Factores del estilo de vida
Historial médico
Mediciones clínicas
Evaluaciones cognitivas y funcionales
Síntomas
Información de diagnóstico
Información confidencial
Información para el paciente
Identificación del paciente
PatientID : un identificador único asignado a cada paciente (4751 a 6900).
Datos demográficos
Edad : La edad de los pacientes oscila entre 60 y 90 años.
Género : Género de los pacientes, donde 0 representa masculino y 1 representa femenino.
Etnicidad : La etnicidad de los pacientes, codificada de la siguiente manera:
0: caucásico
1: Afroamericano
2: asiático
3: Otros
NivelEducación : El nivel de educación de los pacientes, codificado de la siguiente manera:
0: Ninguno
1: Escuela secundaria
2: Licenciatura
3: Superior
Factores del estilo de vida
IMC : Índice de Masa Corporal de los pacientes, que oscila entre 15 y 40.
Fumar : Estado de fumador, donde 0 indica No y 1 indica Sí.
ConsumoDeAlcohol : Consumo semanal de alcohol en unidades, que varía de 0 a 20.
ActividadFísica : Actividad física semanal en horas, de 0 a 10.
DietQuality : Puntuación de calidad de la dieta, que va de 0 a 10.
SleepQuality : Puntuación de la calidad del sueño, que va de 4 a 10.
Historial médico
HistorialFamiliaAlzheimer : Antecedentes familiares de enfermedad de Alzheimer, donde 0 indica No y 1 indica Sí.
EnfermedadCardiovascular : Presencia de enfermedad cardiovascular, donde 0 indica No y 1 indica Sí.
Diabetes : Presencia de diabetes, donde 0 indica No y 1 indica Sí.
Depresión : Presencia de depresión, donde 0 indica No y 1 indica Sí.
Lesión en la cabeza : antecedentes de lesión en la cabeza, donde 0 indica No y 1 indica Sí.
Hipertensión : Presencia de hipertensión, donde 0 indica No y 1 indica Sí.
Mediciones clínicas
Presión arterial sistólica : presión arterial sistólica, que varía entre 90 y 180 mmHg.
Presión arterial diastólica : presión arterial diastólica, que varía entre 60 y 120 mmHg.
ColesterolTotal : Niveles de colesterol total, que varían entre 150 y 300 mg/dL.
ColesterolLDL : Niveles de colesterol de lipoproteínas de baja densidad, que varían entre 50 y 200 mg/dL.
ColesterolHDL : Niveles de colesterol de lipoproteínas de alta densidad, que varían entre 20 y 100 mg/dL.
ColesterolTriglicéridos : Niveles de triglicéridos, que varían entre 50 y 400 mg/dL.
Evaluaciones cognitivas y funcionales
MMSE : Puntuación del Mini-Examen del Estado Mental, que varía de 0 a 30. Las puntuaciones más bajas indican deterioro cognitivo.
Evaluación funcional : Puntuación de evaluación funcional, que va de 0 a 10. Las puntuaciones más bajas indican un mayor deterioro.
Quejas de memoria : Presencia de quejas de memoria, donde 0 indica No y 1 indica Sí.
Problemas de conducta : Presencia de problemas de conducta, donde 0 indica No y 1 indica Sí.
ADL : Puntuación de actividades de la vida diaria, que varía de 0 a 10. Las puntuaciones más bajas indican un mayor deterioro.
Síntomas
Confusión : Presencia de confusión, donde 0 indica No y 1 indica Sí.
Desorientación : Presencia de desorientación, donde 0 indica No y 1 indica Sí.
PersonalityChanges : Presencia de cambios de personalidad, donde 0 indica No y 1 indica Sí.
Dificultad para completar tareas : Presencia de dificultad para completar tareas, donde 0 indica No y 1 indica Sí.
Olvido : Presencia de olvido, donde 0 indica No y 1 indica Sí.
Información de diagnóstico
Diagnóstico : Estado del diagnóstico de la enfermedad de Alzheimer, donde 0 indica No y 1 indica Sí.
Información confidencial
DoctorInCharge : Esta columna contiene información confidencial sobre el médico a cargo, con "XXXConfid" como valor para todos los pacientes.
Conclusión
Este conjunto de datos ofrece información detallada sobre los factores asociados con la enfermedad de Alzheimer, incluidas variables demográficas, de estilo de vida, médicas, cognitivas y funcionales. Es ideal para desarrollar modelos predictivos, realizar análisis estadísticos y explorar la compleja interacción de los factores que contribuyen a la enfermedad de Alzheimer.