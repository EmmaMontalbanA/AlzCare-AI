import pandas as pd
from modelos import AlzheimerDetector


# Model selection and evaluation with LazyPredict
from lazypredict.Supervised import LazyClassifier

# Machine Learning models
from sklearn.ensemble import VotingClassifier, BaggingClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, f1_score
import joblib

# Rutas de los archivos (ajusta según tu estructura de directorios)
data_paths = {
    'X_train': r'C:\F5_Proyectos\AlzCare-AI\AlzCare-AI\AlzCare-AI\data\processed\X_train_resampled.csv',
    'y_train': r'C:\F5_Proyectos\AlzCare-AI\AlzCare-AI\AlzCare-AI\data\processed\y_train_resampled.csv',
    'X_test': r'C:\F5_Proyectos\AlzCare-AI\AlzCare-AI\AlzCare-AI\data\processed\X_test.csv',
    'y_test': r'C:\F5_Proyectos\AlzCare-AI\AlzCare-AI\AlzCare-AI\data\processed\y_test.csv'
}

# Cargar datos
detector = AlzheimerDetector()

# Cargar DataFrames
X_train = pd.read_csv(data_paths['X_train'])
y_train = pd.read_csv(data_paths['y_train'])
X_test = pd.read_csv(data_paths['X_test'])
y_test = pd.read_csv(data_paths['y_test'])

# Establecer datos de entrenamiento y prueba
detector.set_train_test_data(X_train, X_test, y_train, y_test)

# Ejecutar LazyPredict
print("Resultados de LazyPredict:")
lazypredict_results = detector.run_lazypredict()
print(lazypredict_results)

# Crear modelo de ensamble
print("\nCreando Modelo de Ensamble...")
ensemble_model, metrics = detector.create_ensemble_model()

# Crear red neuronal
print("\nCreando Red Neuronal...")
nn_model, metrics = detector.create_neural_network()