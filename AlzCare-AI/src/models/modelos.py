import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    recall_score, 
    confusion_matrix,
    balanced_accuracy_score,
    roc_auc_score,
    roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Importaciones de LazyPredict y modelos de Machine Learning
from lazypredict.Supervised import LazyClassifier
from sklearn.ensemble import VotingClassifier, BaggingClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

class AlzheimerDetector:
    def __init__(self, data_path=None):
        """
        Inicializar el detector de Alzheimer
        
        :param data_path: Ruta al archivo CSV (opcional)
        """
        self.data_path = data_path
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_data(self):
        """
        Cargar datos desde un archivo CSV
        
        :return: DataFrame con los datos
        """
        if self.data_path:
            self.data = pd.read_csv(self.data_path)
            return self.data
        return None

    def set_train_test_data(self, X_train, X_test, y_train, y_test):
        """
        Establecer datos de entrenamiento y prueba
        
        :param X_train: Características de entrenamiento
        :param X_test: Características de prueba
        :param y_train: Etiquetas de entrenamiento
        :param y_test: Etiquetas de prueba
        """
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def run_lazypredict(self):
        """
        Ejecutar LazyPredict para encontrar los mejores modelos
        
        :return: Resultados de LazyPredict
        """
        clf = LazyClassifier(
            verbose=0, 
            ignore_warnings=True, 
            custom_metric=None
        )
        models, predictions = clf.fit(
            self.X_train, 
            self.X_test, 
            self.y_train, 
            self.y_test
        )
        return models

    def create_ensemble_model(self):
        """
        Crear y evaluar un modelo de ensamble, incluyendo análisis de overfitting
        
        :return: Modelo de ensamble entrenado, métricas
        """
        # Definir los modelos base para el ensamble
        lgbm = LGBMClassifier(random_state=42)
        xgb = XGBClassifier(random_state=42)
        bagging = BaggingClassifier(
            base_estimator=LGBMClassifier(), 
            n_estimators=10, 
            random_state=42
        )

        # Crear un clasificador de votación (ensamble)
        ensemble = VotingClassifier(
            estimators=[
                ('lgbm', lgbm),
                ('xgb', xgb),
                ('bagging', bagging)
            ], 
            voting='hard'
        )

        # Evaluar overfitting de cada modelo base
        print("\nAnálisis de Overfitting de Modelos Base:")
        base_models = {
            'LightGBM': LGBMClassifier(random_state=42),
            'XGBoost': XGBClassifier(random_state=42),
            'Bagging': BaggingClassifier(base_estimator=LGBMClassifier(), n_estimators=10, random_state=42)
        }
        
        for name, model in base_models.items():
            # Entrenar el modelo
            model.fit(self.X_train, 
                    self.y_train.values.ravel() if isinstance(self.y_train, pd.DataFrame) else self.y_train)
            
            # Calcular puntajes de entrenamiento y prueba
            train_score = model.score(
                self.X_train, 
                self.y_train.values.ravel() if isinstance(self.y_train, pd.DataFrame) else self.y_train
            )
            test_score = model.score(
                self.X_test, 
                self.y_test.values.ravel() if isinstance(self.y_test, pd.DataFrame) else self.y_test
            )
            
            # Imprimir resultados de overfitting
            print(f"\n{name} Overfitting Analysis:")
            print(f"Train Score: {train_score:.2f}")
            print(f"Test Score: {test_score:.2f}")
            
            # Calcular y mostrar diferencia de rendimiento
            performance_diff = abs(train_score - test_score)
            print(f"Performance Difference: {performance_diff:.2f}")
            
            if performance_diff < 0.05:
                print(f"{name} has less than 5% overfitting.")
            else:
                print(f"{name} has more than 5% overfitting.")

        # Evaluar overfitting del modelo de ensamble
        print("\nEnsemble Overfitting Analysis:")
        
        # Entrenar el ensamble
        ensemble.fit(
            self.X_train, 
            self.y_train.values.ravel() if isinstance(self.y_train, pd.DataFrame) else self.y_train
        )

        # Calcular puntajes de entrenamiento y prueba para el ensamble
        train_score = ensemble.score(
            self.X_train, 
            self.y_train.values.ravel() if isinstance(self.y_train, pd.DataFrame) else self.y_train
        )
        test_score = ensemble.score(
            self.X_test, 
            self.y_test.values.ravel() if isinstance(self.y_test, pd.DataFrame) else self.y_test
        )

        print(f"Ensemble Train Score: {train_score:.2f}")
        print(f"Ensemble Test Score: {test_score:.2f}")
        
        # Calcular y mostrar diferencia de rendimiento del ensamble
        ensemble_performance_diff = abs(train_score - test_score)
        print(f"Ensemble Performance Difference: {ensemble_performance_diff:.2f}")
        
        if ensemble_performance_diff < 0.05:
            print("Ensemble has less than 5% overfitting.")
        else:
            print("Ensemble has more than 5% overfitting.")

        # Predecir en el conjunto de prueba
        y_pred = ensemble.predict(self.X_test)

        # Calcular métricas de rendimiento
        metrics = self._calculate_metrics(y_pred)

        # Guardar las métricas en una tabla
        self._save_metrics(metrics)
        
        # Visualizar matriz de confusión
        self._plot_confusion_matrix(y_pred)
        
        # Visualizar y guardar la curva ROC
        self._plot_roc_curve(y_pred)
        
        # Guardar el modelo
        #joblib.dump(ensemble, 'ensemble_model.joblib')
        # Guardar el modelo
        self._save_model(ensemble)

        return ensemble, metrics    
    
    def _calculate_metrics(self, y_pred):
        """
        Calcular métricas de rendimiento
        
        :param y_pred: Predicciones del modelo
        :return: Diccionario con métricas
        """
        metrics = {
            'accuracy': accuracy_score(self.y_test, y_pred),
            'balanced_accuracy': balanced_accuracy_score(self.y_test, y_pred),
            'roc_auc': roc_auc_score(self.y_test, y_pred),
            'f1_score': f1_score(self.y_test, y_pred),
            'recall': recall_score(self.y_test, y_pred)
        }

        # Imprimir métricas
        print("Métricas del Modelo de Ensamble:")
        for metric, value in metrics.items():
            print(f'{metric.replace("_", " ").title()}: {value:.4f}')

        return metrics

    def _plot_confusion_matrix(self, y_pred):
        """
        Graficar la matriz de confusión
        
        :param y_pred: Predicciones del modelo
        """
        cm = confusion_matrix(self.y_test, y_pred)
        plt.figure(figsize=(8,6))
        sns.heatmap(cm, annot=True, cmap='Blues')
        plt.title('Ensemble Model Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        
        # Crear directorio 'models/graph' si no existe
        os.makedirs('graph', exist_ok=True)
        
        # Guardar la matriz de confusión
        plt.savefig('graph/confusion_matrix.png')
        plt.close()
        
    def _plot_roc_curve(self, y_pred):
        """
        Graficar la curva ROC y guardar la imagen
        
        :param y_pred: Predicciones del modelo
        """
        fpr, tpr, _ = roc_curve(self.y_test, y_pred)
        plt.figure(figsize=(8,6))
        plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc_score(self.y_test, y_pred))
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc='lower right')
        plt.tight_layout()
        
        # Guardar la curva ROC
        plt.savefig('graph/roc_curve.png')
        plt.close()

    def _save_metrics(self, metrics):
        """
        Guardar las métricas en un archivo CSV
        
        :param metrics: Diccionario con las métricas
        """
        # Crear directorio 'models/metric' si no existe
        os.makedirs('metrics', exist_ok=True)
        
        # Convertir métricas en DataFrame y guardar
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv('metrics/metrics.csv', index=False)

    def _save_model(self, model):
        """
        Guardar el modelo entrenado en un archivo
        
        :param model: Modelo entrenado
        """

        # Guardar el modelo
        joblib.dump(model, 'alzheimer_detector.pkl')

