import os
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, Any
from pathlib import Path
import joblib
from datetime import datetime

# Type aliases
ModelType = Any  # Could be more specific based on your needs
MetricsType = Dict[str, float]

@dataclass
class ModelConfig:
    """Configuration for model training and evaluation"""
    name: str
    random_state: int = 42
    batch_size: int = 32
    epochs: int = 100
    validation_split: float = 0.2
    early_stopping_patience: int = 10

class ModelMetrics:
    """Handle model metrics calculation and storage"""
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.metrics_history = pd.DataFrame()
        self.metrics_dir = output_dir / 'metrics'
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray) -> MetricsType:
        """Calculate model performance metrics"""
        from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                                roc_auc_score, f1_score, recall_score)
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
            'roc_auc': roc_auc_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred)
        }

    def save(self, metrics: MetricsType, model_name: str, overfitting: float):
        """Save metrics to CSV files"""
        metrics_dict = {'model_name': model_name, **metrics,  'overfitting': overfitting}
        metrics_df = pd.DataFrame([metrics_dict])
        
        self.metrics_history = pd.concat([self.metrics_history, metrics_df], ignore_index=True)
        self.metrics_history.to_csv(self.metrics_dir / 'all_models_metrics.csv', index=False)
        metrics_df.to_csv(self.metrics_dir / f'{model_name}.csv', index=False)

class Visualizer:
    """Handle model visualization tasks"""
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.graph_dir = output_dir / 'graph'
        self.graph_dir.mkdir(parents=True, exist_ok=True)

    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                            model_name: str):
        """Plot and save confusion matrix"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import confusion_matrix
        
        plt.figure(figsize=(8,6))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, cmap='Blues')
        plt.title(f'{model_name} Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(self.graph_dir / f'{model_name}_confusion_matrix.png')
        plt.close()

    def plot_roc_curve(self, y_true: np.ndarray, y_pred: np.ndarray, 
                    model_name: str):
        """Plot and save ROC curve"""
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve, roc_auc_score
        
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        plt.figure(figsize=(8,6))
        plt.plot(fpr, tpr, 
                label=f'ROC curve (area = {roc_auc_score(y_true, y_pred):.2f})')
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{model_name} ROC Curve')
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig(self.graph_dir / f'{model_name}_roc_curve.png')
        plt.close()

    def plot_training_history(self, history: Any, model_name: str):
        """Plot and save neural network training history"""
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        
        plt.tight_layout()
        plt.savefig(self.graph_dir / f'{model_name}_training_history.png')
        plt.close()

class ModelFactory:
    """Factory class for creating different types of models"""
    @staticmethod
    def create_ensemble():
        """Create ensemble model"""
        from sklearn.ensemble import VotingClassifier, BaggingClassifier, RandomForestClassifier
        from lightgbm import LGBMClassifier
        from xgboost import XGBClassifier
        
        base_models = {
            'LightGBM': LGBMClassifier(
                random_state=42,
                n_estimators=100,
                learning_rate=0.1),
            'RandomForest': RandomForestClassifier(
                n_estimators=100,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                max_features='sqrt',
                random_state=42),
            'Bagging': BaggingClassifier(
                base_estimator=LGBMClassifier(), 
                n_estimators=10, 
                random_state=42
            )
        }
        
        return VotingClassifier(
            estimators=[(name, model) for name, model in base_models.items()],
            voting='soft',  # Cambiado a 'soft' para usar probabilidades
            weights=[1] * len(base_models)  # Pesos iguales para todos los modelos
        ), base_models

    @staticmethod
    def create_neural_network(input_shape: Tuple[int, ...]):
        """Create neural network model"""
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
        from tensorflow.keras import regularizers
        
        model = Sequential([
            Dense(64, activation='relu', 
                input_shape=input_shape,
                kernel_regularizer=regularizers.l2(0.01)),
            BatchNormalization(),
            Dropout(0.5),
            Dense(32, activation='relu', 
                kernel_regularizer=regularizers.l2(0.01)),
            BatchNormalization(),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model

class AlzheimerDetector:
    """Main class for Alzheimer's detection"""
    def __init__(self, output_dir: str = './'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics = ModelMetrics(self.output_dir)
        self.visualizer = Visualizer(self.output_dir)
        self.model_factory = ModelFactory()
        
        self.X_train: Optional[np.ndarray] = None
        self.X_test: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.y_test: Optional[np.ndarray] = None

    def set_train_test_data(self, X_train, X_test, y_train, y_test):
        """Set training and test data"""
        self.X_train = self._ensure_numpy(X_train)
        self.X_test = self._ensure_numpy(X_test)
        self.y_train = self._ensure_numpy(y_train).ravel()
        self.y_test = self._ensure_numpy(y_test).ravel()

    def run_lazypredict(self):
        """Run LazyPredict to find best models"""
        from lazypredict.Supervised import LazyClassifier
        
        clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
        return clf.fit(self.X_train, self.X_test, self.y_train, self.y_test)

    def create_ensemble_model(self) -> Tuple[ModelType, MetricsType]:
        """Create and evaluate ensemble model"""
        ensemble, base_models = self.model_factory.create_ensemble()
        
        # Train and evaluate base models
        for name, model in base_models.items():
            self._train_and_evaluate_model(model, ModelConfig(name=name))
        
        # Train and evaluate ensemble
        return self._train_and_evaluate_model(
            ensemble, 
            ModelConfig(name='Ensemble')
        )

    def create_neural_network(self) -> Tuple[ModelType, MetricsType]:
        """Create and evaluate neural network model"""
        from tensorflow.keras.callbacks import EarlyStopping
        
        config = ModelConfig(name='NeuralNetwork')
        model = self.model_factory.create_neural_network(
            (self.X_train.shape[1],)
        )
        
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=config.early_stopping_patience,
            restore_best_weights=True
        )
        
        history = model.fit(
            self.X_train, self.y_train,
            epochs=config.epochs,
            batch_size=config.batch_size,
            validation_split=config.validation_split,
            callbacks=[early_stopping],
            verbose=1
        )
        
        train_score = model.evaluate(self.X_train, self.y_train)[1]
        y_pred = (model.predict(self.X_test) > 0.5).astype('int32')
        
        metrics = self.metrics.calculate(self.y_test, y_pred)
        overfitting = abs(train_score - metrics['accuracy'])
        
        self.metrics.save(metrics, config.name, overfitting)
        self.visualizer.plot_confusion_matrix(self.y_test, y_pred, config.name)
        self.visualizer.plot_roc_curve(self.y_test, y_pred, config.name)
        self.visualizer.plot_training_history(history, config.name)
        
        model.save(self.output_dir / f'{config.name}.h5')
        return model, metrics

    def _train_and_evaluate_model(
        self, 
        model: ModelType, 
        config: ModelConfig
    ) -> Tuple[ModelType, MetricsType]:
        """Train and evaluate a model"""
        model.fit(self.X_train, self.y_train)
        
        train_score = model.score(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        
        metrics = self.metrics.calculate(self.y_test, y_pred)
        overfitting = abs(train_score - metrics['accuracy'])
        
        self.metrics.save(metrics, config.name, overfitting)
        self.visualizer.plot_confusion_matrix(self.y_test, y_pred, config.name)
        self.visualizer.plot_roc_curve(self.y_test, y_pred, config.name)
        
        joblib.dump(model, self.output_dir / f'{config.name}.pkl')
        return model, metrics

    @staticmethod
    def _ensure_numpy(data: Any) -> np.ndarray:
        """Convert input data to numpy array"""
        if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
            return data.values
        return np.array(data)