from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
import numpy as np
import pandas as pd

def create_realistic_fraud_dataset():
    """Crea un dataset realista de detección de fraude"""
    
    # Generar dataset base
    X, y = make_classification(
        n_samples=50000,
        n_features=10,
        n_informative=8,
        n_redundant=2,
        n_classes=2,
        weights=[0.99, 0.01],  # 99% legítimas, 1% fraudulentas
        flip_y=0.005,  # 0.5% de ruido
        random_state=42
    )
    
    # Nombres de features realistas
    feature_names = [
        'transaction_amount',
        'merchant_category', 
        'transaction_hour',
        'days_since_last_transaction',
        'transaction_count_1h',
        'avg_transaction_amount_30d',
        'distance_from_home',
        'is_weekend',
        'account_age_days',
        'previous_failed_attempts'
    ]
    
    # Crear DataFrame para manipulación
    df = pd.DataFrame(X, columns=feature_names)
    
    # Hacer features más realistas
    df['transaction_amount'] = np.abs(df['transaction_amount'] * 500 + 100)  # $100-3000
    df['merchant_category'] = np.abs(df['merchant_category'] * 1000 + 5000).astype(int)  # Códigos MCC
    df['transaction_hour'] = np.abs(df['transaction_hour'] * 12 + 6).astype(int) % 24  # 0-23 horas
    df['days_since_last_transaction'] = np.abs(df['days_since_last_transaction'] * 15).astype(int)  # 0-30 días
    df['transaction_count_1h'] = np.abs(df['transaction_count_1h'] * 5).astype(int)  # 0-10 transacciones
    df['avg_transaction_amount_30d'] = np.abs(df['avg_transaction_amount_30d'] * 300 + 50)  # $50-1000
    df['distance_from_home'] = np.abs(df['distance_from_home'] * 50)  # 0-200 km
    df['is_weekend'] = (df['is_weekend'] > 0).astype(int)  # 0 o 1
    df['account_age_days'] = np.abs(df['account_age_days'] * 1000 + 30).astype(int)  # 30-2000 días
    df['previous_failed_attempts'] = np.abs(df['previous_failed_attempts'] * 3).astype(int)  # 0-5 intentos
    
    return df.values, y, feature_names

def train_fraud_detection_model():
    # Crear dataset
    X, y, feature_names = create_realistic_fraud_dataset()
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Entrenar modelo
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        random_state=42,
        class_weight='balanced'  # Importante para dataset desbalanceado
    )
    model.fit(X_train, y_train)
    
    # Evaluar
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    print("=== EVALUACIÓN DEL MODELO ===")
    print(f"AUC-ROC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Importancia de features
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n=== IMPORTANCIA DE FEATURES ===")
    for _, row in feature_importance.head().iterrows():
        print(f"{row['feature']}: {row['importance']:.4f}")
    
    # Guardar modelo y metadata
    joblib.dump(model, 'models/fraud_detection_model.pkl')
    joblib.dump(feature_names, 'models/feature_names.pkl')
    
    # Estadísticas del dataset para documentación
    stats = {
        'total_samples': len(X),
        'fraud_percentage': (y.sum() / len(y)) * 100,
        'feature_ranges': {}
    }
    
    for i, feature in enumerate(feature_names):
        stats['feature_ranges'][feature] = {
            'min': float(X[:, i].min()),
            'max': float(X[:, i].max()),
            'mean': float(X[:, i].mean())
        }
    
    joblib.dump(stats, 'models/dataset_stats.pkl')
    
    print(f"\n=== DATASET INFO ===")
    print(f"Total samples: {stats['total_samples']:,}")
    print(f"Fraud percentage: {stats['fraud_percentage']:.2f}%")
    
    return model, feature_names, stats

if __name__ == "__main__":
    print("🚀 Entrenando modelo de detección de fraude...")
    train_fraud_detection_model()
    print("✅ Modelo entrenado y guardado exitosamente!")