from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
import joblib
import numpy as np
from typing import List, Dict
from datetime import datetime
import os

app = FastAPI(
    title="Fraud Detection API",
    description="Detecta transacciones fraudulentas en tiempo real usando Machine Learning",
    version="1.0.0"
)

# Cargar modelo y metadata con manejo de errores robusto
def load_models():
    """Cargar modelos con manejo de errores"""
    try:
        # Verificar que los archivos existan
        model_files = [
            "models/fraud_detection_model.pkl",
            "models/feature_names.pkl", 
            "models/dataset_stats.pkl"
        ]
        
        for file_path in model_files:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Archivo de modelo no encontrado: {file_path}")
        
        # Cargar modelos
        model = joblib.load("models/fraud_detection_model.pkl")
        feature_names = joblib.load("models/feature_names.pkl")
        dataset_stats = joblib.load("models/dataset_stats.pkl")
        
        # Validar que el modelo tenga las propiedades esperadas
        if not hasattr(model, 'predict'):
            raise ValueError("El modelo cargado no tiene método predict")
        if not hasattr(model, 'predict_proba'):
            raise ValueError("El modelo cargado no tiene método predict_proba")
        
        # Validar que feature_names sea una lista
        if not isinstance(feature_names, list):
            raise ValueError("feature_names debe ser una lista")
        
        # Validar que dataset_stats sea un diccionario
        if not isinstance(dataset_stats, dict):
            raise ValueError("dataset_stats debe ser un diccionario")
            
        return model, feature_names, dataset_stats
        
    except FileNotFoundError as e:
        raise RuntimeError(f"Error de archivo: {e}")
    except Exception as e:
        raise RuntimeError(f"Error cargando modelos: {e}")

# Cargar modelos al iniciar la aplicación
try:
    model, feature_names, dataset_stats = load_models()
    print("✅ Modelos cargados exitosamente")
except Exception as e:
    print(f"❌ Error cargando modelos: {e}")
    raise

class TransactionRequest(BaseModel):
    transaction_amount: float = Field(..., description="Monto de la transacción en USD", ge=0.01, le=50000)
    merchant_category: int = Field(..., description="Código de categoría del comerciante (MCC)", ge=1000, le=9999)
    transaction_hour: int = Field(..., description="Hora de la transacción (0-23)", ge=0, le=23)
    days_since_last_transaction: int = Field(..., description="Días desde la última transacción", ge=0, le=365)
    transaction_count_1h: int = Field(..., description="Número de transacciones en la última hora", ge=0, le=50)
    avg_transaction_amount_30d: float = Field(..., description="Monto promedio de transacciones en 30 días", ge=0, le=10000)
    distance_from_home: float = Field(..., description="Distancia desde ubicación habitual (km)", ge=0, le=20000)
    is_weekend: bool = Field(..., description="¿La transacción fue en fin de semana?")
    account_age_days: int = Field(..., description="Antigüedad de la cuenta en días", ge=1, le=10000)
    previous_failed_attempts: int = Field(..., description="Intentos fallidos previos", ge=0, le=20)
    
    @validator('transaction_amount')
    def validate_amount(cls, v):
        if v < 0.01 or v > 50000:
            raise ValueError('Monto debe estar entre 0.01 y 50000 USD')
        return v
    
    @validator('merchant_category')
    def validate_mcc(cls, v):
        if v < 1000 or v > 9999:
            raise ValueError('Código MCC debe estar entre 1000 y 9999')
        return v
    
    @validator('transaction_hour')
    def validate_hour(cls, v):
        if v < 0 or v > 23:
            raise ValueError('Hora debe estar entre 0 y 23')
        return v

class FraudPredictionResponse(BaseModel):
    is_fraud: bool
    fraud_probability: float
    risk_level: str
    risk_factors: List[str]
    recommendation: str
    model_confidence: float

def get_merchant_category_name(mcc: int) -> str:
    """Convertir código MCC a nombre legible"""
    mcc_categories = {
        5411: "Supermercados",
        5812: "Restaurantes",
        5541: "Gasolineras", 
        5999: "Tiendas varias",
        4111: "Transporte",
        5311: "Grandes almacenes",
        5944: "Joyerías",
        7011: "Hoteles",
        4900: "Servicios públicos",
        5200: "Construcción"
    }
    # Encontrar la categoría más cercana
    closest_mcc = min(mcc_categories.keys(), key=lambda x: abs(x - mcc))
    return mcc_categories.get(closest_mcc, "Comercio general")

def analyze_risk_factors(features: np.ndarray, feature_names: List[str]) -> List[str]:
    """Identificar factores de riesgo basado en las features"""
    risk_factors = []
    
    # Validar que features tenga la longitud correcta
    if len(features) != len(feature_names):
        return ["Error: Número de features no coincide"]
    
    # Índices de features (deben coincidir con el orden en train_model.py)
    amount = features[0]
    hour = features[2]
    days_since_last = features[3]
    tx_count_1h = features[4]
    distance = features[6]
    is_weekend = features[7]
    failed_attempts = features[9]
    
    if amount > 1000:
        risk_factors.append("Monto alto de transacción")
    
    if hour < 6 or hour > 22:
        risk_factors.append("Transacción en horario inusual")
    
    if days_since_last > 30:
        risk_factors.append("Largo tiempo sin actividad")
    
    if tx_count_1h > 3:
        risk_factors.append("Múltiples transacciones recientes")
        
    if distance > 100:
        risk_factors.append("Ubicación muy distante a la habitual")
        
    if failed_attempts > 2:
        risk_factors.append("Múltiples intentos fallidos")
    
    return risk_factors

def get_risk_level(probability: float) -> str:
    """Determinar nivel de riesgo"""
    if probability < 0.3:
        return "BAJO"
    elif probability < 0.7:
        return "MEDIO" 
    else:
        return "ALTO"

def get_recommendation(is_fraud: bool, risk_level: str, probability: float) -> str:
    """Generar recomendación de acción"""
    if is_fraud and risk_level == "ALTO":
        return "🚨 BLOQUEAR - Transacción de alto riesgo, requiere verificación inmediata"
    elif is_fraud and risk_level == "MEDIO":
        return "⚠️ REVISAR - Solicitar autenticación adicional (2FA, SMS)"
    elif risk_level == "MEDIO":
        return "⚡ MONITOREAR - Permitir pero aumentar monitoreo"
    else:
        return "✅ APROBAR - Transacción de bajo riesgo"

@app.get("/")
def root():
    return {
        "message": "Fraud Detection API",
        "model": "Random Forest Classifier",
        "version": "1.0.0",
        "fraud_rate_trained": f"{dataset_stats['fraud_percentage']:.2f}%"
    }

@app.get("/features")
def get_features():
    """Información sobre las features del modelo"""
    feature_info = {}
    for feature in feature_names:
        stats = dataset_stats['feature_ranges'][feature]
        feature_info[feature] = {
            "min_value": stats['min'],
            "max_value": stats['max'],
            "average": stats['mean']
        }
    return {"features": feature_info}

@app.get("/stats")
def get_model_stats():
    """Estadísticas del modelo y dataset"""
    return {
        "dataset_size": dataset_stats['total_samples'],
        "fraud_percentage": dataset_stats['fraud_percentage'],
        "model_type": "Random Forest",
        "features_count": len(feature_names)
    }

@app.post("/predict", response_model=FraudPredictionResponse)
def detect_fraud(request: TransactionRequest):
    try:
        # Convertir request a array numpy
        features = np.array([[
            request.transaction_amount,
            request.merchant_category,
            request.transaction_hour,
            request.days_since_last_transaction,
            request.transaction_count_1h,
            request.avg_transaction_amount_30d,
            request.distance_from_home,
            int(request.is_weekend),
            request.account_age_days,
            request.previous_failed_attempts
        ]])
        
        # Validar que features tenga la forma correcta
        if features.shape[1] != len(feature_names):
            raise ValueError(f"Se esperaban {len(feature_names)} features, se recibieron {features.shape[1]}")
        
        # Realizar predicción con manejo de errores
        try:
            prediction = model.predict(features)[0]
        except Exception as e:
            raise ValueError(f"Error en predicción del modelo: {e}")
        
        # Obtener probabilidades con validación
        try:
            probabilities = model.predict_proba(features)[0]
            if len(probabilities) != 2:
                raise ValueError("El modelo debe tener exactamente 2 clases (legítimo/fraude)")
        except Exception as e:
            raise ValueError(f"Error obteniendo probabilidades: {e}")
        
        fraud_probability = probabilities[1]  # Probabilidad de fraude
        
        # Análisis adicional
        risk_level = get_risk_level(fraud_probability)
        risk_factors = analyze_risk_factors(features[0], feature_names)
        recommendation = get_recommendation(bool(prediction), risk_level, fraud_probability)
        
        # Confianza del modelo (basada en la diferencia entre probabilidades)
        model_confidence = abs(probabilities[1] - probabilities[0])
        
        return FraudPredictionResponse(
            is_fraud=bool(prediction),
            fraud_probability=round(fraud_probability, 4),
            risk_level=risk_level,
            risk_factors=risk_factors if risk_factors else ["Sin factores de riesgo identificados"],
            recommendation=recommendation,
            model_confidence=round(model_confidence, 4)
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Error de validación: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno del servidor: {str(e)}")

@app.post("/batch-predict")
def batch_fraud_detection(transactions: List[TransactionRequest]):
    """Procesar múltiples transacciones de una vez"""
    if len(transactions) > 100:
        raise HTTPException(status_code=400, detail="Máximo 100 transacciones por lote")
    
    results = []
    for i, transaction in enumerate(transactions):
        try:
            result = detect_fraud(transaction)
            results.append({
                "transaction_id": i + 1,
                "result": result
            })
        except HTTPException as e:
            results.append({
                "transaction_id": i + 1,
                "error": e.detail
            })
        except Exception as e:
            results.append({
                "transaction_id": i + 1,
                "error": f"Error inesperado: {str(e)}"
            })
    
    return {"batch_results": results, "processed_count": len(results)}

@app.get("/sample")
def get_sample_transactions():
    """Devuelve transacciones de ejemplo para probar la API"""
    return {
        "legitimate_transaction": {
            "transaction_amount": 45.67,
            "merchant_category": 5411,
            "transaction_hour": 14,
            "days_since_last_transaction": 1,
            "transaction_count_1h": 1,
            "avg_transaction_amount_30d": 52.30,
            "distance_from_home": 2.5,
            "is_weekend": False,
            "account_age_days": 547,
            "previous_failed_attempts": 0
        },
        "suspicious_transaction": {
            "transaction_amount": 2500.00,
            "merchant_category": 5944,
            "transaction_hour": 3,
            "days_since_last_transaction": 45,
            "transaction_count_1h": 5,
            "avg_transaction_amount_30d": 75.20,
            "distance_from_home": 850.0,
            "is_weekend": True,
            "account_age_days": 30,
            "previous_failed_attempts": 3
        }
    }

@app.get("/health")
def health_check():
    try:
        # Verificar que el modelo esté funcionando
        test_features = np.array([[100, 5411, 12, 1, 1, 50, 5, 0, 365, 0]])
        test_prediction = model.predict(test_features)
        test_probabilities = model.predict_proba(test_features)
        
        return {
            "status": "healthy",
            "model_loaded": True,
            "features_loaded": len(feature_names),
            "model_test": "passed",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "model_loaded": True,
            "features_loaded": len(feature_names),
            "model_test": f"failed: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }