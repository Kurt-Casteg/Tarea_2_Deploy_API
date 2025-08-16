# API de Detección de Fraude

API de Machine Learning para detectar transacciones fraudulentas en tiempo real usando Random Forest.

## 🎯 Características

- **Detección en tiempo real**: Analiza transacciones individuales
- **Procesamiento por lotes**: Hasta 100 transacciones simultáneamente
- **Análisis de riesgo**: Identifica factores de riesgo específicos
- **Recomendaciones**: Sugiere acciones basadas en el nivel de riesgo
- **Documentación automática**: Swagger UI integrado
- **Manejo robusto de errores**: Validaciones completas y mensajes informativos

## 📊 Endpoints Disponibles

### Predicción Individual
```bash
POST /predict
```
Analiza una transacción individual y devuelve:
- Predicción de fraude (sí/no)
- Probabilidad de fraude
- Nivel de riesgo (BAJO/MEDIO/ALTO)
- Factores de riesgo identificados
- Recomendación de acción

### Procesamiento por Lotes
```bash
POST /batch-predict
```
Procesa hasta 100 transacciones simultáneamente.

### Información del Modelo
```bash
GET /features     # Información sobre las features
GET /stats        # Estadísticas del dataset
GET /sample       # Ejemplos de transacciones
GET /health       # Estado del servicio
```

## 🚀 Instalación y Uso

### Local
```bash
# Instalar dependencias
pip install -r requirements.txt

# Entrenar modelo (si es necesario)
python train_model.py

# Ejecutar servidor
uvicorn main:app --reload
```

### Render (Deploy Automático)
El proyecto está configurado para deploy automático en Render.

## 📝 Ejemplo de Uso

### Transacción Legítima
```json
{
  "transaction_amount": 45.67,
  "merchant_category": 5411,
  "transaction_hour": 14,
  "days_since_last_transaction": 1,
  "transaction_count_1h": 1,
  "avg_transaction_amount_30d": 52.30,
  "distance_from_home": 2.5,
  "is_weekend": false,
  "account_age_days": 547,
  "previous_failed_attempts": 0
}
```

### Transacción Sospechosa
```json
{
  "transaction_amount": 2500.00,
  "merchant_category": 5944,
  "transaction_hour": 3,
  "days_since_last_transaction": 45,
  "transaction_count_1h": 5,
  "avg_transaction_amount_30d": 75.20,
  "distance_from_home": 850.0,
  "is_weekend": true,
  "account_age_days": 30,
  "previous_failed_attempts": 3
}
```

## 🧪 Testing

Ejecuta el cliente de pruebas:
```bash
python client.py
```

## 📚 Documentación Interactiva

- **Swagger UI**: `https://tu-app.onrender.com/docs`
- **ReDoc**: `https://tu-app.onrender.com/redoc`

## 🏗️ Arquitectura

- **Framework**: FastAPI
- **Modelo ML**: Random Forest Classifier
- **Features**: 10 características de transacciones
- **Validación**: Pydantic con rangos específicos
- **Deploy**: Render (Python Web Service)

## 📊 Métricas del Modelo

- **Dataset**: 50,000 transacciones
- **Tasa de fraude**: ~1% (realista)
- **Features**: 10 características relevantes
- **Algoritmo**: Random Forest con class_weight='balanced'

## 🔒 Seguridad

- Validación de rangos para todos los campos
- Manejo de errores robusto
- Límites en procesamiento por lotes
- Timeouts configurados
- Validación de integridad del modelo

## 🛠️ Características Técnicas

### Validaciones Implementadas
- Verificación de existencia de archivos de modelo
- Validación de propiedades del modelo cargado
- Comprobación de dimensiones de features
- Validación de probabilidades del modelo
- Manejo diferenciado de errores (400 vs 500)

### Manejo de Errores
- Errores de validación (400 Bad Request)
- Errores internos del servidor (500 Internal Server Error)
- Mensajes de error descriptivos
- Health check con prueba del modelo

## 📞 Soporte

Para problemas o preguntas, revisa la documentación interactiva en `/docs`.

## 🔄 Deploy en Render

1. Conecta tu repositorio a Render
2. Configura como Web Service
3. Build Command: `pip install -r requirements.txt`
4. Start Command: `uvicorn main:app --host 0.0.0.0 --port $PORT`

El proyecto incluye `render.yaml` para configuración automática.