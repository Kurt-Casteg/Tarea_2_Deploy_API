# 🚀 API de Detección de Fraude en Tiempo Real

API de Machine Learning escalable para detectar transacciones fraudulentas en tiempo real utilizando un modelo de Random Forest. Esta solución está diseñada para ser desplegada fácilmente en Render y ofrece una interfaz RESTful para integración con sistemas existentes.

## 🎯 Características

- **Detección en tiempo real**: Analiza transacciones individuales
- **Procesamiento por lotes**: Hasta 100 transacciones simultáneamente
- **Análisis de riesgo**: Identifica factores de riesgo específicos
- **Recomendaciones**: Sugiere acciones basadas en el nivel de riesgo
- **Documentación automática**: Swagger UI integrado
- **Manejo robusto de errores**: Validaciones completas y mensajes informativos

## 🚀 Inicio Rápido

### Requisitos Previos
- Python 3.9.23
- pip (gestor de paquetes de Python)
- Cuenta en [Render](https://render.com) para despliegue
- URL de la API en producción: (https://tarea-2-deploy-api-1.onrender.com)

### Instalación Local

1. Clona el repositorio:
   ```bash
   git clone [URL_DEL_REPOSITORIO]
   cd Tarea_2_Deploy_API
   ```

2. Crea y activa un entorno virtual (recomendado):
   ```bash
   python -m venv venv
   source venv/bin/activate  # En Windows: .\venv\Scripts\activate
   ```

3. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```

4. Ejecuta el servidor de desarrollo:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port $PORT --reload
   ```

5. Accede a la documentación interactiva en [http://localhost:8000/docs](http://localhost:8000/docs)

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

### Detect Fraud
#### Request Body schema: application/json

```
transaction_amount
required
number (Transaction Amount) [ 0.01 .. 50000 ]
Monto de la transacción en USD

merchant_category
required
integer (Merchant Category) [ 1000 .. 9999 ]
Código de categoría del comerciante (MCC)

transaction_hour
required
integer (Transaction Hour) [ 0 .. 23 ]
Hora de la transacción (0-23)

days_since_last_transaction
required
integer (Days Since Last Transaction) [ 0 .. 365 ]
Días desde la última transacción

transaction_count_1h
required
integer (Transaction Count 1H) [ 0 .. 50 ]
Número de transacciones en la última hora

avg_transaction_amount_30d
required
number (Avg Transaction Amount 30D) [ 0 .. 10000 ]
Monto promedio de transacciones en 30 días

distance_from_home
required
number (Distance From Home) [ 0 .. 20000 ]
Distancia desde ubicación habitual (km)

is_weekend
required
boolean (Is Weekend)
¿La transacción fue en fin de semana?

account_age_days
required
integer (Account Age Days) [ 1 .. 10000 ]
Antigüedad de la cuenta en días

previous_failed_attempts
required
integer (Previous Failed Attempts) [ 0 .. 20 ]
Intentos fallidos previos
```

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

## 🚀 Despliegue en Render

### Configuración Automática (Recomendada)

1. Haz fork de este repositorio
2. Crea una nueva cuenta en [Render](https://render.com) si aún no tienes una
3. Haz clic en "New" y selecciona "Web Service"
4. Conecta tu cuenta de GitHub/GitLab y selecciona el repositorio
5. Render detectará automáticamente la configuración del `render.yaml`

### Configuración Manual

Si necesitas configurar manualmente:

1. **Runtime**: Python 3
2. **Build Command**: `pip install -r requirements.txt`
3. **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
4. **Plan**: Free (o selecciona el plan que mejor se adapte a tus necesidades)
5. **Auto-Deploy**: Actívalo para despliegues automáticos en cada push

### Variables de Entorno

No se requieren variables de entorno adicionales para el funcionamiento básico.

## 📦 Estructura del Proyecto

```
Tarea_2_Deploy_API/
├── models/                    # Modelos pre-entrenados
│   ├── fraud_detection_model.pkl
│   ├── feature_names.pkl
│   └── dataset_stats.pkl
├── main.py                   # Aplicación FastAPI
├── client.py                 # Cliente de ejemplo
├── train_model.py            # Script para entrenar el modelo
├── requirements.txt          # Dependencias
└── render.yaml               # Configuración de Render
```

## 🤝 Contribución

1. Haz fork del proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Haz commit de tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Haz push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.

## 👏 Reconocimientos

- [FastAPI](https://fastapi.tiangolo.com/) - El framework web utilizado
- [Scikit-learn](https://scikit-learn.org/) - Para el modelo de Machine Learning
- [Render](https://render.com) - Por el hosting gratuito

---

Hecho con ❤️ para el Magíster de Data Science UDD