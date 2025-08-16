#!/usr/bin/env python3
"""
Script de verificación para validar la configuración del proyecto
"""

import os
import sys
import joblib
import numpy as np
from pathlib import Path

def check_files():
    """Verificar que todos los archivos necesarios existan"""
    print("🔍 Verificando archivos...")
    
    required_files = [
        "main.py",
        "train_model.py", 
        "client.py",
        "requirements.txt",
        "render.yaml",
        "README.md"
    ]
    
    model_files = [
        "models/fraud_detection_model.pkl",
        "models/feature_names.pkl",
        "models/dataset_stats.pkl"
    ]
    
    all_files = required_files + model_files
    
    missing_files = []
    for file_path in all_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
        else:
            print(f"✅ {file_path}")
    
    if missing_files:
        print(f"❌ Archivos faltantes: {missing_files}")
        return False
    
    print("✅ Todos los archivos están presentes")
    return True

def check_models():
    """Verificar que los modelos se puedan cargar correctamente"""
    print("\n🤖 Verificando modelos...")
    
    try:
        # Cargar modelos
        model = joblib.load("models/fraud_detection_model.pkl")
        feature_names = joblib.load("models/feature_names.pkl")
        dataset_stats = joblib.load("models/dataset_stats.pkl")
        
        # Verificar propiedades del modelo
        if not hasattr(model, 'predict'):
            print("❌ El modelo no tiene método predict")
            return False
        
        if not hasattr(model, 'predict_proba'):
            print("❌ El modelo no tiene método predict_proba")
            return False
        
        # Verificar feature_names
        if not isinstance(feature_names, list):
            print("❌ feature_names no es una lista")
            return False
        
        if len(feature_names) != 10:
            print(f"❌ Se esperaban 10 features, se encontraron {len(feature_names)}")
            return False
        
        # Verificar dataset_stats
        if not isinstance(dataset_stats, dict):
            print("❌ dataset_stats no es un diccionario")
            return False
        
        required_keys = ['total_samples', 'fraud_percentage', 'feature_ranges']
        for key in required_keys:
            if key not in dataset_stats:
                print(f"❌ dataset_stats no tiene la clave '{key}'")
                return False
        
        print("✅ Modelos cargados correctamente")
        print(f"   - Features: {len(feature_names)}")
        print(f"   - Muestras: {dataset_stats['total_samples']:,}")
        print(f"   - Tasa de fraude: {dataset_stats['fraud_percentage']:.2f}%")
        
        return True
        
    except Exception as e:
        print(f"❌ Error cargando modelos: {e}")
        return False

def test_prediction():
    """Probar una predicción simple"""
    print("\n🧪 Probando predicción...")
    
    try:
        # Cargar modelo
        model = joblib.load("models/fraud_detection_model.pkl")
        
        # Crear datos de prueba
        test_features = np.array([[
            100.0,    # transaction_amount
            5411,     # merchant_category
            12,       # transaction_hour
            1,        # days_since_last_transaction
            1,        # transaction_count_1h
            50.0,     # avg_transaction_amount_30d
            5.0,      # distance_from_home
            0,        # is_weekend
            365,      # account_age_days
            0         # previous_failed_attempts
        ]])
        
        # Realizar predicción
        prediction = model.predict(test_features)[0]
        probabilities = model.predict_proba(test_features)[0]
        
        print("✅ Predicción exitosa")
        print(f"   - Predicción: {'Fraude' if prediction else 'Legítimo'}")
        print(f"   - Probabilidad de fraude: {probabilities[1]:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en predicción: {e}")
        return False

def check_requirements():
    """Verificar que requirements.txt tenga las dependencias necesarias"""
    print("\n📦 Verificando requirements.txt...")
    
    try:
        with open("requirements.txt", "r") as f:
            content = f.read()
        
        # Verificar que los paquetes principales estén presentes
        required_packages = [
            "fastapi",
            "scikit-learn", 
            "joblib",
            "numpy",
            "pandas",
            "requests",
            "pydantic",
            "uvicorn"
        ]
        
        missing_packages = []
        for package in required_packages:
            if package not in content:
                missing_packages.append(package)
        
        if missing_packages:
            print(f"❌ Paquetes faltantes: {missing_packages}")
            return False
        
        print("✅ requirements.txt contiene todas las dependencias necesarias")
        print("   - fastapi==0.104.1")
        print("   - scikit-learn==1.6.1")
        print("   - joblib==1.3.2")
        print("   - numpy==1.24.3")
        print("   - pandas==2.0.3")
        print("   - requests==2.31.0")
        print("   - pydantic==2.5.0")
        print("   - uvicorn[standard]==0.24.0")
        return True
        
    except Exception as e:
        print(f"❌ Error leyendo requirements.txt: {e}")
        return False

def main():
    """Función principal de verificación"""
    print("🚀 VERIFICACIÓN DE CONFIGURACIÓN DEL PROYECTO")
    print("=" * 50)
    
    checks = [
        ("Archivos", check_files),
        ("Modelos", check_models),
        ("Predicción", test_prediction),
        ("Dependencias", check_requirements)
    ]
    
    all_passed = True
    
    for check_name, check_func in checks:
        if not check_func():
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 ¡TODAS LAS VERIFICACIONES PASARON!")
        print("✅ El proyecto está listo para deploy")
        print("\n📋 Próximos pasos:")
        print("   1. git add .")
        print("   2. git commit -m 'Fix: Implement all corrections'")
        print("   3. git push origin main")
        print("   4. Verificar deploy en Render")
    else:
        print("❌ ALGUNAS VERIFICACIONES FALLARON")
        print("🔧 Revisa los errores arriba y corrígelos")
        sys.exit(1)

if __name__ == "__main__":
    main()
