#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔮 Script de Predicción de Churn - TelecomX

Este script permite hacer predicciones de churn usando el modelo entrenado.
"""

import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

def load_model_and_scaler():
    """Carga el modelo entrenado y el escalador"""
    try:
        # Cargar el mejor modelo (asumiendo que es Logistic Regression)
        model = joblib.load('best_model_Logistic_Regression.pkl')
        scaler = joblib.load('scaler.pkl')
        print("✅ Modelo y escalador cargados correctamente")
        return model, scaler
    except Exception as e:
        print(f"❌ Error cargando modelo: {e}")
        return None, None

def prepare_sample_data():
    """Prepara datos de ejemplo para predicción"""
    
    # Crear un DataFrame de ejemplo con diferentes tipos de clientes
    sample_data = {
        'customerID': ['CUST001', 'CUST002', 'CUST003', 'CUST004', 'CUST005'],
        'gender': ['Male', 'Female', 'Male', 'Female', 'Male'],
        'SeniorCitizen': ['No', 'Yes', 'No', 'No', 'Yes'],
        'Partner': ['Yes', 'No', 'Yes', 'No', 'Yes'],
        'Dependents': ['No', 'Yes', 'No', 'No', 'Yes'],
        'tenure': [1, 72, 12, 3, 24],
        'PhoneService': ['Yes', 'Yes', 'Yes', 'Yes', 'No'],
        'MultipleLines': ['No', 'Yes', 'Yes', 'No', 'No phone service'],
        'InternetService': ['DSL', 'Fiber optic', 'DSL', 'Fiber optic', 'No'],
        'OnlineSecurity': ['No', 'No', 'Yes', 'No', 'No internet service'],
        'OnlineBackup': ['Yes', 'No', 'No', 'No', 'No internet service'],
        'DeviceProtection': ['No', 'No', 'Yes', 'No', 'No internet service'],
        'TechSupport': ['No', 'No', 'Yes', 'No', 'No internet service'],
        'StreamingTV': ['No', 'Yes', 'No', 'Yes', 'No internet service'],
        'StreamingMovies': ['No', 'Yes', 'No', 'Yes', 'No internet service'],
        'Contract': ['Month-to-month', 'Two year', 'One year', 'Month-to-month', 'One year'],
        'PaperlessBilling': ['Yes', 'No', 'Yes', 'Yes', 'No'],
        'PaymentMethod': ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Electronic check', 'Mailed check'],
        'Charges.Monthly': [29.85, 89.10, 56.95, 79.85, 25.75],
        'Charges.Total': [29.85, 6414.20, 683.40, 239.55, 618.00]
    }
    
    return pd.DataFrame(sample_data)

def preprocess_data(df, scaler):
    """Preprocesa los datos para predicción"""
    
    # Eliminar customerID
    df_ml = df.drop(['customerID'], axis=1)
    
    # Identificar variables categóricas
    categorical_columns = df_ml.select_dtypes(include=['object']).columns.tolist()
    
    # One-Hot Encoding
    df_encoded = pd.get_dummies(df_ml, columns=categorical_columns, drop_first=True)
    
    # Cargar el dataset original para obtener todas las columnas esperadas
    try:
        original_data = pd.read_csv('telecom_data_processed_for_ml.csv')
        original_features = original_data.drop('Churn', axis=1).columns.tolist()
        
        # Agregar columnas faltantes con valores 0
        for col in original_features:
            if col not in df_encoded.columns:
                df_encoded[col] = 0
        
        # Reordenar columnas para que coincidan con el entrenamiento
        df_encoded = df_encoded[original_features]
        
    except Exception as e:
        print(f"⚠️  Advertencia: No se pudo cargar el dataset original: {e}")
    
    # Normalizar variables numéricas
    numeric_features = df_encoded.select_dtypes(include=[np.number]).columns.tolist()
    df_encoded[numeric_features] = scaler.transform(df_encoded[numeric_features])
    
    return df_encoded

def predict_churn(model, scaler, customer_data):
    """Hace predicciones de churn"""
    
    # Preprocesar datos
    processed_data = preprocess_data(customer_data, scaler)
    
    # Hacer predicciones
    predictions = model.predict(processed_data)
    probabilities = model.predict_proba(processed_data)[:, 1]
    
    # Crear DataFrame con resultados
    results = pd.DataFrame({
        'customerID': customer_data['customerID'],
        'predicted_churn': predictions,
        'churn_probability': probabilities,
        'risk_level': ['Alto' if p > 0.7 else 'Medio' if p > 0.4 else 'Bajo' for p in probabilities]
    })
    
    return results

def analyze_customer_risk(results):
    """Analiza el riesgo de churn de los clientes"""
    
    print("\n" + "="*60)
    print("🔮 ANÁLISIS DE RIESGO DE CHURN")
    print("="*60)
    
    for _, row in results.iterrows():
        print(f"\n👤 Cliente: {row['customerID']}")
        print(f"   Predicción: {'🔴 CHURN' if row['predicted_churn'] == 1 else '🟢 NO CHURN'}")
        print(f"   Probabilidad de Churn: {row['churn_probability']:.1%}")
        print(f"   Nivel de Riesgo: {row['risk_level']}")
        
        # Recomendaciones específicas
        if row['predicted_churn'] == 1:
            if row['churn_probability'] > 0.8:
                print("   ⚠️  ACCIÓN URGENTE: Cliente en alto riesgo de cancelación")
            elif row['churn_probability'] > 0.6:
                print("   🎯 ACCIÓN INMEDIATA: Implementar estrategia de retención")
            else:
                print("   📞 SEGUIMIENTO: Contactar para mejorar satisfacción")
        else:
            if row['churn_probability'] < 0.2:
                print("   ✅ CLIENTE LEAL: Mantener estrategia actual")
            else:
                print("   👀 MONITOREO: Cliente estable pero requiere atención")

def main():
    """Función principal"""
    
    print("🔮 Iniciando Predicción de Churn - TelecomX")
    print("="*60)
    
    # Cargar modelo y escalador
    model, scaler = load_model_and_scaler()
    if model is None or scaler is None:
        return
    
    # Preparar datos de ejemplo
    print("\n📊 Preparando datos de ejemplo...")
    sample_data = prepare_sample_data()
    
    print("\n📋 Datos de ejemplo:")
    print(sample_data[['customerID', 'tenure', 'Contract', 'Charges.Monthly', 'InternetService']].to_string(index=False))
    
    # Hacer predicciones
    print("\n🤖 Realizando predicciones...")
    results = predict_churn(model, scaler, sample_data)
    
    # Mostrar resultados
    print("\n📊 Resultados de Predicción:")
    print(results.to_string(index=False))
    
    # Analizar riesgo
    analyze_customer_risk(results)
    
    # Resumen estadístico
    print("\n" + "="*60)
    print("📈 RESUMEN ESTADÍSTICO")
    print("="*60)
    
    total_customers = len(results)
    predicted_churn = sum(results['predicted_churn'])
    avg_probability = results['churn_probability'].mean()
    
    print(f"Total de clientes analizados: {total_customers}")
    print(f"Clientes predichos con churn: {predicted_churn} ({predicted_churn/total_customers:.1%})")
    print(f"Probabilidad promedio de churn: {avg_probability:.1%}")
    
    # Distribución de niveles de riesgo
    risk_dist = results['risk_level'].value_counts()
    print(f"\nDistribución de niveles de riesgo:")
    for risk, count in risk_dist.items():
        print(f"   {risk}: {count} clientes ({count/total_customers:.1%})")
    
    print("\n" + "="*60)
    print("✅ Predicción completada exitosamente!")
    print("="*60)

if __name__ == "__main__":
    main()
