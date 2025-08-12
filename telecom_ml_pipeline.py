#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🤖 Pipeline de Machine Learning para Predicción de Churn - TelecomX

Este script implementa un pipeline completo de Machine Learning para predecir 
la fuga de clientes (churn) en TelecomX, siguiendo la guía paso a paso.

Autor: Asistente IA
Fecha: 2024
"""

import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# PASO 1: PREPARACIÓN DEL ENTORNO
# =============================================================================

print("🚀 Iniciando Pipeline de Machine Learning para TelecomX")
print("="*60)

# 1.1 Librerías Necesarias
print("\n📦 Importando librerías...")

# Manipulación de datos
import pandas as pd
import numpy as np
import requests

# Visualización
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocesamiento
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from pandas import get_dummies

# Modelos de Machine Learning
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# Métricas de evaluación
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_auc_score, roc_curve

# Tratamiento de desbalance
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# Configuración de visualización
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (10, 6)

print("✅ Librerías importadas correctamente")

# =============================================================================
# PASO 2: CARGA DE DATOS
# =============================================================================

print("\n📥 Cargando datos...")

# Opción 1: Cargar desde el CSV guardado
try:
    df = pd.read_csv('telecom_data_limpio.csv')
    print("✅ Datos cargados desde CSV")
except:
    # Opción 2: Procesar desde la fuente original
    print("📥 Descargando datos desde la fuente original...")
    url = "https://raw.githubusercontent.com/alura-cursos/challenge2-data-science-LATAM/refs/heads/main/TelecomX_Data.json"
    response = requests.get(url)
    data = response.json()
    
    # Normalizar JSON anidado
    from pandas import json_normalize
    df = json_normalize(data, sep='_')
    
    # Renombrar columnas
    df.rename(columns={
        'customer_gender': 'gender',
        'customer_SeniorCitizen': 'SeniorCitizen',
        'customer_Partner': 'Partner',
        'customer_Dependents': 'Dependents',
        'customer_tenure': 'tenure',
        'phone_PhoneService': 'PhoneService',
        'phone_MultipleLines': 'MultipleLines',
        'internet_InternetService': 'InternetService',
        'internet_OnlineSecurity': 'OnlineSecurity',
        'internet_OnlineBackup': 'OnlineBackup',
        'internet_DeviceProtection': 'DeviceProtection',
        'internet_TechSupport': 'TechSupport',
        'internet_StreamingTV': 'StreamingTV',
        'internet_StreamingMovies': 'StreamingMovies',
        'account_Contract': 'Contract',
        'account_PaperlessBilling': 'PaperlessBilling',
        'account_PaymentMethod': 'PaymentMethod',
        'account_Charges_Monthly': 'Charges.Monthly',
        'account_Charges_Total': 'Charges.Total'
    }, inplace=True)
    
    # Limpiar datos
    df['Charges.Total'] = pd.to_numeric(df['Charges.Total'], errors='coerce')
    df['tenure'] = pd.to_numeric(df['tenure'], errors='coerce')
    df['SeniorCitizen'] = df['SeniorCitizen'].map({0: 'No', 1: 'Yes'})
    df['Charges.Total'].fillna(0, inplace=True)
    df = df[df['Churn'].isin(['Yes', 'No'])]
    
    print("✅ Datos procesados desde fuente original")

print(f"📊 Dataset shape: {df.shape}")
print(f"📋 Columnas: {list(df.columns)}")

# =============================================================================
# PASO 3: PREPARACIÓN DE DATOS PARA ML
# =============================================================================

print("\n🔧 Preparando datos para Machine Learning...")

# 3.1 Eliminación de Columnas Irrelevantes
df_ml = df.drop(['customerID'], axis=1)
print(f"Dataset para ML: {df_ml.shape}")

# 3.2 Análisis de Balance de Clases
churn_distribution = df_ml['Churn'].value_counts()
churn_percentage = df_ml['Churn'].value_counts(normalize=True) * 100

print("\n=== DISTRIBUCIÓN DE CHURN ===")
print(f"No Churn: {churn_distribution['No']} ({churn_percentage['No']:.1f}%)")
print(f"Churn: {churn_distribution['Yes']} ({churn_percentage['Yes']:.1f}%)")

# Evaluar si hay desbalance significativo
imbalance_ratio = churn_distribution['No'] / churn_distribution['Yes']
print(f"Ratio de desbalance: {imbalance_ratio:.2f}:1")

if imbalance_ratio > 3:
    print("⚠️  DESBALANCE DETECTADO - Considerar técnicas de balanceo")
else:
    print("✅ Balance de clases aceptable")

# 3.3 Codificación de Variables Categóricas
categorical_columns = df_ml.select_dtypes(include=['object']).columns.tolist()
categorical_columns.remove('Churn')  # Excluir variable objetivo

print(f"\nVariables categóricas a codificar: {len(categorical_columns)}")

# One-Hot Encoding con pandas
df_encoded = pd.get_dummies(df_ml, columns=categorical_columns, drop_first=True)

print(f"Dimensiones después de encoding: {df_encoded.shape}")
print(f"Nuevas variables creadas: {df_encoded.shape[1] - df_ml.shape[1]}")

# Convertir variable objetivo a numérica
df_encoded['Churn'] = df_encoded['Churn'].map({'No': 0, 'Yes': 1})

# 3.4 Análisis de Correlación Avanzado
correlation_matrix = df_encoded.corr()
churn_correlations = correlation_matrix['Churn'].sort_values(key=abs, ascending=False)

print("\n=== TOP 10 VARIABLES MÁS CORRELACIONADAS CON CHURN ===")
for i, (var, corr) in enumerate(churn_correlations.head(11).items()):
    if var != 'Churn':  # Excluir la correlación consigo misma
        print(f"{i}: {var:<30} | Correlación: {corr:>6.3f}")

# =============================================================================
# PASO 4: DIVISIÓN DE DATOS Y NORMALIZACIÓN
# =============================================================================

print("\n🔧 Dividiendo datos y normalizando...")

# 4.1 Separación de Variables
X = df_encoded.drop('Churn', axis=1)
y = df_encoded['Churn']

print(f"Características (X): {X.shape}")
print(f"Variable objetivo (y): {y.shape}")

# 4.2 División Train/Test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.3,  # 70% entrenamiento, 30% prueba
    random_state=42,
    stratify=y  # Mantener proporción de clases
)

print(f"Entrenamiento: {X_train.shape[0]} muestras")
print(f"Prueba: {X_test.shape[0]} muestras")

# 4.3 Normalización de Datos
numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
scaler = StandardScaler()

X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[numeric_features] = scaler.fit_transform(X_train[numeric_features])
X_test_scaled[numeric_features] = scaler.transform(X_test[numeric_features])

print("✅ Normalización completada")

# =============================================================================
# PASO 5: ENTRENAMIENTO DE MODELOS
# =============================================================================

print("\n🤖 Entrenando modelos...")

# 5.1 Configuración de Modelos
models = {
    'Logistic_Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random_Forest': RandomForestClassifier(random_state=42, n_estimators=100),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Decision_Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
    'SVM': SVC(random_state=42, probability=True)
}

# Diccionario para almacenar resultados
results = {}
trained_models = {}

# 5.2 Función de Evaluación
def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    """Entrena un modelo y calcula métricas de evaluación"""
    print(f"🔄 Entrenando {name}...")
    
    # Entrenar modelo
    model.fit(X_train, y_train)
    
    # Predicciones
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Probabilidades (para ROC-AUC)
    try:
        y_test_proba = model.predict_proba(X_test)[:, 1]
    except:
        y_test_proba = y_test_pred  # Para modelos sin predict_proba
    
    # Métricas
    metrics = {
        'train_accuracy': accuracy_score(y_train, y_train_pred),
        'test_accuracy': accuracy_score(y_test, y_test_pred),
        'precision': precision_score(y_test, y_test_pred),
        'recall': recall_score(y_test, y_test_pred),
        'f1_score': f1_score(y_test, y_test_pred),
        'roc_auc': roc_auc_score(y_test, y_test_proba) if len(np.unique(y_test_proba)) > 1 else 0
    }
    
    # Calcular overfitting
    metrics['overfitting'] = metrics['train_accuracy'] - metrics['test_accuracy']
    
    print(f"✅ {name} entrenado")
    print(f"   Accuracy Test: {metrics['test_accuracy']:.3f}")
    print(f"   F1-Score: {metrics['f1_score']:.3f}")
    print(f"   Overfitting: {metrics['overfitting']:.3f}")
    
    return metrics, model

# 5.3 Entrenamiento y Evaluación
print("=== ENTRENAMIENTO CON DATOS NORMALIZADOS ===")
for name, model in models.items():
    if name in ['Logistic_Regression', 'KNN', 'SVM']:  # Modelos que necesitan normalización
        metrics, trained_model = evaluate_model(name, model, X_train_scaled, X_test_scaled, y_train, y_test)
    else:  # Modelos basados en árboles
        metrics, trained_model = evaluate_model(name, model, X_train, X_test, y_train, y_test)
    
    results[name] = metrics
    trained_models[name] = trained_model

print("\n✅ Entrenamiento de todos los modelos completado")

# =============================================================================
# PASO 6: EVALUACIÓN Y COMPARACIÓN DE MODELOS
# =============================================================================

print("\n📊 Evaluando y comparando modelos...")

# 6.1 Tabla Comparativa
results_df = pd.DataFrame(results).T
results_df = results_df.round(3)

print("=== COMPARACIÓN DE MODELOS ===")
print(results_df.to_string())

# Identificar mejor modelo
best_model_name = results_df['test_accuracy'].idxmax()
best_f1_model = results_df['f1_score'].idxmax()

print(f"\n🏆 Mejor modelo por Accuracy: {best_model_name} ({results_df.loc[best_model_name, 'test_accuracy']:.3f})")
print(f"🎯 Mejor modelo por F1-Score: {best_f1_model} ({results_df.loc[best_f1_model, 'f1_score']:.3f})")

# 6.2 Matriz de Confusión del Mejor Modelo
best_model = trained_models[best_f1_model]

# Hacer predicciones
if best_f1_model in ['Logistic_Regression', 'KNN', 'SVM']:
    y_pred = best_model.predict(X_test_scaled)
else:
    y_pred = best_model.predict(X_test)

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred)

print(f"\n=== MATRIZ DE CONFUSIÓN - {best_f1_model} ===")
print("Matriz de Confusión:")
print(cm)

# Reporte de clasificación
print(f"\n=== REPORTE DETALLADO - {best_f1_model} ===")
print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))

# =============================================================================
# PASO 7: ANÁLISIS DE IMPORTANCIA DE VARIABLES
# =============================================================================

print("\n🔍 Analizando importancia de variables...")

# 7.1 Feature Importance (Random Forest)
if 'Random_Forest' in trained_models:
    rf_model = trained_models['Random_Forest']
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("=== TOP 10 VARIABLES MÁS IMPORTANTES (Random Forest) ===")
    for i, row in feature_importance.head(10).iterrows():
        print(f"{i+1:2d}. {row['feature']:<30} | Importancia: {row['importance']:.3f}")

# 7.2 Coeficientes (Regresión Logística)
if 'Logistic_Regression' in trained_models:
    lr_model = trained_models['Logistic_Regression']
    coefficients = pd.DataFrame({
        'feature': X_train.columns,
        'coefficient': lr_model.coef_[0]
    })
    coefficients['abs_coefficient'] = abs(coefficients['coefficient'])
    coefficients = coefficients.sort_values('abs_coefficient', ascending=False)
    
    print("\n=== TOP 10 COEFICIENTES MÁS INFLUYENTES (Regresión Logística) ===")
    for i, row in coefficients.head(10).iterrows():
        direction = "🔴 Aumenta" if row['coefficient'] > 0 else "🟢 Disminuye"
        print(f"{i+1:2d}. {row['feature']:<30} | {direction} Churn: {row['coefficient']:>6.3f}")

# =============================================================================
# PASO 8: TRATAMIENTO DE DESBALANCE (OPCIONAL)
# =============================================================================

print("\n⚖️ Evaluando necesidad de balanceo...")

if imbalance_ratio > 3:
    print("=== APLICANDO SMOTE PARA BALANCEAR CLASES ===")
    
    # Aplicar SMOTE
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
    
    print(f"Antes de SMOTE: {y_train.value_counts().to_dict()}")
    print(f"Después de SMOTE: {pd.Series(y_train_balanced).value_counts().to_dict()}")
    
    # Re-entrenar mejor modelo con datos balanceados
    best_model_balanced = type(trained_models[best_f1_model])(random_state=42)
    best_model_balanced.fit(X_train_balanced, y_train_balanced)
    
    # Evaluar modelo balanceado
    if best_f1_model in ['Logistic_Regression', 'KNN', 'SVM']:
        y_pred_balanced = best_model_balanced.predict(X_test_scaled)
    else:
        y_pred_balanced = best_model_balanced.predict(X_test)
    
    print(f"\n=== COMPARACIÓN CON/SIN BALANCEO ===")
    print(f"F1-Score Original: {f1_score(y_test, y_pred):.3f}")
    print(f"F1-Score Balanceado: {f1_score(y_test, y_pred_balanced):.3f}")
    print(f"Recall Original: {recall_score(y_test, y_pred):.3f}")
    print(f"Recall Balanceado: {recall_score(y_test, y_pred_balanced):.3f}")
else:
    print("✅ No se requiere balanceo - las clases están equilibradas")

# =============================================================================
# PASO 9: INSIGHTS Y RECOMENDACIONES ESTRATÉGICAS
# =============================================================================

print("\n" + "="*60)
print("🎯 FACTORES CLAVE QUE INFLUYEN EN LA CANCELACIÓN")
print("="*60)

# Basado en el análisis anterior, identifica los factores más importantes
key_factors = [
    "Contract_Month-to-month",  # Contratos mensuales
    "tenure",                   # Antigüedad baja
    "Charges.Monthly",          # Cargos mensuales altos
    "InternetService_Fiber optic",  # Servicio de fibra óptica
    "PaymentMethod_Electronic check"  # Método de pago
]

print("\n📈 TOP FACTORES DE RIESGO:")
for i, factor in enumerate(key_factors, 1):
    if 'Random_Forest' in trained_models and factor in feature_importance['feature'].values:
        importance = feature_importance[feature_importance['feature'] == factor]['importance'].iloc[0]
        print(f"{i}. {factor:<35} | Importancia: {importance:.3f}")
    else:
        print(f"{i}. {factor}")

# Recomendaciones estratégicas
print("\n" + "="*60)
print("💡 RECOMENDACIONES ESTRATÉGICAS")
print("="*60)

recommendations = [
    "🎯 RETENCIÓN TEMPRANA: Implementar programa de seguimiento intensivo para clientes nuevos (primeros 6 meses)",
    "📋 CONTRATOS ANUALES: Incentivar migración de contratos mes-a-mes a anuales con descuentos atractivos",
    "💰 REVISIÓN DE PRECIOS: Analizar estructura de precios de fibra óptica vs satisfacción del cliente",
    "🔧 CALIDAD DE SERVICIO: Mejorar estabilidad y soporte técnico para servicios de fibra óptica",
    "💳 MÉTODOS DE PAGO: Promocionar métodos de pago más estables (débito automático vs cheque electrónico)",
    "🎁 PROGRAMAS DE LEALTAD: Crear incentivos progresivos basados en antigüedad del cliente",
    "📞 INTERVENCIÓN PROACTIVA: Identificar clientes en riesgo usando el modelo predictivo para intervención temprana"
]

for rec in recommendations:
    print(f"\n{rec}")

print(f"\n{'='*60}")
print(f"📊 MODELO RECOMENDADO: {best_f1_model}")
print(f"🎯 F1-Score: {results_df.loc[best_f1_model, 'f1_score']:.3f}")
print(f"🎯 Accuracy: {results_df.loc[best_f1_model, 'test_accuracy']:.3f}")
print(f"📈 ROC-AUC: {results_df.loc[best_f1_model, 'roc_auc']:.3f}")
print("="*60)

# =============================================================================
# PASO 10: GUARDAR RESULTADOS
# =============================================================================

print("\n💾 Guardando resultados...")

import joblib

# Guardar modelo y escalador
joblib.dump(trained_models[best_f1_model], f'best_model_{best_f1_model}.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Guardar resultados
results_df.to_csv('model_comparison_results.csv')
if 'Random_Forest' in trained_models:
    feature_importance.to_csv('feature_importance.csv', index=False)

# Guardar dataset procesado
df_encoded.to_csv('telecom_data_processed_for_ml.csv', index=False)

print("✅ Modelos y resultados guardados exitosamente:")
print(f"   - best_model_{best_f1_model}.pkl")
print(f"   - scaler.pkl")
print(f"   - model_comparison_results.csv")
if 'Random_Forest' in trained_models:
    print(f"   - feature_importance.csv")
print(f"   - telecom_data_processed_for_ml.csv")

# =============================================================================
# RESUMEN FINAL
# =============================================================================

print("\n" + "="*60)
print("🎯 RESUMEN DEL PIPELINE COMPLETO")
print("="*60)

print("\n✅ Checklist de Completion:")
checklist_items = [
    "Preparación de datos: Eliminación de columnas irrelevantes",
    "Codificación categórica: One-hot encoding aplicado",
    "Análisis de balance: Evaluación de desbalance de clases",
    "División de datos: Train/test split estratificado",
    "Normalización: StandardScaler para modelos apropiados",
    "Entrenamiento: 5 modelos diferentes entrenados",
    "Evaluación: Métricas completas calculadas",
    "Comparación: Identificación del mejor modelo",
    "Análisis de importancia: Variables más influyentes identificadas",
    "Recomendaciones: Estrategias basadas en insights",
    "Guardado: Modelos y resultados preservados"
]

for item in checklist_items:
    print(f"   ✅ {item}")

print(f"\n📊 Métricas Clave del Mejor Modelo ({best_f1_model}):")
print(f"   🎯 Accuracy: {results_df.loc[best_f1_model, 'test_accuracy']:.3f}")
print(f"   🎯 Precision: {results_df.loc[best_f1_model, 'precision']:.3f}")
print(f"   🎯 Recall: {results_df.loc[best_f1_model, 'recall']:.3f}")
print(f"   🎯 F1-Score: {results_df.loc[best_f1_model, 'f1_score']:.3f}")
print(f"   📈 ROC-AUC: {results_df.loc[best_f1_model, 'roc_auc']:.3f}")

print("\n🎯 Entregables Finales:")
deliverables = [
    "Script completo con todo el pipeline",
    "Modelo entrenado (.pkl file)",
    "Reporte de métricas comparativas",
    "Lista de variables importantes con interpretación",
    "Recomendaciones estratégicas basadas en resultados"
]

for deliverable in deliverables:
    print(f"   ✅ {deliverable}")

print("\n" + "="*60)
print("🚀 ¡Pipeline de Machine Learning completado exitosamente!")
print("="*60)
