# 🤖 Instructivo: Machine Learning para Predicción de Churn - TelecomX

## 📋 Introducción

Este instructivo te guiará paso a paso para desarrollar modelos predictivos de churn en TelecomX, basándote en el análisis exploratorio ya completado en la Parte 1.

---

## 🎯 Objetivos del Proyecto

- ✅ Preparar los datos para modelado predictivo
- ✅ Entrenar y evaluar múltiples modelos de clasificación
- ✅ Identificar los factores más importantes para la predicción
- ✅ Proporcionar recomendaciones estratégicas basadas en datos

---

## 📂 PASO 1: Preparación del Entorno

### 1.1 Librerías Necesarias

```python
# Manipulación de datos
import pandas as pd
import numpy as np

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
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
```

### 1.2 Carga de Datos

```python
# Opción 1: Si tienes el CSV guardado de la Parte 1
df = pd.read_csv('telecom_data_limpio.csv')

# Opción 2: Si necesitas procesar desde el notebook original
# Ejecuta las celdas de extracción y transformación del notebook
```

---

## 📊 PASO 2: Preparación de Datos para ML

### 2.1 Eliminación de Columnas Irrelevantes

```python
# Eliminar ID del cliente (no aporta valor predictivo)
df_ml = df.drop(['customerID'], axis=1)

print(f"Dataset original: {df.shape}")
print(f"Dataset para ML: {df_ml.shape}")
print(f"Columnas eliminadas: customerID")
```

### 2.2 Análisis de Balance de Clases

```python
# Verificar distribución de la variable objetivo
churn_distribution = df_ml['Churn'].value_counts()
churn_percentage = df_ml['Churn'].value_counts(normalize=True) * 100

print("=== DISTRIBUCIÓN DE CHURN ===")
print(f"No Churn: {churn_distribution['No']} ({churn_percentage['No']:.1f}%)")
print(f"Churn: {churn_distribution['Yes']} ({churn_percentage['Yes']:.1f}%)")

# Visualización
plt.figure(figsize=(8, 6))
sns.countplot(data=df_ml, x='Churn', palette=['lightgreen', 'lightcoral'])
plt.title('Distribución de Clases - Churn')
for i, v in enumerate(churn_distribution):
    plt.text(i, v + 50, f'{churn_percentage.iloc[i]:.1f}%', 
             ha='center', va='bottom', fontsize=12, fontweight='bold')
plt.show()

# Evaluar si hay desbalance significativo
imbalance_ratio = churn_distribution['No'] / churn_distribution['Yes']
print(f"\nRatio de desbalance: {imbalance_ratio:.2f}:1")

if imbalance_ratio > 3:
    print("⚠️  DESBALANCE DETECTADO - Considerar técnicas de balanceo")
else:
    print("✅ Balance de clases aceptable")
```

### 2.3 Codificación de Variables Categóricas

```python
# Identificar variables categóricas
categorical_columns = df_ml.select_dtypes(include=['object']).columns.tolist()
categorical_columns.remove('Churn')  # Excluir variable objetivo

print("Variables categóricas a codificar:", categorical_columns)

# Método 1: One-Hot Encoding con pandas
df_encoded = pd.get_dummies(df_ml, columns=categorical_columns, drop_first=True)

print(f"Dimensiones después de encoding: {df_encoded.shape}")
print(f"Nuevas variables creadas: {df_encoded.shape[1] - df_ml.shape[1]}")

# Convertir variable objetivo a numérica
df_encoded['Churn'] = df_encoded['Churn'].map({'No': 0, 'Yes': 1})

# Verificar el resultado
print("\nPrimeras columnas del dataset codificado:")
print(df_encoded.columns.tolist()[:10])
```

### 2.4 Análisis de Correlación Avanzado

```python
# Calcular matriz de correlación
correlation_matrix = df_encoded.corr()

# Correlaciones con la variable objetivo
churn_correlations = correlation_matrix['Churn'].sort_values(key=abs, ascending=False)

print("=== TOP 10 VARIABLES MÁS CORRELACIONADAS CON CHURN ===")
for i, (var, corr) in enumerate(churn_correlations.head(11).items()):
    if var != 'Churn':  # Excluir la correlación consigo misma
        print(f"{i}: {var:<30} | Correlación: {corr:>6.3f}")

# Visualización de correlaciones importantes
plt.figure(figsize=(12, 8))
top_corr = churn_correlations.drop('Churn').head(15)
sns.barplot(x=top_corr.values, y=top_corr.index, palette='RdYlBu_r')
plt.title('Top 15 Variables más Correlacionadas con Churn')
plt.xlabel('Correlación con Churn')
plt.axvline(x=0, color='black', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
```

---

## 🔧 PASO 3: División de Datos y Normalización

### 3.1 Separación de Variables

```python
# Separar características (X) y variable objetivo (y)
X = df_encoded.drop('Churn', axis=1)
y = df_encoded['Churn']

print(f"Características (X): {X.shape}")
print(f"Variable objetivo (y): {y.shape}")
print(f"Distribución de y: {y.value_counts().to_dict()}")
```

### 3.2 División Train/Test

```python
# División estratificada para mantener proporción de clases
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.3,  # 70% entrenamiento, 30% prueba
    random_state=42,
    stratify=y  # Mantener proporción de clases
)

print("=== DIVISIÓN DE DATOS ===")
print(f"Entrenamiento: {X_train.shape[0]} muestras")
print(f"Prueba: {X_test.shape[0]} muestras")
print(f"Distribución Churn en entrenamiento: {y_train.value_counts().to_dict()}")
print(f"Distribución Churn en prueba: {y_test.value_counts().to_dict()}")
```

### 3.3 Normalización de Datos

```python
# Identificar variables numéricas que necesitan normalización
numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
print(f"Variables numéricas a normalizar: {numeric_features}")

# Crear el escalador
scaler = StandardScaler()

# Ajustar el escalador solo con datos de entrenamiento
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[numeric_features] = scaler.fit_transform(X_train[numeric_features])
X_test_scaled[numeric_features] = scaler.transform(X_test[numeric_features])

print("✅ Normalización completada")

# Verificar normalización
print("\nEstadísticas después de normalización (variables numéricas):")
print(X_train_scaled[numeric_features].describe().round(3))
```

---

## 🤖 PASO 4: Entrenamiento de Modelos

### 4.1 Configuración de Modelos

```python
# Diccionario de modelos a entrenar
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
```

### 4.2 Entrenamiento y Evaluación

```python
def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    """
    Entrena un modelo y calcula métricas de evaluación
    """
    print(f"\n🔄 Entrenando {name}...")
    
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

# Entrenar modelos con datos normalizados
print("=== ENTRENAMIENTO CON DATOS NORMALIZADOS ===")
for name, model in models.items():
    if name in ['Logistic_Regression', 'KNN', 'SVM']:  # Modelos que necesitan normalización
        metrics, trained_model = evaluate_model(name, model, X_train_scaled, X_test_scaled, y_train, y_test)
    else:  # Modelos basados en árboles
        metrics, trained_model = evaluate_model(name, model, X_train, X_test, y_train, y_test)
    
    results[name] = metrics
    trained_models[name] = trained_model
```

---

## 📊 PASO 5: Evaluación y Comparación de Modelos

### 5.1 Tabla Comparativa

```python
# Crear DataFrame con resultados
results_df = pd.DataFrame(results).T
results_df = results_df.round(3)

print("=== COMPARACIÓN DE MODELOS ===")
print(results_df.to_string())

# Identificar mejor modelo
best_model_name = results_df['test_accuracy'].idxmax()
best_f1_model = results_df['f1_score'].idxmax()

print(f"\n🏆 Mejor modelo por Accuracy: {best_model_name} ({results_df.loc[best_model_name, 'test_accuracy']:.3f})")
print(f"🎯 Mejor modelo por F1-Score: {best_f1_model} ({results_df.loc[best_f1_model, 'f1_score']:.3f})")
```

### 5.2 Visualización de Resultados

```python
# Gráfico comparativo de métricas
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Comparación de Rendimiento de Modelos', fontsize=16)

# Accuracy
axes[0,0].bar(results_df.index, results_df['test_accuracy'], color='skyblue')
axes[0,0].set_title('Accuracy')
axes[0,0].set_ylabel('Score')
axes[0,0].tick_params(axis='x', rotation=45)

# F1-Score
axes[0,1].bar(results_df.index, results_df['f1_score'], color='lightgreen')
axes[0,1].set_title('F1-Score')
axes[0,1].set_ylabel('Score')
axes[0,1].tick_params(axis='x', rotation=45)

# Precision vs Recall
axes[1,0].scatter(results_df['precision'], results_df['recall'], s=100, alpha=0.7)
for i, model in enumerate(results_df.index):
    axes[1,0].annotate(model, (results_df.iloc[i]['precision'], results_df.iloc[i]['recall']))
axes[1,0].set_xlabel('Precision')
axes[1,0].set_ylabel('Recall')
axes[1,0].set_title('Precision vs Recall')

# Overfitting
colors = ['red' if x > 0.05 else 'green' for x in results_df['overfitting']]
axes[1,1].bar(results_df.index, results_df['overfitting'], color=colors, alpha=0.7)
axes[1,1].set_title('Overfitting (Train Acc - Test Acc)')
axes[1,1].set_ylabel('Diferencia')
axes[1,1].tick_params(axis='x', rotation=45)
axes[1,1].axhline(y=0.05, color='red', linestyle='--', alpha=0.5, label='Umbral Overfitting')

plt.tight_layout()
plt.show()
```

### 5.3 Matriz de Confusión del Mejor Modelo

```python
# Seleccionar el mejor modelo
best_model = trained_models[best_f1_model]

# Hacer predicciones
if best_f1_model in ['Logistic_Regression', 'KNN', 'SVM']:
    y_pred = best_model.predict(X_test_scaled)
else:
    y_pred = best_model.predict(X_test)

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['No Churn', 'Churn'], 
            yticklabels=['No Churn', 'Churn'])
plt.title(f'Matriz de Confusión - {best_f1_model}')
plt.ylabel('Valores Reales')
plt.xlabel('Predicciones')
plt.show()

# Reporte de clasificación
print(f"\n=== REPORTE DETALLADO - {best_f1_model} ===")
print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))
```

---

## 🔍 PASO 6: Análisis de Importancia de Variables

### 6.1 Feature Importance (Random Forest)

```python
if 'Random_Forest' in trained_models:
    rf_model = trained_models['Random_Forest']
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(data=feature_importance.head(15), y='feature', x='importance', palette='viridis')
    plt.title('Top 15 Variables más Importantes (Random Forest)')
    plt.xlabel('Importancia')
    plt.tight_layout()
    plt.show()
    
    print("=== TOP 10 VARIABLES MÁS IMPORTANTES ===")
    for i, row in feature_importance.head(10).iterrows():
        print(f"{row.name+1:2d}. {row['feature']:<30} | Importancia: {row['importance']:.3f}")
```

### 6.2 Coeficientes (Regresión Logística)

```python
if 'Logistic_Regression' in trained_models:
    lr_model = trained_models['Logistic_Regression']
    coefficients = pd.DataFrame({
        'feature': X_train.columns,
        'coefficient': lr_model.coef_[0]
    })
    coefficients['abs_coefficient'] = abs(coefficients['coefficient'])
    coefficients = coefficients.sort_values('abs_coefficient', ascending=False)
    
    plt.figure(figsize=(12, 8))
    top_coef = coefficients.head(15)
    colors = ['red' if x < 0 else 'blue' for x in top_coef['coefficient']]
    sns.barplot(data=top_coef, y='feature', x='coefficient', palette=colors)
    plt.title('Top 15 Coeficientes más Influyentes (Regresión Logística)')
    plt.xlabel('Coeficiente (+ aumenta Churn, - disminuye Churn)')
    plt.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()
    
    print("=== TOP 10 COEFICIENTES MÁS INFLUYENTES ===")
    for i, row in coefficients.head(10).iterrows():
        direction = "🔴 Aumenta" if row['coefficient'] > 0 else "🟢 Disminuye"
        print(f"{i+1:2d}. {row['feature']:<30} | {direction} Churn: {row['coefficient']:>6.3f}")
```

---

## 📋 PASO 7: Tratamiento de Desbalance (Opcional)

### 7.1 Aplicar SMOTE

```python
# Solo si detectamos desbalance significativo
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
```

---

## 📊 PASO 8: Insights y Recomendaciones Estratégicas

### 8.1 Análisis de Factores Clave

```python
print("="*60)
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
    if factor in feature_importance['feature'].values:
        importance = feature_importance[feature_importance['feature'] == factor]['importance'].iloc[0]
        print(f"{i}. {factor:<35} | Importancia: {importance:.3f}")

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
```

---

## 💾 PASO 9: Guardar Resultados

```python
# Guardar el mejor modelo
import joblib

# Guardar modelo y escalador
joblib.dump(trained_models[best_f1_model], f'best_model_{best_f1_model}.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Guardar resultados
results_df.to_csv('model_comparison_results.csv')
feature_importance.to_csv('feature_importance.csv', index=False)

# Guardar dataset procesado
df_encoded.to_csv('telecom_data_processed_for_ml.csv', index=False)

print("✅ Modelos y resultados guardados exitosamente:")
print(f"   - best_model_{best_f1_model}.pkl")
print(f"   - scaler.pkl")
print(f"   - model_comparison_results.csv")
print(f"   - feature_importance.csv")
print(f"   - telecom_data_processed_for_ml.csv")
```

---

## 🎯 Resumen del Pipeline Completo

### ✅ Checklist de Completion

- [ ] **Preparación de datos**: Eliminación de columnas irrelevantes
- [ ] **Codificación categórica**: One-hot encoding aplicado
- [ ] **Análisis de balance**: Evaluación de desbalance de clases
- [ ] **División de datos**: Train/test split estratificado
- [ ] **Normalización**: StandardScaler para modelos apropiados
- [ ] **Entrenamiento**: 5 modelos diferentes entrenados
- [ ] **Evaluación**: Métricas completas calculadas
- [ ] **Comparación**: Identificación del mejor modelo
- [ ] **Análisis de importancia**: Variables más influyentes identificadas
- [ ] **Recomendaciones**: Estrategias basadas en insights
- [ ] **Guardado**: Modelos y resultados preservados

### 📊 Métricas Clave a Reportar

1. **Accuracy**: Precisión general del modelo
2. **Precision**: De los predichos como churn, cuántos realmente lo son
3. **Recall**: De los que realmente cancelan, cuántos detectamos
4. **F1-Score**: Balance entre precision y recall
5. **ROC-AUC**: Capacidad de discriminación del modelo
6. **Feature Importance**: Variables más predictivas

### 🎯 Entregables Finales

1. **Notebook completo** con todo el pipeline
2. **Modelo entrenado** (.pkl file)
3. **Reporte de métricas** comparativas
4. **Lista de variables importantes** con interpretación
5. **Recomendaciones estratégicas** basadas en resultados

---

¡Con este instructivo tendrás una guía completa para desarrollar un pipeline robusto de Machine Learning para la predicción de churn en TelecomX! 🚀