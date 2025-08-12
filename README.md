# 🤖 Pipeline de Machine Learning para Predicción de Churn - TelecomX

## 📋 Descripción

Este proyecto implementa un pipeline completo de Machine Learning para predecir la fuga de clientes (churn) en TelecomX. El pipeline incluye desde la preparación de datos hasta el entrenamiento de múltiples modelos y la generación de recomendaciones estratégicas.

## 🎯 Objetivos

- ✅ Preparar los datos para modelado predictivo
- ✅ Entrenar y evaluar múltiples modelos de clasificación
- ✅ Identificar los factores más importantes para la predicción
- ✅ Proporcionar recomendaciones estratégicas basadas en datos

## 📁 Estructura del Proyecto

```
telecom-x-project/
├── Analisis_TelecomX_Corregido.ipynb    # Análisis exploratorio original
├── TelecomX_ML_Pipeline.ipynb           # Notebook del pipeline de ML
├── telecom_ml_pipeline.py               # Script ejecutable del pipeline
├── telecom_ml_guide.md                  # Guía detallada del proceso
├── requirements.txt                     # Dependencias del proyecto
├── README.md                           # Este archivo
└── TelecomX_diccionario.md             # Diccionario de variables
```

## 🚀 Instalación y Configuración

### 1. Instalar Dependencias

```bash
pip install -r requirements.txt
```

### 2. Verificar Instalación

```python
python -c "import pandas, numpy, sklearn, imblearn; print('✅ Todas las dependencias instaladas correctamente')"
```

## 📊 Datos

Los datos se obtienen automáticamente desde:
- **Fuente**: GitHub de Alura Cursos
- **Formato**: JSON anidado
- **Tamaño**: ~7,267 registros
- **Variables**: 21 características + variable objetivo (Churn)

### Variables Principales

- **Demográficas**: gender, SeniorCitizen, Partner, Dependents
- **Servicios**: PhoneService, InternetService, StreamingTV, etc.
- **Contrato**: Contract, PaperlessBilling, PaymentMethod
- **Financieras**: Charges.Monthly, Charges.Total, tenure
- **Objetivo**: Churn (Yes/No)

## 🔧 Uso del Pipeline

### Opción 1: Script Python (Recomendado)

```bash
python telecom_ml_pipeline.py
```

### Opción 2: Notebook Jupyter

1. Abrir `TelecomX_ML_Pipeline.ipynb`
2. Ejecutar todas las celdas secuencialmente
3. Revisar resultados y visualizaciones

### Opción 3: Ejecución por Pasos

Puedes ejecutar secciones específicas del notebook según tus necesidades:

1. **Paso 1-2**: Preparación y carga de datos
2. **Paso 3-4**: Preprocesamiento y normalización
3. **Paso 5**: Entrenamiento de modelos
4. **Paso 6-7**: Evaluación y análisis
5. **Paso 8-9**: Recomendaciones y guardado

## 🤖 Modelos Implementados

El pipeline entrena y compara 5 modelos diferentes:

1. **Regresión Logística**: Modelo lineal interpretable
2. **Random Forest**: Ensemble robusto y preciso
3. **K-Nearest Neighbors**: Basado en similitud
4. **Árbol de Decisión**: Modelo simple y visual
5. **Support Vector Machine**: Clasificador de margen máximo

## 📈 Métricas de Evaluación

### Métricas Principales

- **Accuracy**: Precisión general del modelo
- **Precision**: De los predichos como churn, cuántos realmente lo son
- **Recall**: De los que realmente cancelan, cuántos detectamos
- **F1-Score**: Balance entre precision y recall
- **ROC-AUC**: Capacidad de discriminación del modelo

### Análisis de Overfitting

- Comparación entre accuracy de entrenamiento y prueba
- Identificación de modelos que generalizan mejor

## 🔍 Análisis de Importancia de Variables

### Random Forest Feature Importance
- Identifica las variables más predictivas
- Ranking de importancia numérica

### Coeficientes de Regresión Logística
- Interpretación de dirección (aumenta/disminuye churn)
- Magnitud del efecto de cada variable

## 📊 Resultados Esperados

### Factores Clave Identificados

1. **Contract_Month-to-month**: Contratos mensuales (alto riesgo)
2. **tenure**: Antigüedad baja del cliente
3. **Charges.Monthly**: Cargos mensuales altos
4. **InternetService_Fiber optic**: Servicio de fibra óptica
5. **PaymentMethod_Electronic check**: Método de pago

### Recomendaciones Estratégicas

- 🎯 **Retención Temprana**: Seguimiento intensivo para clientes nuevos
- 📋 **Contratos Anuales**: Incentivos para migración de contratos
- 💰 **Revisión de Precios**: Análisis de estructura de precios
- 🔧 **Calidad de Servicio**: Mejora de estabilidad de fibra óptica
- 💳 **Métodos de Pago**: Promoción de métodos más estables
- 🎁 **Programas de Lealtad**: Incentivos progresivos
- 📞 **Intervención Proactiva**: Uso del modelo para detección temprana

## 💾 Archivos Generados

Después de ejecutar el pipeline, se generan los siguientes archivos:

### Modelos
- `best_model_[MODELO].pkl`: Modelo entrenado con mejor rendimiento
- `scaler.pkl`: Escalador para normalización

### Resultados
- `model_comparison_results.csv`: Comparación de métricas de todos los modelos
- `feature_importance.csv`: Importancia de variables (Random Forest)
- `telecom_data_processed_for_ml.csv`: Dataset procesado para ML

## 📋 Checklist de Completion

- [x] **Preparación de datos**: Eliminación de columnas irrelevantes
- [x] **Codificación categórica**: One-hot encoding aplicado
- [x] **Análisis de balance**: Evaluación de desbalance de clases
- [x] **División de datos**: Train/test split estratificado
- [x] **Normalización**: StandardScaler para modelos apropiados
- [x] **Entrenamiento**: 5 modelos diferentes entrenados
- [x] **Evaluación**: Métricas completas calculadas
- [x] **Comparación**: Identificación del mejor modelo
- [x] **Análisis de importancia**: Variables más influyentes identificadas
- [x] **Recomendaciones**: Estrategias basadas en insights
- [x] **Guardado**: Modelos y resultados preservados

## 🔧 Personalización

### Modificar Parámetros de Modelos

```python
# En el script, puedes modificar los parámetros de los modelos:
models = {
    'Logistic_Regression': LogisticRegression(random_state=42, max_iter=1000, C=1.0),
    'Random_Forest': RandomForestClassifier(random_state=42, n_estimators=200, max_depth=15),
    'KNN': KNeighborsClassifier(n_neighbors=7, weights='distance'),
    'Decision_Tree': DecisionTreeClassifier(random_state=42, max_depth=12),
    'SVM': SVC(random_state=42, probability=True, C=1.0, kernel='rbf')
}
```

### Ajustar División de Datos

```python
# Modificar la proporción train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,  # Cambiar a 20% para prueba
    random_state=42,
    stratify=y
)
```

### Agregar Nuevos Modelos

```python
# Agregar nuevos modelos al diccionario
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB

models['Gradient_Boosting'] = GradientBoostingClassifier(random_state=42)
models['Naive_Bayes'] = GaussianNB()
```

## 🐛 Solución de Problemas

### Error de Importación

```bash
# Si hay problemas con imbalanced-learn
pip install --upgrade imbalanced-learn
```

### Error de Memoria

```python
# Reducir el número de estimadores en Random Forest
'Random_Forest': RandomForestClassifier(random_state=42, n_estimators=50)
```

### Error de Convergencia

```python
# Aumentar max_iter en Regresión Logística
'Logistic_Regression': LogisticRegression(random_state=42, max_iter=2000)
```

## 📚 Recursos Adicionales

- [Guía de Machine Learning](telecom_ml_guide.md): Documentación detallada
- [Diccionario de Variables](TelecomX_diccionario.md): Descripción de cada variable
- [Análisis Exploratorio](Analisis_TelecomX_Corregido.ipynb): EDA completo

## 🤝 Contribuciones

Para contribuir al proyecto:

1. Fork el repositorio
2. Crear una rama para tu feature
3. Commit tus cambios
4. Push a la rama
5. Abrir un Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.

## 📞 Contacto

Para preguntas o soporte:
- 📧 Email: [tu-email@ejemplo.com]
- 🐛 Issues: [Crear un issue en GitHub]

---

**¡Disfruta explorando el pipeline de Machine Learning para TelecomX! 🚀**
