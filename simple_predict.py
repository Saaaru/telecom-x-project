#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔮 Script Simplificado de Predicción de Churn - TelecomX
"""

import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

def main():
    """Función principal simplificada"""
    
    print("🔮 Predicción Simplificada de Churn - TelecomX")
    print("="*60)
    
    try:
        # Cargar modelo y escalador
        model = joblib.load('best_model_Logistic_Regression.pkl')
        scaler = joblib.load('scaler.pkl')
        print("✅ Modelo y escalador cargados correctamente")
        
        # Cargar datos procesados para obtener la estructura correcta
        original_data = pd.read_csv('telecom_data_processed_for_ml.csv')
        print(f"✅ Dataset original cargado: {original_data.shape}")
        
        # Tomar una muestra pequeña para predicción
        sample_data = original_data.sample(n=5, random_state=42)
        X_sample = sample_data.drop('Churn', axis=1)
        y_true = sample_data['Churn']
        
        print(f"\n📊 Muestra de {len(X_sample)} clientes para predicción:")
        
        # Hacer predicciones
        predictions = model.predict(X_sample)
        probabilities = model.predict_proba(X_sample)[:, 1]
        
        # Crear resultados
        results = pd.DataFrame({
            'Cliente_ID': range(1, len(X_sample) + 1),
            'Tenure': X_sample['tenure'],
            'Contract_Type': ['Month-to-month' if 'Contract_Month-to-month' in X_sample.columns and X_sample.iloc[i]['Contract_Month-to-month'] == 1 
                             else 'One year' if 'Contract_One year' in X_sample.columns and X_sample.iloc[i]['Contract_One year'] == 1
                             else 'Two year' for i in range(len(X_sample))],
            'Monthly_Charges': X_sample['Charges.Monthly'],
            'Churn_Real': y_true,
            'Churn_Predicho': predictions,
            'Probabilidad_Churn': probabilities,
            'Nivel_Riesgo': ['Alto' if p > 0.7 else 'Medio' if p > 0.4 else 'Bajo' for p in probabilities]
        })
        
        print("\n📊 Resultados de Predicción:")
        print(results.to_string(index=False))
        
        # Análisis de precisión
        accuracy = (predictions == y_true).mean()
        print(f"\n🎯 Precisión en la muestra: {accuracy:.1%}")
        
        # Análisis detallado
        print("\n" + "="*60)
        print("🔍 ANÁLISIS DETALLADO")
        print("="*60)
        
        for i, row in results.iterrows():
            print(f"\n👤 Cliente {row['Cliente_ID']}:")
            print(f"   Antigüedad: {row['Tenure']} meses")
            print(f"   Contrato: {row['Contract_Type']}")
            print(f"   Cargo Mensual: ${row['Monthly_Charges']:.2f}")
            print(f"   Churn Real: {'🔴 Sí' if row['Churn_Real'] == 1 else '🟢 No'}")
            print(f"   Churn Predicho: {'🔴 Sí' if row['Churn_Predicho'] == 1 else '🟢 No'}")
            print(f"   Probabilidad: {row['Probabilidad_Churn']:.1%}")
            print(f"   Nivel de Riesgo: {row['Nivel_Riesgo']}")
            
            # Recomendaciones
            if row['Churn_Predicho'] == 1:
                if row['Probabilidad_Churn'] > 0.8:
                    print("   ⚠️  ACCIÓN URGENTE: Cliente en alto riesgo")
                elif row['Probabilidad_Churn'] > 0.6:
                    print("   🎯 ACCIÓN INMEDIATA: Implementar retención")
                else:
                    print("   📞 SEGUIMIENTO: Contactar para mejorar satisfacción")
            else:
                if row['Probabilidad_Churn'] < 0.2:
                    print("   ✅ CLIENTE LEAL: Mantener estrategia actual")
                else:
                    print("   👀 MONITOREO: Cliente estable pero requiere atención")
        
        # Resumen estadístico
        print("\n" + "="*60)
        print("📈 RESUMEN ESTADÍSTICO")
        print("="*60)
        
        total_customers = len(results)
        predicted_churn = sum(results['Churn_Predicho'])
        real_churn = sum(results['Churn_Real'])
        avg_probability = results['Probabilidad_Churn'].mean()
        
        print(f"Total de clientes analizados: {total_customers}")
        print(f"Churn real: {real_churn} ({real_churn/total_customers:.1%})")
        print(f"Churn predicho: {predicted_churn} ({predicted_churn/total_customers:.1%})")
        print(f"Precisión del modelo: {accuracy:.1%}")
        print(f"Probabilidad promedio de churn: {avg_probability:.1%}")
        
        # Distribución de niveles de riesgo
        risk_dist = results['Nivel_Riesgo'].value_counts()
        print(f"\nDistribución de niveles de riesgo:")
        for risk, count in risk_dist.items():
            print(f"   {risk}: {count} clientes ({count/total_customers:.1%})")
        
        print("\n" + "="*60)
        print("✅ Predicción completada exitosamente!")
        print("="*60)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Asegúrate de haber ejecutado primero el pipeline principal")

if __name__ == "__main__":
    main()
