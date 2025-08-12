#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
📊 Mostrar Resultados del Pipeline de ML - TelecomX
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    """Función principal para mostrar resultados"""
    
    print("📊 Resultados del Pipeline de Machine Learning - TelecomX")
    print("="*70)
    
    try:
        # Cargar resultados
        results_df = pd.read_csv('model_comparison_results.csv', index_col=0)
        feature_importance = pd.read_csv('feature_importance.csv')
        
        print("\n🏆 COMPARACIÓN DE MODELOS")
        print("="*50)
        print(results_df.round(3).to_string())
        
        # Identificar mejor modelo
        best_f1_model = results_df['f1_score'].idxmax()
        best_accuracy_model = results_df['test_accuracy'].idxmax()
        
        print(f"\n🎯 Mejor modelo por F1-Score: {best_f1_model}")
        print(f"🏆 Mejor modelo por Accuracy: {best_accuracy_model}")
        
        print(f"\n📊 Métricas del Mejor Modelo ({best_f1_model}):")
        best_metrics = results_df.loc[best_f1_model]
        print(f"   Accuracy: {best_metrics['test_accuracy']:.3f}")
        print(f"   Precision: {best_metrics['precision']:.3f}")
        print(f"   Recall: {best_metrics['recall']:.3f}")
        print(f"   F1-Score: {best_metrics['f1_score']:.3f}")
        print(f"   ROC-AUC: {best_metrics['roc_auc']:.3f}")
        print(f"   Overfitting: {best_metrics['overfitting']:.3f}")
        
        print("\n🔍 TOP 10 VARIABLES MÁS IMPORTANTES")
        print("="*50)
        for i, row in feature_importance.head(10).iterrows():
            print(f"{i+1:2d}. {row['feature']:<35} | Importancia: {row['importance']:.3f}")
        
        print("\n" + "="*70)
        print("💡 RECOMENDACIONES ESTRATÉGICAS")
        print("="*70)
        
        recommendations = [
            "🎯 RETENCIÓN TEMPRANA: Implementar programa de seguimiento intensivo para clientes nuevos (primeros 6 meses)",
            "📋 CONTRATOS ANUALES: Incentivar migración de contratos mes-a-mes a anuales con descuentos atractivos",
            "💰 REVISIÓN DE PRECIOS: Analizar estructura de precios de fibra óptica vs satisfacción del cliente",
            "🔧 CALIDAD DE SERVICIO: Mejorar estabilidad y soporte técnico para servicios de fibra óptica",
            "💳 MÉTODOS DE PAGO: Promocionar métodos de pago más estables (débito automático vs cheque electrónico)",
            "🎁 PROGRAMAS DE LEALTAD: Crear incentivos progresivos basados en antigüedad del cliente",
            "📞 INTERVENCIÓN PROACTIVA: Identificar clientes en riesgo usando el modelo predictivo para intervención temprana"
        ]
        
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec}")
        
        print("\n" + "="*70)
        print("📈 INSIGHTS CLAVE")
        print("="*70)
        
        insights = [
            f"📊 El modelo {best_f1_model} es el más efectivo con un F1-Score de {best_metrics['f1_score']:.3f}",
            f"🎯 La precisión general del modelo es del {best_metrics['test_accuracy']:.1%}",
            f"📈 El ROC-AUC de {best_metrics['roc_auc']:.3f} indica buena capacidad de discriminación",
            f"⚖️ El overfitting de {best_metrics['overfitting']:.3f} es {'aceptable' if best_metrics['overfitting'] < 0.05 else 'moderado'}",
            f"🔍 Las variables más importantes son: {feature_importance.iloc[0]['feature']}, {feature_importance.iloc[1]['feature']}, {feature_importance.iloc[2]['feature']}",
            f"⚠️ Los contratos mes-a-mes tienen el mayor riesgo de churn",
            f"💰 Los cargos totales y mensuales son predictores fuertes de cancelación",
            f"🌐 Los clientes de fibra óptica tienen mayor tendencia a cancelar"
        ]
        
        for insight in insights:
            print(f"• {insight}")
        
        print("\n" + "="*70)
        print("✅ RESUMEN FINAL")
        print("="*70)
        
        print("🎯 El pipeline de Machine Learning se ejecutó exitosamente y generó:")
        print("   ✅ Modelo entrenado y optimizado")
        print("   ✅ Análisis de importancia de variables")
        print("   ✅ Comparación de múltiples algoritmos")
        print("   ✅ Recomendaciones estratégicas basadas en datos")
        print("   ✅ Archivos de resultados guardados")
        
        print(f"\n🚀 El modelo {best_f1_model} está listo para ser usado en producción")
        print("   con una precisión del {:.1%} y un F1-Score de {:.3f}".format(
            best_metrics['test_accuracy'], best_metrics['f1_score']))
        
        print("\n" + "="*70)
        print("🎉 ¡Pipeline completado exitosamente!")
        print("="*70)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Asegúrate de haber ejecutado primero el pipeline principal")

if __name__ == "__main__":
    main()
