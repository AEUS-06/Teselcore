import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def graficar_resultados(ruta_json="comparativa_resultados.json"):
    # 1. Cargar Datos
    if not Path(ruta_json).exists():
        print(f"Error: No se encuentra el archivo {ruta_json}")
        return

    with open(ruta_json, "r", encoding="utf-8") as f:
        res = json.load(f)

    # Definir colores pastel solicitados
    color_cnn = "#B39EB5"     # Morado Pastel
    color_penrose = "#77DD77" # Verde Pastel
    
    sns.set_theme(style="whitegrid")
    fig = plt.figure(figsize=(16, 12))
    
    # --- GRÁFICA 1: PRECISIÓN GLOBAL (Barras) ---
    ax1 = plt.subplot(2, 2, 1)
    modelos = ["CNNClasico", "PenroseNet"]
    # Extraer datos de los diccionarios (usando las claves 'cnn' y 'penrose' si existen)
    accs = [res.get("cnn", {}).get("precision_global", 0) * 100, 
            res.get("penrose", {}).get("precision_global", 0) * 100]
    
    bars = ax1.bar(modelos, accs, color=[color_cnn, color_penrose], alpha=0.8)
    ax1.set_title("Precisión Global en MNIST (%)", fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 110)
    ax1.set_ylabel("Accuracy (%)")
    
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1, f'{height:.2f}%', ha='center', fontsize=12)

    # --- GRÁFICA 2: TIEMPO DE INFERENCIA (Línea/Puntos) ---
    ax2 = plt.subplot(2, 2, 2)
    tiempos = [res.get("cnn", {}).get("tiempo_por_muestra_us", 0), 
               res.get("penrose", {}).get("tiempo_por_muestra_us", 0)]
    
    ax2.plot(modelos, tiempos, marker='o', markersize=10, linewidth=3, color="#FFB347") # Naranja pastel para contraste
    ax2.set_title("Velocidad de Inferencia (Latencia)", fontsize=14, fontweight='bold')
    ax2.set_ylabel("Microsegundos (us) por imagen")
    ax2.grid(True, linestyle='--', alpha=0.6)

    # --- GRÁFICA 3: MATRIZ DE CONFUSIÓN CNN ---
    if "cnn" in res:
        ax3 = plt.subplot(2, 2, 3)
        cm_cnn = np.array(res["cnn"]["matriz_confusion"])
        sns.heatmap(cm_cnn, annot=True, fmt='d', cmap="Purples", ax=ax3, cbar=False)
        ax3.set_title(f"Matriz de Confusión: {res['cnn']['nombre']}", color="#6a0dad")
        ax3.set_xlabel("Predicción")
        ax3.set_ylabel("Real")

    # --- GRÁFICA 4: MATRIZ DE CONFUSIÓN PENROSENET ---
    if "penrose" in res:
        ax4 = plt.subplot(2, 2, 4)
        cm_pen = np.array(res["penrose"]["matriz_confusion"])
        sns.heatmap(cm_pen, annot=True, fmt='d', cmap="Greens", ax=ax4, cbar=False)
        ax4.set_title(f"Matriz de Confusión: {res['penrose']['nombre']}", color="#2d5a27")
        ax4.set_xlabel("Predicción")
        ax4.set_ylabel("Real")

    plt.tight_layout()
    plt.suptitle("Comparativa de Arquitecturas: Teselación de Penrose vs CNN Clásica", fontsize=18, y=1.02)
    
    # Guardar la imagen
    plt.savefig("comparativa_visual.png", dpi=300, bbox_inches='tight')
    print("[✓] Gráficas generadas y guardadas en 'comparativa_visual.png'")
    plt.show()

if __name__ == "__main__":
    # Asegúrate de que el nombre del archivo coincida con el generado por el otro script
    graficar_resultados("comparativa_resultados.json")