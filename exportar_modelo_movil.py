"""
Exportador del Modelo de IA LSC70 a JavaScript / JSON para ejecución 100% en el dispositivo móvil (Zero Server).
"""
import os
import json
import numpy as np

def exportar():
    npz_path = os.path.join("modelos_guardados", "modelo_ia_lsc70.npz")
    if not os.path.exists(npz_path):
        print(f"Error: No existe {npz_path}")
        return

    data = np.load(npz_path, allow_pickle=True)
    
    modelo_dict = {
        "clases": data["clases"].tolist(),
        "scaler_mean": [round(float(v), 6) for v in data["scaler_mean"]],
        "scaler_scale": [round(float(v), 6) for v in data["scaler_scale"]],
        "w0": [[round(float(v), 6) for v in row] for row in data["mlp_w0"]],
        "b0": [round(float(v), 6) for v in data["mlp_b0"]],
        "w1": [[round(float(v), 6) for v in row] for row in data["mlp_w1"]],
        "b1": [round(float(v), 6) for v in data["mlp_b1"]],
        "w2": [[round(float(v), 6) for v in row] for row in data["mlp_w2"]],
        "b2": [round(float(v), 6) for v in data["mlp_b2"]],
    }

    # 1. Guardar JSON
    json_path = os.path.join("modelos_guardados", "modelo_lsc_movil.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(modelo_dict, f)
    print(f"[OK] Modelo JSON exportado a: {json_path} ({os.path.getsize(json_path)/1024:.1f} KB)")

    # 2. Guardar JS para el cliente móvil (incluible directamente con <script>)
    js_path = os.path.join("estilo", "modelo_ia_cliente.js")
    with open(js_path, "w", encoding="utf-8") as f:
        f.write("// Modelo Neuronal LSC 109D exportado para ejecución local sin servidor\n")
        f.write("window.MODELO_LSC = ")
        json.dump(modelo_dict, f)
        f.write(";\n")
    print(f"[OK] Modelo JS exportado a: {js_path} ({os.path.getsize(js_path)/1024:.1f} KB)")

if __name__ == "__main__":
    exportar()
