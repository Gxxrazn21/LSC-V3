"""
===========================================================================
  SERVIDOR DE SINCRONIZACIÓN EN VIVO (LIVE SYNC & OTA) - SEÑA LSC v4.0
===========================================================================
Permite que tu celular cargue instantáneamente cualquier cambio de código,
diseño o modelo de IA que realices en tu PC sin tener que reinstalar la APK.

Uso:
  python servidor_sync_movil.py
  python servidor_sync_movil.py --port 8000
"""

import os
import sys
import socket
import argparse
from http.server import HTTPServer, SimpleHTTPRequestHandler


def obtener_ip_local() -> str:
    """Detecta la IP local del PC en la red Wi-Fi."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip


class CORSAndNoCacheRequestHandler(SimpleHTTPRequestHandler):
    """Manejador HTTP con soporte CORS y desactivación de caché para recargas instantáneas."""

    def end_headers(self):
        # Habilitar CORS para WebViews y navegadores
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS, HEAD")
        self.send_header("Access-Control-Allow-Headers", "*")
        # Desactivar caché para que los cambios en PC se reflejen de inmediato en el celular
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200, "OK")
        self.end_headers()


def main():
    parser = argparse.ArgumentParser(description="Servidor de Sincronización en Vivo LSC")
    parser.add_argument("--port", type=int, default=8000, help="Puerto de escucha (default: 8000)")
    args = parser.parse_args()

    # Directorio que contiene los recursos web del app
    directorio_web = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app_lsc", "assets", "web")
    if not os.path.exists(directorio_web):
        directorio_web = os.path.join(os.path.dirname(os.path.abspath(__file__)), "estilo")

    os.chdir(directorio_web)
    ip_pc = obtener_ip_local()

    print("\n" + "=" * 76)
    print("  🚀 SERVIDOR DE SINCRONIZACIÓN EN VIVO (LIVE SYNC & OTA) - SEÑA LSC v4.0")
    print("=" * 76)
    print(f"  Carpeta servida : {directorio_web}")
    print(f"  IP de tu PC     : {ip_pc}:{args.port}")
    print(f"  URL Local       : http://localhost:{args.port}/index.html")
    print("=" * 76)
    print("  📱 CÓMO VER TUS CAMBIOS EN EL CELULAR SIN REINSTALAR LA APK:")
    print(f"  1. Asegúrate de que el celular esté conectado al MISMO Wi-Fi que este PC.")
    print(f"  2. Abre la app 'Seña LSC' en tu celular.")
    print(f"  3. Toca el botón arriba a la derecha: [ ⚡ SYNC ]")
    print(f"  4. En el campo 'IP de tu Computador' escribe:")
    print(f"     👉  {ip_pc}:{args.port}")
    print(f"  5. Elige una de las 2 opciones:")
    print(f"     A) '⚡ Activar Modo En Vivo': Cada vez que guardes código en el PC,")
    print(f"        solo tocas 'Recargar' en el celular y lo ves en 1 segundo.")
    print(f"     B) '📥 Descargar Recursos a Celular (OTA)': Descarga los archivos")
    print(f"        a la memoria del celular para usarlos luego sin Wi-Fi.")
    print("=" * 76)
    print("  (Presiona CTRL+C para detener el servidor)\n")

    servidor = HTTPServer(("0.0.0.0", args.port), CORSAndNoCacheRequestHandler)
    try:
        servidor.serve_forever()
    except KeyboardInterrupt:
        print("\n  [OK] Servidor detenido por el usuario.")
        servidor.server_close()


if __name__ == "__main__":
    main()
