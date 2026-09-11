"""
=============================================================
LANZADOR DEL SERVIDOR DE PRUEBAS MÓVILES - LSC v3.0
=============================================================
Detecta automáticamente la IP local de tu computador y
muestra las instrucciones para conectar la cámara de tu celular.

Uso:
  python iniciar_servidor_movil.py
  python iniciar_servidor_movil.py --port 8000
"""

import os
import sys
import socket
import argparse
import datetime
import ipaddress
import uvicorn


def obtener_ip_local() -> str:
    """Obtiene la dirección IP del computador en la red Wi-Fi local."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip


def asegurar_certificados_ssl(cert_path="cert.pem", key_path="key.pem", ip="127.0.0.1"):
    """
    Genera certificados SSL autofirmados con SAN (Subject Alternative Names)
    para permitir HTTPS en Chrome/Safari móvil y habilitar WebRTC/getUserMedia.
    """
    if os.path.exists(cert_path) and os.path.exists(key_path):
        return cert_path, key_path

    try:
        from cryptography import x509
        from cryptography.x509.oid import NameOID
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.hazmat.primitives import serialization

        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        subject = issuer = x509.Name([
            x509.NameAttribute(NameOID.COMMON_NAME, ip),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "LSC v3.0 Plataforma"),
        ])

        san_list = [
            x509.DNSName("localhost"),
            x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
        ]
        try:
            san_list.append(x509.IPAddress(ipaddress.IPv4Address(ip)))
        except Exception:
            pass

        cert = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(issuer)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(datetime.datetime.utcnow())
            .not_valid_after(datetime.datetime.utcnow() + datetime.timedelta(days=365))
            .add_extension(x509.SubjectAlternativeName(san_list), critical=False)
            .sign(key, hashes.SHA256())
        )

        with open(key_path, "wb") as f:
            f.write(
                key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.TraditionalOpenSSL,
                    encryption_algorithm=serialization.NoEncryption(),
                )
            )

        with open(cert_path, "wb") as f:
            f.write(cert.public_bytes(serialization.Encoding.PEM))

        return cert_path, key_path
    except Exception as e:
        print(f"  [AVISO] No se pudo generar certificado SSL autofirmado: {e}")
        return None, None


def main():
    parser = argparse.ArgumentParser(description="Servidor de Pruebas Móviles LSC v3.0")
    parser.add_argument("--host", default="0.0.0.0", help="Host de escucha (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Puerto de escucha (default: 8000)")
    parser.add_argument("--no-ssl", action="store_true", help="Desactivar HTTPS (no recomendado para celulares)")
    args = parser.parse_args()

    ip_local = obtener_ip_local()
    usar_ssl = not args.no_ssl

    cert_file, key_file = None, None
    if usar_ssl:
        cert_file, key_file = asegurar_certificados_ssl(ip=ip_local)
        if not cert_file or not key_file:
            usar_ssl = False

    protocolo = "https" if usar_ssl else "http"

    print("\n" + "=" * 72)
    print(f"  LSC v3.0 — SERVIDOR DE PRUEBAS CON CÁMARA MÓVIL ({protocolo.upper()})")
    print("=" * 72)
    print(f"  PC Local     : {protocolo}://localhost:{args.port}")
    print(f"  TELÉFONO     : {protocolo}://{ip_local}:{args.port}")
    print("=" * 72)
    if usar_ssl:
        print("  PASOS PARA CONECTAR TU TELÉFONO (POLÍTICA DE SEGURIDAD MÓVIL):")
        print(f"  1. Conecta tu teléfono a la misma red Wi-Fi de este computador.")
        print(f"  2. Abre Chrome o Safari en tu teléfono e ingresa a:")
        print(f"     👉  https://{ip_local}:{args.port}")
        print(f"  3. Chrome mostrará: 'La conexión no es privada' (por ser red local).")
        print(f"     -> Toca 'Configuración avanzada' (o 'Detalles').")
        print(f"     -> Toca 'Acceder a {ip_local} (no seguro)' o 'Continuar'.")
        print(f"  4. Acepta el permiso de la cámara en el celular y ¡listo!")
        print(f"     La cámara de alta resolución transmitirá en tiempo real.")
    else:
        print("  INSTRUCCIONES PARA PROBAR CON TU CELULAR:")
        print(f"  1. Conecta tu teléfono a la misma red Wi-Fi que este computador.")
        print(f"  2. Abre Chrome en tu teléfono: http://{ip_local}:{args.port}")
        print(f"  3. Si la cámara no abre, visita chrome://flags/#unsafely-treat-insecure-origin-as-secure")
        print(f"     y añade: http://{ip_local}:{args.port}")
    print("=" * 72)
    print("  (Presiona CTRL+C para detener el servidor)\n")

    if usar_ssl:
        uvicorn.run(
            "motor_lsc.servidor_app:app",
            host=args.host,
            port=args.port,
            ssl_keyfile=key_file,
            ssl_certfile=cert_file,
            log_level="info",
        )
    else:
        uvicorn.run("motor_lsc.servidor_app:app", host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()

