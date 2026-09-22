import 'dart:convert';
import 'dart:io';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:path_provider/path_provider.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:shelf/shelf.dart';
import 'package:shelf/shelf_io.dart' as io;
import 'package:webview_flutter/webview_flutter.dart';
import 'package:webview_flutter_android/webview_flutter_android.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  SystemChrome.setPreferredOrientations([DeviceOrientation.portraitUp]);
  runApp(const LSCApp());
}

class LSCApp extends StatelessWidget {
  const LSCApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Gestual Vision AI',
      debugShowCheckedModeBanner: false,
      theme: ThemeData.light().copyWith(
        scaffoldBackgroundColor: const Color(0xFFF4F7FB),
        colorScheme: const ColorScheme.light(
          primary: Color(0xFF1D4ED8),
          secondary: Color(0xFFF59E0B),
        ),
      ),
      home: const LSCHomePage(),
    );
  }
}

class LSCHomePage extends StatefulWidget {
  const LSCHomePage({super.key});

  @override
  State<LSCHomePage> createState() => _LSCHomePageState();
}

class _LSCHomePageState extends State<LSCHomePage> {
  static const _nativeChannel = MethodChannel('com.lsc.app/native');
  HttpServer? _server;
  late final WebViewController _controller;
  bool _serverReady = false;
  bool _splashVisible = true; // controla el fade-out gradual del splash nativo
  String _statusMessage = 'Iniciando Gestual Vision AI v6.2...';

  // Gestión de Live Sync y Recursos OTA
  String _pcHost = '192.168.1.15:8000';
  bool _isLiveMode = false;
  bool _hasDownloadedAssets = false;

  @override
  void initState() {
    super.initState();
    _startLocalServerAndApp();
  }

  Future<Directory> _getWebAssetsDir() async {
    final docDir = await getApplicationDocumentsDirectory();
    final assetsDir = Directory('${docDir.path}/web_assets');
    if (!await assetsDir.exists()) {
      await assetsDir.create(recursive: true);
    }
    return assetsDir;
  }

  Future<void> _checkDownloadedAssets() async {
    try {
      final assetsDir = await _getWebAssetsDir();
      final indexFile = File('${assetsDir.path}/index.html');
      final exists = await indexFile.exists();
      if (mounted) {
        setState(() => _hasDownloadedAssets = exists);
      }
    } catch (_) {}
  }

  Future<void> _startLocalServerAndApp() async {
    setState(() => _statusMessage = 'Iniciando motor autónomo...');

    // Cargar preferencias de sincronización
    final prefs = await SharedPreferences.getInstance();
    _pcHost = prefs.getString('pref_pc_host') ?? '192.168.1.15:8000';
    _isLiveMode = prefs.getBool('pref_is_live_mode') ?? false;
    await _checkDownloadedAssets();

    try {
      final handler = const Pipeline()
          .addMiddleware(logRequests())
          .addHandler(_handleAssetRequest);

      try {
        _server = await io.serve(handler, InternetAddress.loopbackIPv4, 8765);
      } catch (_) {
        _server = await io.serve(handler, InternetAddress.loopbackIPv4, 0);
      }
      final port = _server!.port;
      debugPrint('Servidor local interno corriendo en: http://127.0.0.1:$port');

      late final PlatformWebViewControllerCreationParams params;
      if (WebViewPlatform.instance is AndroidWebViewPlatform) {
        params = AndroidWebViewControllerCreationParams();
      } else {
        params = const PlatformWebViewControllerCreationParams();
      }

      final controller = WebViewController.fromPlatformCreationParams(params);

      controller
        ..setJavaScriptMode(JavaScriptMode.unrestricted)
        ..setBackgroundColor(const Color(0xFFF4F7FB))
        ..addJavaScriptChannel(
          'NativeBridge',
          onMessageReceived: (JavaScriptMessage jsMessage) async {
            try {
              final data =
                  jsonDecode(jsMessage.message) as Map<String, dynamic>;
              final action = data['action'];
              if (action == 'speak') {
                final text = data['text'] as String? ?? '';
                if (text.isNotEmpty) {
                  await _nativeChannel.invokeMethod('speak', {'text': text});
                }
              } else if (action == 'vibrate') {
                final duration = data['duration'] as int? ?? 45;
                await _nativeChannel.invokeMethod('vibrate', {
                  'duration': duration,
                });
              } else if (action == 'showSyncModal' || action == 'openSync') {
                _showSyncModal();
              }
            } catch (e) {
              debugPrint('Error en NativeBridge: $e');
            }
          },
        )
        ..setNavigationDelegate(
          NavigationDelegate(
            onPageStarted: (String url) => debugPrint('Página cargando: $url'),
            onPageFinished: (String url) {
              debugPrint('Página lista: $url');
              setState(() => _serverReady = true);
              // Fade-out gradual del splash nativo (350ms) y luego lo ocultamos
              Future.delayed(const Duration(milliseconds: 380), () {
                if (mounted) setState(() => _splashVisible = false);
              });
            },
            onWebResourceError: (WebResourceError error) {
              debugPrint('Error de recurso: ${error.description}');
              if (_isLiveMode) {
                // Si el modo en vivo remoto falla, volver automáticamente a modo autónomo local
                _isLiveMode = false;
                final fallbackPort = _server?.port ?? 8765;
                _controller.loadRequest(
                  Uri.parse('http://127.0.0.1:$fallbackPort/index.html'),
                );
              }
            },
          ),
        );

      if (controller.platform is AndroidWebViewController) {
        AndroidWebViewController.enableDebugging(true);
        (controller.platform as AndroidWebViewController)
            .setMediaPlaybackRequiresUserGesture(false);
        (controller.platform as AndroidWebViewController)
            .setOnPlatformPermissionRequest((request) {
              debugPrint('Permiso de WebRTC otorgado: ${request.types}');
              request.grant();
            });
      }

      _controller = controller;

      // Cargar modo en vivo (Wi-Fi PC) o modo autónomo local
      if (_isLiveMode && _pcHost.isNotEmpty) {
        await _controller.loadRequest(Uri.parse('http://$_pcHost/index.html'));
      } else {
        await _controller.loadRequest(
          Uri.parse('http://127.0.0.1:$port/index.html'),
        );
      }
    } catch (e) {
      debugPrint('Error iniciando app: $e');
      setState(() => _statusMessage = 'Error al iniciar: $e');
    }
  }

  Future<Response> _handleAssetRequest(Request request) async {
    String path = request.url.path;
    if (path.isEmpty || path == '/') {
      path = 'index.html';
    }

    try {
      Uint8List bytes;
      final assetsDir = await _getWebAssetsDir();
      final localFile = File('${assetsDir.path}/$path');

      // Si existe recurso descargado por OTA en almacenamiento del teléfono, servirlo primero
      if (await localFile.exists()) {
        bytes = await localFile.readAsBytes();
      } else {
        final byteData = await rootBundle.load('assets/web/$path');
        bytes = byteData.buffer.asUint8List();
      }

      String mime = 'text/plain';
      if (path.endsWith('.html')) {
        mime = 'text/html; charset=utf-8';
      } else if (path.endsWith('.js')) {
        mime = 'application/javascript; charset=utf-8';
      } else if (path.endsWith('.css')) {
        mime = 'text/css; charset=utf-8';
      } else if (path.endsWith('.json')) {
        mime = 'application/json; charset=utf-8';
      } else if (path.endsWith('.png')) {
        mime = 'image/png';
      }

      return Response.ok(
        bytes,
        headers: {
          'content-type': mime,
          'access-control-allow-origin': '*',
          'cache-control': 'no-cache, no-store, must-revalidate',
          'pragma': 'no-cache',
          'expires': '0',
        },
      );
    } catch (e) {
      return Response.notFound('Recurso no encontrado: $path');
    }
  }

  static const String _defaultCloudBase =
      'https://raw.githubusercontent.com/Gxxrazn21/LSC-V3/main/estilo';

  Future<bool> _downloadResourcesFromCloud({String? baseUrl}) async {
    final base = (baseUrl != null && baseUrl.isNotEmpty)
        ? baseUrl
        : _defaultCloudBase;
    try {
      final assetsDir = await _getWebAssetsDir();
      final filesToSync = [
        'index.html',
        'motor_inferencia_local.js',
        'modelo_ia_cliente.js',
        'logo_simbolo.png',
        'logo.png',
        'icon-192.png',
        'icon-512.png',
      ];

      final client = HttpClient();
      client.connectionTimeout = const Duration(seconds: 25);

      for (final fileName in filesToSync) {
        final uri = Uri.parse(
          '$base/$fileName?nocache=${DateTime.now().millisecondsSinceEpoch}',
        );
        final request = await client.getUrl(uri);
        final response = await request.close();

        if (response.statusCode == 200) {
          final fileBytes = await consolidateHttpClientResponseBytes(response);
          final targetFile = File('${assetsDir.path}/$fileName');
          await targetFile.writeAsBytes(fileBytes, flush: true);
        } else {
          client.close();
          return false;
        }
      }
      client.close();
      await _checkDownloadedAssets();
      return true;
    } catch (e) {
      debugPrint('Error descargando recursos OTA de la nube: $e');
      return false;
    }
  }

  void _showSyncModal() {
    final textController = TextEditingController(text: _pcHost);
    bool downloading = false;
    bool showAdvanced = false;

    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: const Color(0xFF0F172A),
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(24)),
      ),
      builder: (ctx) {
        return StatefulBuilder(
          builder: (context, setModalState) {
            final port = _server?.port ?? 8765;
            return Padding(
              padding: EdgeInsets.only(
                left: 20,
                right: 20,
                top: 20,
                bottom: MediaQuery.of(context).viewInsets.bottom + 24,
              ),
              child: Column(
                mainAxisSize: MainAxisSize.min,
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      const Row(
                        children: [
                          Icon(
                            Icons.cloud_sync,
                            color: Color(0xFF00E5FF),
                            size: 26,
                          ),
                          SizedBox(width: 10),
                          Text(
                            'Actualización & Nube LSC v6.2',
                            style: TextStyle(
                              fontSize: 18,
                              fontWeight: FontWeight.w800,
                              color: Colors.white,
                            ),
                          ),
                        ],
                      ),
                      IconButton(
                        onPressed: () => Navigator.pop(ctx),
                        icon: const Icon(Icons.close, color: Colors.white54),
                      ),
                    ],
                  ),
                  const SizedBox(height: 8),
                  Container(
                    padding: const EdgeInsets.all(12),
                    decoration: BoxDecoration(
                      color: const Color(0xFF1E293B),
                      borderRadius: BorderRadius.circular(12),
                      border: Border.all(color: Colors.white10),
                    ),
                    child: Row(
                      children: [
                        Icon(
                          _hasDownloadedAssets
                              ? Icons.check_circle
                              : Icons.phone_android,
                          color: _hasDownloadedAssets
                              ? const Color(0xFF00FF9D)
                              : const Color(0xFF00E5FF),
                          size: 20,
                        ),
                        const SizedBox(width: 10),
                        Expanded(
                          child: Text(
                            _hasDownloadedAssets
                                ? 'Versión Nube IA v6.2 Maestro (Cámara Limpia & UI Fluida)'
                                : 'Ejecutando versión embebida de la APK (IA v6.2)',
                            style: const TextStyle(
                              fontSize: 12,
                              color: Colors.white70,
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),
                  const SizedBox(height: 16),
                  // BOTÓN 1 PRINCIPAL: ACTUALIZACIÓN AUTOMÁTICA DESDE LA NUBE (CERO CONFIGURACIÓN)
                  SizedBox(
                    width: double.infinity,
                    height: 52,
                    child: ElevatedButton.icon(
                      style: ElevatedButton.styleFrom(
                        backgroundColor: const Color(0xFF00E5FF),
                        foregroundColor: Colors.black,
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(14),
                        ),
                        elevation: 4,
                      ),
                      icon: downloading
                          ? const SizedBox(
                              width: 20,
                              height: 20,
                              child: CircularProgressIndicator(
                                strokeWidth: 2.5,
                                color: Colors.black,
                              ),
                            )
                          : const Icon(Icons.cloud_download, size: 22),
                      label: Text(
                        downloading
                            ? 'Descargando modelo desde la nube...'
                            : '☁️ Actualizar Modelo desde la Nube',
                        style: const TextStyle(
                          fontWeight: FontWeight.w800,
                          fontSize: 14,
                        ),
                      ),
                      onPressed: downloading
                          ? null
                          : () async {
                              final messenger = ScaffoldMessenger.of(
                                this.context,
                              );
                              final nav = Navigator.of(ctx);

                              setModalState(() => downloading = true);
                              final ok = await _downloadResourcesFromCloud();
                              setModalState(() => downloading = false);

                              if (ok) {
                                final prefs =
                                    await SharedPreferences.getInstance();
                                await prefs.setBool('pref_is_live_mode', false);

                                if (!mounted) return;
                                setState(() {
                                  _isLiveMode = false;
                                });

                                nav.pop();
                                await _controller.clearCache();
                                _controller.loadRequest(
                                  Uri.parse(
                                    'http://127.0.0.1:$port/index.html?v=${DateTime.now().millisecondsSinceEpoch}',
                                  ),
                                );
                                messenger.showSnackBar(
                                  SnackBar(
                                    content: const Text(
                                      '✅ ¡Interfaz y modelo actualizados a Gestual Vision AI v6.2 Maestro!',
                                    ),
                                    backgroundColor: Colors.green,
                                    duration: const Duration(seconds: 4),
                                    behavior: SnackBarBehavior.floating,
                                    shape: RoundedRectangleBorder(
                                      borderRadius: BorderRadius.circular(14),
                                    ),
                                    margin: const EdgeInsets.fromLTRB(
                                      14,
                                      0,
                                      14,
                                      14,
                                    ),
                                  ),
                                );
                              } else {
                                messenger.showSnackBar(
                                  SnackBar(
                                    content: const Text(
                                      '❌ No se pudo descargar. Verifica que tengas conexión a internet.',
                                    ),
                                    backgroundColor: Colors.redAccent,
                                    behavior: SnackBarBehavior.floating,
                                    shape: RoundedRectangleBorder(
                                      borderRadius: BorderRadius.circular(14),
                                    ),
                                    margin: const EdgeInsets.fromLTRB(
                                      14,
                                      0,
                                      14,
                                      14,
                                    ),
                                  ),
                                );
                              }
                            },
                    ),
                  ),
                  const SizedBox(height: 10),
                  // BOTONES SECUNDARIOS: RECARGAR Y RESTAURAR
                  Row(
                    children: [
                      Expanded(
                        child: OutlinedButton.icon(
                          style: OutlinedButton.styleFrom(
                            side: const BorderSide(color: Colors.white24),
                            foregroundColor: Colors.white,
                            shape: RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(12),
                            ),
                            padding: const EdgeInsets.symmetric(vertical: 12),
                          ),
                          icon: const Icon(Icons.refresh, size: 18),
                          label: const Text(
                            'Recargar',
                            style: TextStyle(fontWeight: FontWeight.w700),
                          ),
                          onPressed: () async {
                            Navigator.of(ctx).pop();
                            await _controller.clearCache();
                            _controller.reload();
                          },
                        ),
                      ),
                      const SizedBox(width: 8),
                      Expanded(
                        child: OutlinedButton.icon(
                          style: OutlinedButton.styleFrom(
                            side: const BorderSide(color: Color(0xFFF59E0B)),
                            foregroundColor: const Color(0xFFF59E0B),
                            shape: RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(12),
                            ),
                            padding: const EdgeInsets.symmetric(vertical: 12),
                          ),
                          icon: const Icon(Icons.restore, size: 18),
                          label: const Text(
                            'Restaurar APK',
                            style: TextStyle(fontWeight: FontWeight.w700),
                          ),
                          onPressed: () async {
                            final messenger = ScaffoldMessenger.of(
                              this.context,
                            );
                            final nav = Navigator.of(ctx);
                            final prefs = await SharedPreferences.getInstance();
                            await prefs.setBool('pref_is_live_mode', false);
                            final assetsDir = await _getWebAssetsDir();
                            if (await assetsDir.exists()) {
                              await assetsDir.delete(recursive: true);
                            }
                            await _checkDownloadedAssets();

                            if (!mounted) return;
                            setState(() {
                              _isLiveMode = false;
                            });

                            nav.pop();
                            await _controller.clearCache();
                            _controller.loadRequest(
                              Uri.parse(
                                'http://127.0.0.1:$port/index.html?v=${DateTime.now().millisecondsSinceEpoch}',
                              ),
                            );
                            messenger.showSnackBar(
                              SnackBar(
                                content: const Text(
                                  'Restablecido a los archivos originales de la APK',
                                ),
                                backgroundColor: Colors.amber,
                                behavior: SnackBarBehavior.floating,
                                shape: RoundedRectangleBorder(
                                  borderRadius: BorderRadius.circular(14),
                                ),
                                margin: const EdgeInsets.fromLTRB(
                                  14,
                                  0,
                                  14,
                                  14,
                                ),
                              ),
                            );
                          },
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 12),
                  // SECCIÓN AVANZADA COLAPSABLE: WI-FI LOCAL OPCIONAL
                  GestureDetector(
                    onTap: () =>
                        setModalState(() => showAdvanced = !showAdvanced),
                    child: Row(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        AnimatedSwitcher(
                          duration: const Duration(milliseconds: 220),
                          reverseDuration: const Duration(milliseconds: 160),
                          switchInCurve: Curves.easeOutCubic,
                          switchOutCurve: Curves.easeInCubic,
                          transitionBuilder: (child, anim) =>
                              FadeTransition(opacity: anim, child: child),
                          child: Icon(
                            showAdvanced
                                ? Icons.expand_less
                                : Icons.expand_more,
                            key: ValueKey<bool>(showAdvanced),
                            color: Colors.white38,
                            size: 18,
                          ),
                        ),
                        const SizedBox(width: 4),
                        Text(
                          showAdvanced
                              ? 'Ocultar Opciones Avanzadas'
                              : 'Opciones Avanzadas (IP Local)',
                          style: const TextStyle(
                            color: Colors.white38,
                            fontSize: 12,
                            fontWeight: FontWeight.w600,
                          ),
                        ),
                      ],
                    ),
                  ),
                  AnimatedCrossFade(
                    firstChild: const SizedBox.shrink(),
                    secondChild: Padding(
                      padding: const EdgeInsets.only(top: 12),
                      child: Column(
                        children: [
                          TextField(
                            controller: textController,
                            decoration: InputDecoration(
                              labelText: 'IP y Puerto del PC (Opcional)',
                              hintText: 'Ej: 192.168.1.15:8000',
                              prefixIcon: const Icon(
                                Icons.computer,
                                color: Color(0xFF00E5FF),
                              ),
                              border: OutlineInputBorder(
                                borderRadius: BorderRadius.circular(12),
                                borderSide: const BorderSide(
                                  color: Colors.white24,
                                ),
                              ),
                              filled: true,
                              fillColor: const Color(0xFF070913),
                            ),
                          ),
                          const SizedBox(height: 10),
                          SizedBox(
                            width: double.infinity,
                            height: 42,
                            child: TextButton.icon(
                              style: TextButton.styleFrom(
                                foregroundColor: const Color(0xFF00E5FF),
                                backgroundColor: const Color(
                                  0xFF00E5FF,
                                ).withValues(alpha: 0.1),
                                shape: RoundedRectangleBorder(
                                  borderRadius: BorderRadius.circular(10),
                                ),
                              ),
                              icon: const Icon(Icons.wifi_tethering, size: 18),
                              label: const Text(
                                'Conectar a Servidor Wi-Fi Local',
                              ),
                              onPressed: () async {
                                final host = textController.text.trim();
                                if (host.isEmpty) return;
                                final messenger = ScaffoldMessenger.of(
                                  this.context,
                                );
                                final nav = Navigator.of(ctx);

                                final prefs =
                                    await SharedPreferences.getInstance();
                                await prefs.setString('pref_pc_host', host);
                                await prefs.setBool('pref_is_live_mode', true);

                                if (!mounted) return;
                                setState(() {
                                  _pcHost = host;
                                  _isLiveMode = true;
                                });

                                nav.pop();
                                _controller.loadRequest(
                                  Uri.parse('http://$host/index.html'),
                                );
                                messenger.showSnackBar(
                                  SnackBar(
                                    content: Text(
                                      'Conectando a http://$host/index.html',
                                    ),
                                    behavior: SnackBarBehavior.floating,
                                    shape: RoundedRectangleBorder(
                                      borderRadius: BorderRadius.circular(14),
                                    ),
                                    margin: const EdgeInsets.fromLTRB(
                                      14,
                                      0,
                                      14,
                                      14,
                                    ),
                                  ),
                                );
                              },
                            ),
                          ),
                        ],
                      ),
                    ),
                    crossFadeState: showAdvanced
                        ? CrossFadeState.showSecond
                        : CrossFadeState.showFirst,
                    duration: const Duration(milliseconds: 320),
                    reverseDuration: const Duration(milliseconds: 220),
                    firstCurve: Curves.easeInCubic,
                    secondCurve: Curves.easeOutCubic,
                    sizeCurve: Curves.easeInOutCubic,
                  ),
                ],
              ),
            );
          },
        );
      },
    );
  }

  @override
  void dispose() {
    _server?.close(force: true);
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SafeArea(
        child: Stack(
          children: [
            if (_server != null) WebViewWidget(controller: _controller),
            // Splash nativo animado: se oculta con fade-out gradual (GPU: opacity + transform)
            AnimatedOpacity(
              opacity: _splashVisible ? 1.0 : 0.0,
              duration: const Duration(milliseconds: 420),
              curve: Curves.easeOutCubic,
              onEnd: () {},
              child: AnimatedContainer(
                duration: const Duration(milliseconds: 380),
                curve: Curves.easeOutCubic,
                transform: Matrix4.diagonal3Values(
                  _splashVisible ? 1.0 : 1.04,
                  _splashVisible ? 1.0 : 1.04,
                  1.0,
                ),
                transformAlignment: Alignment.center,
                child: IgnorePointer(
                  ignoring: !_splashVisible,
                  child: Container(
                    color: const Color(0xFF0B1120),
                    child: Center(
                      child: Column(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          // Logo con Tween entrada spring-like (escala)
                          TweenAnimationBuilder<double>(
                            tween: Tween(begin: 0.82, end: 1.0),
                            duration: const Duration(milliseconds: 850),
                            curve: Curves.elasticOut,
                            builder: (ctx, val, child) {
                              return Opacity(
                                opacity: (val - 0.82) / 0.18,
                                child: Transform.scale(
                                  scale: val,
                                  child: Image.asset(
                                    'assets/web/logo_simbolo.png',
                                    width: 108,
                                    height: 108,
                                    fit: BoxFit.contain,
                                  ),
                                ),
                              );
                            },
                          ),
                          const SizedBox(height: 16),
                          // Título Gestual Vision animado
                          TweenAnimationBuilder<double>(
                            tween: Tween(begin: 0, end: 1),
                            duration: const Duration(milliseconds: 700),
                            curve: Curves.easeOutBack,
                            builder: (ctx, val, _) {
                              return Opacity(
                                opacity: val,
                                child: Transform.translate(
                                  offset: Offset(0, (1 - val) * 18),
                                  child: const Text.rich(
                                    TextSpan(
                                      children: [
                                        TextSpan(
                                          text: 'Gestual',
                                          style: TextStyle(
                                            color: Colors.white,
                                            fontSize: 28,
                                            fontWeight: FontWeight.w900,
                                            letterSpacing: 0.4,
                                          ),
                                        ),
                                        TextSpan(
                                          text: ' Vision',
                                          style: TextStyle(
                                            color: Color(0xFF00E5FF),
                                            fontSize: 28,
                                            fontWeight: FontWeight.w900,
                                            letterSpacing: 0.4,
                                          ),
                                        ),
                                      ],
                                    ),
                                  ),
                                ),
                              );
                            },
                          ),
                          const SizedBox(height: 6),
                          TweenAnimationBuilder<double>(
                            tween: Tween(begin: 0, end: 1),
                            duration: const Duration(milliseconds: 850),
                            curve: Curves.easeOut,
                            builder: (ctx, val, _) => Opacity(
                              opacity: val,
                              child: const Text(
                                '🇨🇴 Lengua de Señas Colombiana',
                                style: TextStyle(
                                  color: Color(0xFF94A3B8),
                                  fontSize: 12,
                                  fontWeight: FontWeight.w700,
                                ),
                              ),
                            ),
                          ),
                          const SizedBox(height: 30),
                          // Loader orbit tricolor Colombia (amarillo • azul • rojo)
                          const _TricolorOrbitLoader(),
                          const SizedBox(height: 18),
                          TweenAnimationBuilder<double>(
                            tween: Tween(begin: 0, end: 1),
                            duration: const Duration(milliseconds: 900),
                            curve: Curves.easeOut,
                            builder: (ctx, val, _) => Opacity(
                              opacity: val,
                              child: Padding(
                                padding: const EdgeInsets.symmetric(
                                  horizontal: 36,
                                ),
                                child: Text(
                                  _statusMessage,
                                  textAlign: TextAlign.center,
                                  style: const TextStyle(
                                    color: Color(0xFF94A3B8),
                                    fontSize: 13,
                                    fontWeight: FontWeight.w600,
                                  ),
                                ),
                              ),
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

/// Loader orbital tricolor Colombia (solo transform y opacity — GPU-safe)
class _TricolorOrbitLoader extends StatefulWidget {
  const _TricolorOrbitLoader();

  @override
  State<_TricolorOrbitLoader> createState() => _TricolorOrbitLoaderState();
}

class _TricolorOrbitLoaderState extends State<_TricolorOrbitLoader>
    with SingleTickerProviderStateMixin {
  late final AnimationController _ctrl;

  @override
  void initState() {
    super.initState();
    _ctrl = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 1350),
    )..repeat();
  }

  @override
  void dispose() {
    _ctrl.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      width: 48,
      height: 48,
      child: AnimatedBuilder(
        animation: _ctrl,
        builder: (ctx, _) {
          return Stack(
            clipBehavior: Clip.none,
            children: [
              _dot(const Color(0xFFF59E0B), _ctrl.value * 360),
              _dot(const Color(0xFF1D4ED8), _ctrl.value * 360 + 120),
              _dot(const Color(0xFFE11D48), _ctrl.value * 360 + 240),
              Center(
                child: Container(
                  width: 10,
                  height: 10,
                  decoration: const BoxDecoration(
                    color: Color(0xFF00E5FF),
                    shape: BoxShape.circle,
                  ),
                ),
              ),
            ],
          );
        },
      ),
    );
  }

  Widget _dot(Color color, double degrees) {
    return Positioned.fill(
      child: Transform.rotate(
        angle: degrees * 3.1415926535 / 180,
        child: const Align(alignment: Alignment.topCenter, child: _Dot()),
      ),
    );
  }
}

class _Dot extends StatelessWidget {
  const _Dot();
  @override
  Widget build(BuildContext context) {
    return Container(
      width: 10,
      height: 10,
      margin: const EdgeInsets.only(top: 2),
      decoration: const BoxDecoration(
        color: Colors.transparent,
        shape: BoxShape.circle,
        boxShadow: [],
      ),
      child: Container(
        decoration: BoxDecoration(
          color:
              context
                  .dependOnInheritedWidgetOfExactType<_InheritedDotColor>()
                  ?.color ??
              const Color(0xFFF59E0B),
          shape: BoxShape.circle,
        ),
      ),
    );
  }
}

class _InheritedDotColor extends InheritedWidget {
  final Color color;
  const _InheritedDotColor({required this.color, required super.child});
  @override
  bool updateShouldNotify(covariant _InheritedDotColor old) =>
      old.color != color;
}
