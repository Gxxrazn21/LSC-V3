import 'dart:convert';
import 'dart:io';
import 'package:crypto/crypto.dart';
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

class OtaUpdateResult {
  final bool success;
  final String fromVersion;
  final String toVersion;
  final String? errorMessage;

  const OtaUpdateResult({
    required this.success,
    required this.fromVersion,
    required this.toVersion,
    this.errorMessage,
  });
}

class LSCHomePage extends StatefulWidget {
  const LSCHomePage({super.key});

  @override
  State<LSCHomePage> createState() => _LSCHomePageState();
}

class _LSCHomePageState extends State<LSCHomePage> {
  static const _nativeChannel = MethodChannel('com.lsc.app/native');
  static const _shellVersion = '8.3.0';
  HttpServer? _server;
  late final WebViewController _controller;
  bool _splashVisible = true; // controla el fade-out gradual del splash nativo
  String _statusMessage = 'Iniciando Gestual Vision AI v7.0...';

  // Gestión de Live Sync y Recursos OTA
  String _pcHost = '192.168.1.15:8000';
  bool _isLiveMode = false;

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

  Future<bool> _checkDownloadedAssets() async {
    try {
      final activeAssets = await _getActiveAssetsDir();
      final indexFile = activeAssets == null
          ? null
          : File('${activeAssets.path}/index.html');
      final exists = indexFile != null && await indexFile.exists();
      return exists;
    } catch (_) {
      return false;
    }
  }

  Future<Directory?> _getActiveAssetsDir() async {
    final prefs = await SharedPreferences.getInstance();
    final version = prefs.getString('pref_active_release');
    if (version == null || !RegExp(r'^[0-9A-Za-z._-]+$').hasMatch(version)) {
      return null;
    }
    // Un release OTA más viejo que el que trae la APK nunca debe tapar al empaquetado
    if (_compareVersions(version, _shellVersion) <= 0) return null;
    final root = await _getWebAssetsDir();
    final candidate = Directory('${root.path}/releases/$version');
    return await candidate.exists() ? candidate : null;
  }

  static int _compareVersions(String a, String b) {
    List<int> parse(String value) => value
        .split('+').first
        .split('-').first
        .split('.')
        .map((part) => int.tryParse(part) ?? 0)
        .toList();
    final pa = parse(a), pb = parse(b);
    for (var i = 0; i < 3; i++) {
      final x = i < pa.length ? pa[i] : 0;
      final y = i < pb.length ? pb[i] : 0;
      if (x != y) return x.compareTo(y);
    }
    return 0;
  }

  bool _isSafeAssetPath(String value) =>
      value.isNotEmpty &&
      !value.contains('..') &&
      !value.contains('\\') &&
      value == Uri.encodeComponent(value);

  bool _isShellCompatible(String minimum) {
    List<int> parse(String value) => value
        .split('+').first
        .split('-').first
        .split('.')
        .map((part) => int.tryParse(part) ?? 0)
        .toList();
    final actual = parse(_shellVersion);
    final required = parse(minimum);
    for (var index = 0; index < 3; index++) {
      final a = index < actual.length ? actual[index] : 0;
      final b = index < required.length ? required[index] : 0;
      if (a != b) return a > b;
    }
    return true;
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
        AndroidWebViewController.enableDebugging(kDebugMode);
        (controller.platform as AndroidWebViewController)
            .setMediaPlaybackRequiresUserGesture(false);
        // Tamaño de letra del sistema fijo al 100 %: la página ajusta sus textos sola
        // y así ninguna palabra se desborda ni se parte con la "letra grande" de Android.
        (controller.platform as AndroidWebViewController).setTextZoom(100);
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
    if (!_isSafeAssetPath(path)) {
      return Response.forbidden('Ruta de recurso no permitida');
    }

    try {
      Uint8List bytes;
      final activeAssets = await _getActiveAssetsDir();
      final localFile = activeAssets == null
          ? null
          : File('${activeAssets.path}/$path');

      // Si existe recurso descargado por OTA en almacenamiento del teléfono, servirlo primero
      if (localFile != null && await localFile.exists()) {
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
      } else if (path.endsWith('.woff2')) {
        mime = 'font/woff2';
      } else if (path.endsWith('.glb')) {
        mime = 'model/gltf-binary';
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
      'https://raw.githubusercontent.com/Gxxrazn21/LSC-V3/main/web';

  Future<OtaUpdateResult> _downloadResourcesFromCloud({String? baseUrl}) async {
    final prefs = await SharedPreferences.getInstance();
    final fromVersion = prefs.getString('pref_active_release') ?? _shellVersion;

    final bases = [
      (baseUrl != null && baseUrl.isNotEmpty) ? baseUrl : _defaultCloudBase,
      'https://raw.githubusercontent.com/Gxxrazn21/LSC-V3/main/estilo',
    ];

    String lastError = 'No se pudo conectar con el repositorio';

    for (final base in bases) {
      HttpClient? client;
      try {
        client = HttpClient();
        client.connectionTimeout = const Duration(seconds: 45);
        final baseUri = Uri.parse(base.endsWith('/') ? base : '$base/');
        if (baseUri.scheme != 'https') continue;

        final manifestRequest = await client.getUrl(
          baseUri.resolve('release-manifest.json?nocache=${DateTime.now().millisecondsSinceEpoch}'),
        );
        final manifestResponse = await manifestRequest.close();
        if (manifestResponse.statusCode != 200) {
          lastError = 'HTTP ${manifestResponse.statusCode} al obtener release-manifest.json';
          continue;
        }

        final manifestBytes = await consolidateHttpClientResponseBytes(manifestResponse);
        final manifest = jsonDecode(utf8.decode(manifestBytes)) as Map<String, dynamic>;
        if (manifest['schema_version'] != 1 ||
            manifest['version'] is! String ||
            !_isShellCompatible(manifest['min_shell_version'] as String? ?? '0.0.0')) {
          lastError = 'Manifiesto no compatible con la versión de la APK';
          continue;
        }

        final toVersion = manifest['version'] as String;
        final files = manifest['files'];
        if (!RegExp(r'^[0-9A-Za-z._-]+$').hasMatch(toVersion) || files is! Map) {
          lastError = 'Versión de manifiesto inválida';
          continue;
        }
        // Solo instalar si la nube trae algo más nuevo que lo que ya corre
        final activa = prefs.getString('pref_active_release');
        final actual = (activa != null && _compareVersions(activa, _shellVersion) > 0) ? activa : _shellVersion;
        if (_compareVersions(toVersion, actual) <= 0) {
          lastError = 'Ya tienes la versión más reciente (v$actual)';
          continue;
        }

        final assetsRoot = await _getWebAssetsDir();
        final staging = Directory('${assetsRoot.path}/.staging-$toVersion');
        if (await staging.exists()) await staging.delete(recursive: true);
        await staging.create(recursive: true);

        try {
          for (final entry in files.entries) {
            final fileName = entry.key;
            final details = entry.value;
            if (fileName is! String || !_isSafeAssetPath(fileName) || details is! Map) {
              throw const FormatException('Manifiesto OTA inválido');
            }
            final expectedHash = details['sha256'];
            if (expectedHash is! String || !RegExp(r'^[a-f0-9]{64}$').hasMatch(expectedHash)) {
              throw const FormatException('Checksum OTA inválido');
            }
            final request = await client.getUrl(baseUri.resolve(fileName));
            final response = await request.close();
            if (response.statusCode != 200) {
              throw HttpException('No se pudo descargar $fileName (HTTP ${response.statusCode})');
            }
            final bytes = await consolidateHttpClientResponseBytes(response);
            if (sha256.convert(bytes).toString() != expectedHash) {
              throw const FormatException('Checksum OTA no coincide');
            }
            await File('${staging.path}/$fileName').writeAsBytes(bytes, flush: true);
          }

          if (!await File('${staging.path}/index.html').exists()) {
            throw const FormatException('El release no contiene index.html');
          }

          final release = Directory('${assetsRoot.path}/releases/$toVersion');
          if (await release.exists()) await release.delete(recursive: true);
          await release.parent.create(recursive: true);
          await staging.rename(release.path);

          await prefs.setString('pref_previous_version', fromVersion);
          await prefs.setString('pref_active_release', toVersion);
          await _checkDownloadedAssets();

          return OtaUpdateResult(
            success: true,
            fromVersion: fromVersion,
            toVersion: toVersion,
          );
        } catch (e) {
          if (await staging.exists()) await staging.delete(recursive: true);
          lastError = e.toString();
          debugPrint('Error descargando archivos OTA desde $base: $e');
        }
      } catch (e) {
        lastError = e.toString();
        debugPrint('Error conectando a $base: $e');
      } finally {
        client?.close(force: true);
      }
    }

    return OtaUpdateResult(
      success: false,
      fromVersion: fromVersion,
      toVersion: fromVersion,
      errorMessage: lastError,
    );
  }

  void _showSyncModal() async {
    final prefs = await SharedPreferences.getInstance();
    final currentInstalledVersion = prefs.getString('pref_active_release') ?? _shellVersion;
    final previousVersion = prefs.getString('pref_previous_version');
    final textController = TextEditingController(text: _pcHost);
    bool downloading = false;
    bool showAdvanced = false;

    if (!mounted) return;

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
            final isLatest = currentInstalledVersion == '7.0.0';

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
                            'Actualización & Nube LSC',
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
                      border: Border.all(
                        color: isLatest
                            ? const Color(0xFF00FF9D).withValues(alpha: 0.3)
                            : const Color(0xFF00E5FF).withValues(alpha: 0.3),
                      ),
                    ),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Row(
                          children: [
                            Icon(
                              isLatest ? Icons.check_circle : Icons.system_update,
                              color: isLatest
                                  ? const Color(0xFF00FF9D)
                                  : const Color(0xFF00E5FF),
                              size: 20,
                            ),
                            const SizedBox(width: 8),
                            Expanded(
                              child: Text(
                                isLatest
                                    ? 'Versión Activa: v$currentInstalledVersion (Al día)'
                                    : 'Actualización Disponible: v$currentInstalledVersion ➔ v7.0.0',
                                style: TextStyle(
                                  fontSize: 13,
                                  fontWeight: FontWeight.bold,
                                  color: isLatest
                                      ? const Color(0xFF00FF9D)
                                      : const Color(0xFF00E5FF),
                                ),
                              ),
                            ),
                          ],
                        ),
                        const SizedBox(height: 6),
                        Text(
                          isLatest
                              ? (previousVersion != null
                                  ? 'Has pasado exitosamente de v$previousVersion a v$currentInstalledVersion.'
                                  : 'Ejecutando la versión más reciente con Invarianza Espacial y 49 clases.')
                              : 'Al actualizar pasarás de la versión v$currentInstalledVersion a la versión v7.0.0.',
                          style: const TextStyle(
                            fontSize: 12,
                            color: Colors.white70,
                          ),
                        ),
                      ],
                    ),
                  ),
                  const SizedBox(height: 16),
                  // BOTÓN 1 PRINCIPAL: ACTUALIZACIÓN AUTOMÁTICA DESDE LA NUBE
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
                            ? 'Descargando e instalando modelo v7.0.0...'
                            : (isLatest
                                ? '☁️ Reinstalar / Sincronizar v7.0.0'
                                : '☁️ Actualizar: Pasar de v$currentInstalledVersion a v7.0.0'),
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
                              final result = await _downloadResourcesFromCloud();
                              setModalState(() => downloading = false);

                              if (result.success) {
                                final p = await SharedPreferences.getInstance();
                                await p.setBool('pref_is_live_mode', false);

                                if (!mounted) return;
                                setState(() {
                                  _isLiveMode = false;
                                });

                                nav.pop();
                                await _controller.clearCache();
                                _controller.loadRequest(
                                  Uri.parse(
                                    'http://127.0.0.1:$port/index.html?from=${result.fromVersion}&to=${result.toVersion}&v=${DateTime.now().millisecondsSinceEpoch}',
                                  ),
                                );

                                // Diálogo y SnackBar explícitos indicando la transición de versión
                                messenger.showSnackBar(
                                  SnackBar(
                                    content: Row(
                                      children: [
                                        const Icon(
                                          Icons.check_circle,
                                          color: Colors.white,
                                        ),
                                        const SizedBox(width: 10),
                                        Expanded(
                                          child: Text(
                                            '🎉 ¡Actualizado con éxito! Pasaste de v${result.fromVersion} a v${result.toVersion}.',
                                            style: const TextStyle(
                                              fontWeight: FontWeight.bold,
                                              fontSize: 13.5,
                                            ),
                                          ),
                                        ),
                                      ],
                                    ),
                                    backgroundColor: const Color(0xFF10B981),
                                    duration: const Duration(seconds: 5),
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
                                    content: Text(
                                      '❌ Error al descargar: ${result.errorMessage ?? "Verifica tu conexión"}',
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
                            await prefs.remove('pref_active_release');
                            await prefs.remove('pref_previous_version');
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
                                opacity: ((val - 0.82) / 0.18).clamp(0.0, 1.0),
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
                                opacity: val.clamp(0.0, 1.0),
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
                              opacity: val.clamp(0.0, 1.0),
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
                              opacity: val.clamp(0.0, 1.0),
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
