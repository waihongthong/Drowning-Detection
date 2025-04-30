import 'package:flutter/material.dart';
import 'dart:convert';
import 'package:webview_flutter/webview_flutter.dart';
import 'package:webview_flutter_wkwebview/webview_flutter_wkwebview.dart';
import 'package:http/http.dart' as http;
import 'dart:async';

class HomeScreen extends StatefulWidget {
  @override
  _HomeScreenState createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  // Raspberry Pi IP and port
  final String raspberryPiIP =
      '192.168.1.100'; // Change to actual Raspberry Pi IP
  final int raspberryPiPort = 5000;

  bool _isConnected = false;
  bool _drowningDetected = false;
  Timer? _statusCheckTimer;

  late final WebViewController _controller;

  @override
  void initState() {
    super.initState();
    _startStatusCheck();
    _initWebView();
  }

  void _initWebView() {
    late final PlatformWebViewControllerCreationParams params;
    if (WebViewPlatform.instance is WebKitWebViewPlatform) {
      params = WebKitWebViewControllerCreationParams(
        allowsInlineMediaPlayback: true,
        mediaTypesRequiringUserAction: const <PlaybackMediaTypes>{},
      );
    } else {
      params = const PlatformWebViewControllerCreationParams();
    }

    final WebViewController controller = WebViewController.fromPlatformCreationParams(params);
    _controller = controller;
  }

  void _startStatusCheck() {
    // Check status every second
    _statusCheckTimer = Timer.periodic(Duration(seconds: 1), (timer) {
      _checkDrowningStatus();
    });
  }

  Future<void> _checkDrowningStatus() async {
    try {
      final response = await http
          .get(
            Uri.parse('http://$raspberryPiIP:$raspberryPiPort/api/status'),
          )
          .timeout(Duration(seconds: 3));

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        setState(() {
          _isConnected = true;
          _drowningDetected = data['drowning_detected'] ?? false;
        });
      } else {
        setState(() {
          _isConnected = false;
        });
      }
    } catch (e) {
      setState(() {
        _isConnected = false;
      });
      print('Error checking status: $e');
    }
  }

  @override
  void dispose() {
    _statusCheckTimer?.cancel();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Pool Monitor'),
        backgroundColor: _drowningDetected ? Colors.red : Colors.blue,
        actions: [
          // Connection status indicator
          Padding(
            padding: const EdgeInsets.all(8.0),
            child: Center(
              child: Container(
                padding: EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                decoration: BoxDecoration(
                  color: _isConnected ? Colors.green : Colors.grey,
                  borderRadius: BorderRadius.circular(10),
                ),
                child: Text(
                  _isConnected ? 'Connected' : 'Disconnected',
                  style: TextStyle(color: Colors.white, fontSize: 12),
                ),
              ),
            ),
          ),
        ],
        elevation: 0,
      ),
      body: SafeArea(
        child: Padding(
          padding: EdgeInsets.all(16.0),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                'Drowning Detection System',
                style: TextStyle(
                  fontSize: 28,
                  fontWeight: FontWeight.bold,
                ),
              ),
              SizedBox(height: 8),
              Text(
                'Live monitoring of swimming area',
                style: TextStyle(
                  fontSize: 16,
                  color: Colors.grey[600],
                ),
              ),
              SizedBox(height: 20),

              // Video Stream Container
              Expanded(
                child: Card(
                  elevation: 5,
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(15),
                  ),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Container(
                        padding: EdgeInsets.all(16),
                        decoration: BoxDecoration(
                          color: _drowningDetected ? Colors.red : Colors.blue,
                          borderRadius: BorderRadius.only(
                            topLeft: Radius.circular(15),
                            topRight: Radius.circular(15),
                          ),
                        ),
                        child: Row(
                          children: [
                            Icon(
                              _drowningDetected
                                  ? Icons.warning
                                  : Icons.videocam,
                              color: Colors.white,
                            ),
                            SizedBox(width: 10),
                            Text(
                              _drowningDetected
                                  ? 'ALERT: Drowning Detected!'
                                  : 'Live Camera Feed',
                              style: TextStyle(
                                fontSize: 18,
                                fontWeight: FontWeight.w600,
                                color: Colors.white,
                              ),
                            ),
                          ],
                        ),
                      ),
                      Expanded(
                        child: ClipRRect(
                          borderRadius: BorderRadius.only(
                            bottomLeft: Radius.circular(15),
                            bottomRight: Radius.circular(15),
                          ),
                          child: _isConnected
                              ? Container(
                                  decoration: BoxDecoration(
                                    border: _drowningDetected
                                        ? Border.all(
                                            color: Colors.red, width: 4)
                                        : null,
                                  ),
                                  child: WebViewWidget(controller: _controller),
                                )
                              : Center(
                                  child: Column(
                                    mainAxisAlignment: MainAxisAlignment.center,
                                    children: [
                                      Icon(
                                        Icons.signal_wifi_off,
                                        size: 50,
                                        color: Colors.grey,
                                      ),
                                      SizedBox(height: 16),
                                      Text(
                                        'Cannot connect to Raspberry Pi',
                                        style: TextStyle(
                                            color: Colors.grey.shade600),
                                      ),
                                      SizedBox(height: 8),
                                      Text(
                                        'Check if the Pi is running and connected',
                                        style: TextStyle(
                                            color: Colors.grey.shade500,
                                            fontSize: 12),
                                      ),
                                      SizedBox(height: 20),
                                      ElevatedButton(
                                        onPressed: _checkDrowningStatus,
                                        child: Text('Retry Connection'),
                                        style: ElevatedButton.styleFrom(
                                          backgroundColor: Colors.blue,
                                          shape: RoundedRectangleBorder(
                                            borderRadius:
                                                BorderRadius.circular(10),
                                          ),
                                        ),
                                      ),
                                    ],
                                  ),
                                ),
                        ),
                      ),
                    ],
                  ),
                ),
              ),

              // Alert container (shows only when drowning is detected)
              if (_drowningDetected)
                Container(
                  margin: EdgeInsets.only(top: 16),
                  padding: EdgeInsets.all(16),
                  decoration: BoxDecoration(
                    color: Colors.red.shade50,
                    border: Border.all(color: Colors.red, width: 2),
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: Row(
                    children: [
                      Icon(Icons.warning_amber_rounded,
                          color: Colors.red, size: 32),
                      SizedBox(width: 12),
                      Expanded(
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text(
                              'DROWNING ALERT',
                              style: TextStyle(
                                fontSize: 18,
                                fontWeight: FontWeight.bold,
                                color: Colors.red,
                              ),
                            ),
                            SizedBox(height: 4),
                            Text(
                              'Possible drowning detected. Please check the swimming area immediately!',
                              style: TextStyle(fontSize: 14),
                            ),
                          ],
                        ),
                      ),
                    ],
                  ),
                ),

              SizedBox(height: 16),
            ],
          ),
        ),
      ),
    );
  }
}
