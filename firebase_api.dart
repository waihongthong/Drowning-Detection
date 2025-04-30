import 'package:firebase_core/firebase_core.dart';
import 'package:firebase_messaging/firebase_messaging.dart';
import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'dart:convert';

final GlobalKey<NavigatorState> navigatorKey = GlobalKey<NavigatorState>();

class FirebaseApi {
  final _firebaseMessaging = FirebaseMessaging.instance;

  // Call this in main() to set up everything
  Future<void> initNotifications() async {
    await Firebase.initializeApp();

    // Request permissions (for iOS)
    await _firebaseMessaging.requestPermission();

    // Get the token (for testing or targeting a specific device)
    final fcmToken = await _firebaseMessaging.getToken();
    print('FCM Token: $fcmToken');

    // Handle foreground messages
    FirebaseMessaging.onMessage.listen(_handleForegroundMessage);

    // Handle background messages
    FirebaseMessaging.onBackgroundMessage(_handleBackgroundMessage);
  }

  // Called when app is in foreground
  void _handleForegroundMessage(RemoteMessage message) {
    print('📥 Foreground message: ${message.notification?.title}');

    if (navigatorKey.currentState != null) {
      final context = navigatorKey.currentState!.overlay!.context;

      // Show in-app snackbar
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text(message.notification?.body ?? 'New alert received!'),
          backgroundColor: Colors.redAccent,
        ),
      );
    }

    // Save to local history
    _addToNotificationHistory(
      message.notification?.title,
      message.notification?.body,
    );
  }

  // Called when app is in background or terminated
  static Future<void> _handleBackgroundMessage(RemoteMessage message) async {
    print('📥 Background message: ${message.notification?.title}');

    // Ensure SharedPreferences is ready
    final prefs = await SharedPreferences.getInstance();
    final now = DateTime.now().toIso8601String();

    final newEntry = {
      'title': message.notification?.title ?? 'No title',
      'body': message.notification?.body ?? 'No body',
      'time': now,
    };

    final existing = prefs.getStringList('notification_history') ?? [];
    existing.add(jsonEncode(newEntry));
    await prefs.setStringList('notification_history', existing);
  }

  // Store notification in local history
  void _addToNotificationHistory(String? title, String? body) async {
    final prefs = await SharedPreferences.getInstance();
    final now = DateTime.now().toIso8601String();

    final newEntry = {
      'title': title ?? 'No title',
      'body': body ?? 'No body',
      'time': now,
    };

    final existing = prefs.getStringList('notification_history') ?? [];
    existing.add(jsonEncode(newEntry));
    await prefs.setStringList('notification_history', existing);
  }
}
