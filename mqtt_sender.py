import paho.mqtt.client as mqtt
import time

client = mqtt.Client()
client.connect("broker.hivemq.com", 1883)  # Free broker

def send_alert():
    client.publish("drowning/alert", "DROWNING DETECTED!")

# Simulate detection every 10s
while True:
    send_alert()
    print("Alert sent")
    time.sleep(10)