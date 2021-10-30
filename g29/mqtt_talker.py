#!/usr/bin/env python3
# encoding: utf-8

import paho.mqtt.client as mqtt
import time
import os, sys, signal
import json

class MQTTMessage(object):
    def __init__(self, msg_name):
        self.name = msg_name
        self.payload = None
        self.is_refreshed = False
    
    def update(self, msg):
        self.payload = msg
        self.is_refreshed = True
    
    def get(self):
        if self.is_refreshed is True:
            self.is_refreshed = False
            return 1, self.payload
        elif self.payload == None:
            return -1, None
        else:
            return 2, self.payload

class MQTTClient(object):
    def __init__(self, name):
        self.FILE_PATH = os.path.abspath(os.path.dirname(__file__))
        self._MQTT_HOST = "119.28.51.55"
        self._MQTT_PORT = 1883
        self._MQTT_KEPPALIVE = 60
        self._MQTT_USERNAME = "admin"
        self._MQTT_PASSWORD = "public"
        self._client = mqtt.Client(name)
        self._client.username_pw_set(self._MQTT_USERNAME, self._MQTT_PASSWORD)
        self._client.on_connect = self.on_connect
        self._client.on_message = self.on_message
        self._client.on_log = self.on_log
        print('[NovaCloud]Connecting')
        ret = self._client.connect(self._MQTT_HOST, self._MQTT_PORT, self._MQTT_KEPPALIVE)
        if ret == 0:
            print('[NovaCloud]Connected')
        else:
            raise RuntimeError
        self.msgSet = {}

    def on_log(self, client, userdata, level, buf):
        pass

    def on_disconnect(self, client, userdata, rc):
        pass

    def on_connect(self, client, userdata, flags, rc):
        pass

    def on_message(self, client, userdata, msg):
        print(msg)

    def add_subscription(self, topic):
        self._client.subscribe(topic,0)

    def publish(self, topic, payload):
        self._client.publish(topic, payload)

    def start(self):
        self._client.loop_start()

    def spin(self):
        self._client.loop_forever()

"""
地点Ａ发送：

在这里的while循环实现pygame对方向盘信号的读取，可以通过ros.rate或其他方式控制频率。用json序列化后发送到topic
hello_world上。
"""
if __name__ == '__main__':
    client = MQTTClient("xp_talker")
    steering = 0
    throttle = 0
    while True:
        time.sleep(0.1)

        data = {
            "steering":steering, 
            "throttle": throttle
        }

        client.publish("hello_world", json.dumps(data))
        print("published:{}".format(json.dumps(data)))
        steering += 0.01
        throttle += 0.01

