#!/usr/bin/env python3
# encoding: utf-8

import paho.mqtt.client as mqtt
import time
import json
import os, sys, signal


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


    
    """
    地点Ｂ接收：

    on_message累似于ros的回调函数，在构造函数里　self._client.on_message = self.on_message 这行注册。
    每次收到消息，都会调用该函数，在这里对信息反序列化得到信息。
    """
    def on_message(self, client, userdata, msg):
        data = json.loads(msg.payload)
        print("received:steering={}, throttle={}, braking={}".format(data["steering"], data["throttle"], data["braking"]))

    def add_subscription(self, topic):
        self._client.subscribe(topic,0)
    
    def publish(self, topic, payload):
        self._client.publish(topic, payload)

    def start(self):
        self._client.loop_start()

    def spin(self):
        self._client.loop_forever()

if __name__ == '__main__':
    client = MQTTClient("xp_listener")
    client.add_subscription("hello_world")
    client.spin()

