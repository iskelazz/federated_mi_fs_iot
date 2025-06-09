# FEDERATED_MUTUAL_INFORMATION_FS/mqtt_handlers/mqtt_communicator.py
import logging
import paho.mqtt.client as mqtt
import json
import time
import numpy as np

class MQTTCommunicator:
    def __init__(self, broker_address, port, client_id_prefix="mqtt_client", publish_callback=None):
        self.broker_address = broker_address
        self.port = port
        timestamp = time.strftime('%Y%m%d%H%M%S')
        random_num = np.random.randint(1000, 9999)
        self.client_id = f"{client_id_prefix}-{timestamp}-{random_num}"
        self._pending = {}  # mid -> t0 (perf counter)
        
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=self.client_id)
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.client.on_disconnect = self._on_disconnect
        self.client.on_publish = self._on_publish

        self.message_callback = None 
        self.connect_callback = None
        self.disconnect_callback = None 
        self.publish_callback = publish_callback
        self._subscribed_topics_qos = {}

    def _on_connect(self, client, userdata, flags, rc, properties=None):
        if rc == 0:
            print(f"MQTTComm [{self.client_id}]: Conectado a {self.broker_address}:{self.port}")
            if self._subscribed_topics_qos:
                for topic, qos_level in self._subscribed_topics_qos.items():
                    self.client.subscribe(topic, qos=qos_level)
            if self.connect_callback:
                self.connect_callback()
        else:
            print(f"MQTTComm [{self.client_id}]: ERROR al conectar, código: {rc}")

    def _on_message(self, client, userdata, msg):
        if self.message_callback:
            self.message_callback(msg.topic, msg.payload)

    def _on_disconnect(self, client, userdata, flags, rc, properties=None):
        print(f"MQTTComm [{self.client_id}]: Desconectado del broker. Código: {rc}")
        if self.disconnect_callback:
            self.disconnect_callback(rc)

    def _on_publish(self, client, userdata, mid, rc, properties=None):
        tup = self._pending.pop(mid, None)
        if tup:
            t0, topic = tup
            elapsed_ms = (time.perf_counter() - t0)*1e3
            logging.info(f"[NET] topic={topic!r} mid={mid} {elapsed_ms:.2f} ms")

            # envía métrica de bench al topic ya existente
            bench = {"sim_client_id": self.client_id,
                     "net_s": elapsed_ms / 1_000}  # seg
            self.client.publish("tfg/fl/pi/bench", json.dumps(bench), qos=0)

        # Deja que externos se enganchen, si los hay
        if self.publish_callback:
            self.publish_callback(mid, rc)

#    def _on_publish(self, client, userdata, mid, rc=None, properties=None):
#        if self.publish_callback:
#            self.publish_callback(mid)

    def set_message_callback(self, callback): self.message_callback = callback
    def set_connect_callback(self, callback): self.connect_callback = callback
    def set_disconnect_callback(self, callback): self.disconnect_callback = callback
    def set_publish_callback(self, callback): self.publish_callback = callback

    def connect(self):
        """
            Intenta establecer una conexión con el broker MQTT. Devuelve True si tiene éxito, False en caso contrario.
        """
        try:
            self.client.connect(self.broker_address, self.port, 60)
            return True
        except Exception as e:
            print(f"MQTTComm [{self.client_id}]: ERROR al conectar - {e}")
            return False

    def start_listening(self): self.client.loop_start()
    def stop_listening(self): self.client.loop_stop()
    def loop_forever(self): self.client.loop_forever()
            
    def disconnect(self):
        print(f"MQTTComm [{self.client_id}]: Solicitando parada del bucle de red...")
        self.stop_listening() # Llama a self.client.loop_stop(), que es bloqueante
        print(f"MQTTComm [{self.client_id}]: Bucle de red detenido. Procediendo a desconectar cliente MQTT...")
        self.client.disconnect() # Solicita la desconexión al broker
        print(f"MQTTComm [{self.client_id}]: Llamada a client.disconnect() completada.")

   # def publish(self, topic, payload_data, qos=0, retain=False):
   #     """
   #         Publica un mensaje en el topic especificado. Convierte automáticamente diccionarios y listas a formato JSON string antes de enviar.  Admite payloads de tipo string, bytes o los convierte a string. Devuelve el resultado de la publicación de Paho-MQTT o None en caso de error.
   #     """
   #     payload_to_send = None
   #     if isinstance(payload_data, (dict, list)):
   #         payload_to_send = json.dumps(payload_data)
   #     elif isinstance(payload_data, str):
   #         payload_to_send = payload_data
   #     elif isinstance(payload_data, bytes):
   #         payload_to_send = payload_data
   #     else:
   #         payload_to_send = str(payload_data)
        
   #     try:
   #         return self.client.publish(topic, payload_to_send, qos=qos, retain=retain)
   #     except Exception as e:
   #         print(f"MQTTComm [{self.client_id}]: ERROR publicando en '{topic}': {e}")
   #         return None
        
    def publish(self, topic: str, payload: dict, qos: int = 1):
        t0 = time.perf_counter()
        info = self.client.publish(topic, json.dumps(payload), qos=qos)
        self._pending[info.mid] = (t0, topic)          # guardamos arranque
        return info   

    def subscribe(self, topic, qos=0):
        self.client.subscribe(topic, qos=qos)
        self._subscribed_topics_qos[topic] = qos

if __name__ == '__main__':
    print("MQTTCommunicator: Prueba directa iniciada.")
    TEST_BROKER = "localhost"; TEST_PORT = 1883
    def msg_h(t,p): print(f"TEST_HANDLER (Msg): '{t}' -> '{p.decode()[:50]}...'")
    def con_h(): print("TEST_HANDLER (Con): Conectado!"); comm.subscribe("test/sub",1); comm.publish("test/pub", {"status":"Comm Test Online"},1)
    comm = MQTTCommunicator(TEST_BROKER,TEST_PORT,"test_comm_main")
    comm.set_message_callback(msg_h); comm.set_connect_callback(con_h)
    if comm.connect():
        comm.start_listening()
        print("MQTTCommunicator: Prueba corriendo. Ctrl+C para salir.")
        try:
            for i in range(2): time.sleep(3); comm.publish("test/pub", {"count":i},1)
        except KeyboardInterrupt: pass
        finally: comm.disconnect()
    else: print("MQTTCommunicator: Fallo conexión en prueba.")