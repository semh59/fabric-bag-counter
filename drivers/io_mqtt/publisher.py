"""Industry 4.0 MQTT Telemetry & Alarm Publisher Driver (§4.4).

Publishes real-time conveyor count events, hardware telemetry, and anomaly alarms
to enterprise MQTT brokers (Mosquitto, EMQX, HiveMQ, AWS IoT Core) using paho-mqtt v2.
"""

from __future__ import annotations

import json
import logging
import ssl
import time
from typing import Any
import paho.mqtt.client as mqtt

logger = logging.getLogger(__name__)


class MqttIndustryPublisher:
    """Enterprise MQTT Publisher for Factory Conveyor Industry 4.0 Integration."""

    def __init__(
        self,
        broker_host: str = "localhost",
        broker_port: int = 1883,
        client_id: str = "fabric-conveyor-gateway",
        topic_prefix: str = "fabric/plant1",
        username: str | None = None,
        password: str | None = None,
        use_tls: bool = False,
        ca_cert_path: str | None = None,
        keepalive: int = 60,
    ) -> None:
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.client_id = client_id
        self.topic_prefix = topic_prefix.rstrip("/")
        self.username = username
        self.password = password
        self.use_tls = use_tls
        self.ca_cert_path = ca_cert_path
        self.keepalive = keepalive
        self.is_connected = False

        # Initialize paho client (v2 CallbackAPIVersion)
        self.client = mqtt.Client(
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
            client_id=self.client_id,
        )

        if self.username and self.password:
            self.client.username_pw_set(self.username, self.password)

        if self.use_tls:
            if self.ca_cert_path:
                self.client.tls_set(ca_certs=self.ca_cert_path, cert_reqs=ssl.CERT_REQUIRED)
            else:
                self.client.tls_set()

        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect

    def _on_connect(self, client: mqtt.Client, userdata: Any, flags: Any, rc: Any, properties: Any = None) -> None:
        if rc == 0:
            self.is_connected = True
            logger.info(f"[MQTT] Successfully connected to broker {self.broker_host}:{self.broker_port}")
            # Publish birth message
            self.publish_telemetry(
                line_id=1,
                payload={"status": "ONLINE", "gateway": self.client_id, "timestamp": time.time()},
                retain=True,
            )
        else:
            self.is_connected = False
            logger.warning(f"[MQTT] Connection failed with result code {rc}")

    def _on_disconnect(self, client: mqtt.Client, userdata: Any, flags: Any, rc: Any, properties: Any = None) -> None:
        self.is_connected = False
        logger.warning(f"[MQTT] Disconnected from broker (rc={rc})")

    def connect(self) -> bool:
        """Connect to broker and start background network loop thread."""
        try:
            self.client.connect(self.broker_host, self.broker_port, self.keepalive)
            self.client.loop_start()
            return True
        except Exception as e:
            logger.error(f"[MQTT] Failed to initiate connection to {self.broker_host}:{self.broker_port}: {e}")
            return False

    def disconnect(self) -> None:
        try:
            self.client.loop_stop()
            self.client.disconnect()
            self.is_connected = False
        except Exception as e:
            logger.debug(f"[MQTT] Disconnect error: {e}")

    def publish_count_event(self, line_id: int, event_data: dict[str, Any], qos: int = 1) -> bool:
        """Publish real-time bag count crossing event."""
        topic = f"{self.topic_prefix}/line/{line_id}/count_event"
        return self._publish_json(topic, event_data, qos=qos)

    def publish_telemetry(self, line_id: int, payload: dict[str, Any], qos: int = 0, retain: bool = False) -> bool:
        """Publish conveyor speed, FPS, and status telemetry."""
        topic = f"{self.topic_prefix}/line/{line_id}/telemetry"
        return self._publish_json(topic, payload, qos=qos, retain=retain)

    def publish_alarm(self, line_id: int, alarm_data: dict[str, Any], qos: int = 2) -> bool:
        """Publish critical anomaly or discrepancy alarm."""
        topic = f"{self.topic_prefix}/line/{line_id}/alarms"
        return self._publish_json(topic, alarm_data, qos=qos, retain=True)

    def publish_thermal_profile(self, line_id: int, profile_data: dict[str, Any], qos: int = 1) -> bool:
        """Publish radiometric surface thermal profile."""
        topic = f"{self.topic_prefix}/line/{line_id}/thermal"
        return self._publish_json(topic, profile_data, qos=qos)

    def _publish_json(self, topic: str, data: dict[str, Any], qos: int = 1, retain: bool = False) -> bool:
        payload_str = json.dumps(data, default=str)
        info = self.client.publish(topic, payload=payload_str, qos=qos, retain=retain)
        return info.rc == mqtt.MQTT_ERR_SUCCESS


# Global singleton
_mqtt_publisher: MqttIndustryPublisher | None = None


def get_mqtt_publisher() -> MqttIndustryPublisher:
    global _mqtt_publisher
    if _mqtt_publisher is None:
        _mqtt_publisher = MqttIndustryPublisher()
    return _mqtt_publisher
