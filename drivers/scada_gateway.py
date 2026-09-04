"""Unified Industry 4.0 & SCADA Gateway Coordinator (§4.4).

Synchronously orchestrates industrial factory communication across:
1. Modbus TCP (Real-time PLC coil/register control)
2. OPC-UA (SCADA/DCS AddressSpace for Siemens/Ignition/Wonderware)
3. MQTT (Cloud/Enterprise IoT pub/sub broker telemetry)
"""

from __future__ import annotations

import logging
import time
from typing import Any

from drivers.io_modbus_tcp.controller import ModbusTcpIoController
from drivers.io_mqtt.publisher import MqttIndustryPublisher, get_mqtt_publisher
from drivers.io_opcua.server import OpcUaServerBridge, get_opcua_server

logger = logging.getLogger(__name__)


class ScadaGatewayCoordinator:
    """Central gateway broadcasting events and states across Modbus TCP, OPC-UA, and MQTT."""

    def __init__(
        self,
        enable_modbus: bool = True,
        enable_opcua: bool = True,
        enable_mqtt: bool = True,
        modbus_host: str = "127.0.0.1",
        modbus_port: int = 502,
        mqtt_broker: str = "localhost",
        mqtt_port: int = 1883,
        opcua_endpoint: str = "opc.tcp://0.0.0.0:4840/fabric/server/",
    ) -> None:
        self.enable_modbus = enable_modbus
        self.enable_opcua = enable_opcua
        self.enable_mqtt = enable_mqtt

        # Modbus controller
        self.modbus: ModbusTcpIoController | None = None
        if self.enable_modbus:
            self.modbus = ModbusTcpIoController(host=modbus_host, port=modbus_port)


        # OPC-UA server bridge
        self.opcua: OpcUaServerBridge | None = None
        if self.enable_opcua:
            self.opcua = get_opcua_server()
            self.opcua.endpoint = opcua_endpoint

        # MQTT publisher
        self.mqtt: MqttIndustryPublisher | None = None
        if self.enable_mqtt:
            self.mqtt = get_mqtt_publisher()
            self.mqtt.broker_host = mqtt_broker
            self.mqtt.broker_port = mqtt_port

    def start_all(self) -> None:
        """Start background services and connections."""
        if self.modbus:
            self.modbus.connect()

        if self.opcua:
            self.opcua.start_in_background()

        if self.mqtt:
            self.mqtt.connect()

        logger.info("[SCADA Gateway] All industrial communication bridges initialized.")

    def stop_all(self) -> None:
        """Gracefully disconnect all industrial bridges."""
        if self.modbus:
            self.modbus.disconnect()
        if self.opcua:
            self.opcua.stop()
        if self.mqtt:
            self.mqtt.disconnect()

    def dispatch_count_event(
        self,
        line_id: int,
        session_id: int,
        counted_total: int,
        target_count: int | None = None,
        belt_speed: float = 0.0,
    ) -> None:
        """Broadcast bag count increment to PLC, SCADA, and IoT Broker."""
        payload = {
            "session_id": session_id,
            "line_id": line_id,
            "counted_total": counted_total,
            "target_count": target_count,
            "belt_speed": belt_speed,
            "timestamp": time.time(),
        }

        # 1. Modbus PLC write
        if self.modbus and self.modbus.is_connected:
            self.modbus.write_holding_register(100, counted_total)
            if target_count:
                self.modbus.write_holding_register(101, target_count)
            self.modbus.write_float32(102, belt_speed)

        # 2. OPC-UA variable update
        if self.opcua and self.opcua.is_running and self.opcua._loop:
            import asyncio
            asyncio.run_coroutine_threadsafe(
                self.opcua.update_line_values(
                    line_id=line_id,
                    counted_total=counted_total,
                    target_count=target_count,
                    belt_speed_mm_s=belt_speed,
                    active_session_id=session_id,
                ),
                self.opcua._loop,
            )

        # 3. MQTT message publish
        if self.mqtt:
            self.mqtt.publish_count_event(line_id=line_id, event_data=payload)

    def dispatch_alarm(
        self,
        line_id: int,
        alarm_type: str,
        description: str,
        is_active: bool = True,
        severity: str = "warning",
    ) -> None:
        """Signal alarm across PLC light tower / siren, OPC-UA alarms, and MQTT broker."""
        # 1. Modbus hardware coil trigger
        if self.modbus and self.modbus.is_connected:
            # Coil 1: warning horn, Coil 3: red error light
            if severity == "critical":
                self.modbus.write_coil(3, is_active)  # Error red
                self.modbus.write_coil(1, is_active)  # Warning horn
            else:
                self.modbus.write_coil(1, is_active)

        # 2. OPC-UA alarm flag
        if self.opcua and self.opcua.is_running and self.opcua._loop:
            import asyncio
            discrepancy = alarm_type == "discrepancy" and is_active
            thermal = alarm_type == "thermal" and is_active
            asyncio.run_coroutine_threadsafe(
                self.opcua.update_line_values(
                    line_id=line_id,
                    discrepancy_alarm=discrepancy,
                    thermal_alarm=thermal,
                    status=f"ALARM_{alarm_type.upper()}" if is_active else "RUNNING",
                ),
                self.opcua._loop,
            )

        # 3. MQTT alarm publish
        if self.mqtt:
            self.mqtt.publish_alarm(
                line_id=line_id,
                alarm_data={
                    "alarm_type": alarm_type,
                    "severity": severity,
                    "description": description,
                    "is_active": is_active,
                    "timestamp": time.time(),
                },
            )

    def get_status(self) -> dict[str, Any]:
        """Inspect status of all industrial bridges."""
        return {
            "modbus": {
                "enabled": self.enable_modbus,
                "connected": self.modbus.is_connected if self.modbus else False,
            },
            "opcua": {
                "enabled": self.enable_opcua,
                "running": self.opcua.is_running if self.opcua else False,
                "endpoint": self.opcua.endpoint if self.opcua else None,
            },
            "mqtt": {
                "enabled": self.enable_mqtt,
                "connected": self.mqtt.is_connected if self.mqtt else False,
                "broker": f"{self.mqtt.broker_host}:{self.mqtt.broker_port}" if self.mqtt else None,
            },
        }


# Global instance
_scada_gateway: ScadaGatewayCoordinator | None = None


def get_scada_gateway() -> ScadaGatewayCoordinator:
    global _scada_gateway
    if _scada_gateway is None:
        _scada_gateway = ScadaGatewayCoordinator()
    return _scada_gateway
