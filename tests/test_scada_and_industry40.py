"""Unit and integration tests for OPC-UA SCADA Server & Industry 4.0 MQTT Gateway (§4.4)."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock
import pytest
from fastapi.testclient import TestClient

from drivers.io_mqtt.publisher import MqttIndustryPublisher
from drivers.io_opcua.server import OpcUaServerBridge
from drivers.scada_gateway import ScadaGatewayCoordinator
from services.api.main import app

client = TestClient(app)


def test_opcua_server_initialization_and_nodes():
    async def _run():
        server = OpcUaServerBridge(
            endpoint="opc.tcp://127.0.0.1:4841/fabric/test/",
            server_name="Test OPCUA Server",
            namespace_uri="http://test.fabric.internal/opcua/",
        )
        await server.start_async()
        assert server.is_running is True
        assert 1 in server._line_nodes

        # Update variable values
        await server.update_line_values(
            line_id=1,
            counted_total=42,
            target_count=500,
            belt_speed_mm_s=240.5,
            discrepancy_alarm=False,
            thermal_alarm=True,
            status="RUNNING",
        )

        # Read back variable value
        node = server._line_nodes[1]["CountedTotal"]
        val = await node.read_value()
        assert val == 42

        node_speed = server._line_nodes[1]["BeltSpeed_mm_s"]
        speed_val = await node_speed.read_value()
        assert abs(speed_val - 240.5) < 0.01

        # Stop server
        await server.stop_async()
        assert server.is_running is False

    asyncio.run(_run())


def test_mqtt_publisher_payloads():
    pub = MqttIndustryPublisher(
        broker_host="127.0.0.1",
        broker_port=1883,
        topic_prefix="fabric/plant1",
    )
    # Mock underlying paho client publish to verify payload formatting
    mock_res = MagicMock()
    mock_res.rc = 0
    pub.client.publish = MagicMock(return_value=mock_res)

    # 1. Count event
    ok1 = pub.publish_count_event(line_id=1, event_data={"counted_total": 105, "confidence": 0.98})
    assert ok1 is True
    pub.client.publish.assert_called()
    args, kwargs = pub.client.publish.call_args
    assert "fabric/plant1/line/1/count_event" in args[0]
    payload_str = kwargs.get("payload") or (args[1] if len(args) > 1 else "")
    assert "105" in payload_str


    # 2. Telemetry
    ok2 = pub.publish_telemetry(line_id=1, payload={"belt_speed": 220.0, "status": "ACTIVE"})
    assert ok2 is True

    # 3. Alarm
    ok3 = pub.publish_alarm(line_id=1, alarm_data={"alarm_type": "hot_leak", "severity": "critical"})
    assert ok3 is True

    # 4. Thermal profile
    ok4 = pub.publish_thermal_profile(line_id=1, profile_data={"mean_temp_c": 64.2})
    assert ok4 is True


def test_scada_gateway_coordinator():
    gateway = ScadaGatewayCoordinator(
        enable_modbus=False,
        enable_opcua=False,
        enable_mqtt=True,
    )
    mock_res = MagicMock()
    mock_res.rc = 0
    assert gateway.mqtt is not None
    gateway.mqtt.client.publish = MagicMock(return_value=mock_res)

    gateway.dispatch_count_event(line_id=1, session_id=10, counted_total=50, target_count=100, belt_speed=200.0)
    gateway.mqtt.client.publish.assert_called()

    gateway.dispatch_alarm(line_id=1, alarm_type="discrepancy", description="Area mismatch > 8%", severity="warning")

    status = gateway.get_status()
    assert "modbus" in status
    assert "opcua" in status
    assert "mqtt" in status


def test_api_scada_status_endpoint():
    res = client.get("/api/system/scada-status")
    assert res.status_code == 200
    data = res.json()
    assert "modbus" in data
    assert "opcua" in data
    assert "mqtt" in data
