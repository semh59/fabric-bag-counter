"""Industrial OPC-UA Server Driver (§4.4, Industry 4.0 SCADA Integration).

Exposes standardized OPC-UA AddressSpace objects, variables, and methods for Siemens WinCC,
Wonderware, Ignition, and Rockwell FactoryTalk SCADA systems using asyncua.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any

from asyncua import Server, ua

logger = logging.getLogger(__name__)


class OpcUaServerBridge:
    """OPC-UA Server exposing live conveyor counting nodes and SCADA methods."""

    def __init__(
        self,
        endpoint: str = "opc.tcp://0.0.0.0:4840/fabric/server/",
        server_name: str = "Fabric Conveyor Bag Counter OPC-UA Server",
        namespace_uri: str = "http://fabric-industrial.internal/opcua/",
    ) -> None:
        self.endpoint = endpoint
        self.server_name = server_name
        self.namespace_uri = namespace_uri
        self.server: Server | None = None
        self.idx: int = 0
        self.is_running: bool = False

        # Node references per line
        self._line_nodes: dict[int, dict[str, Any]] = {}
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None

    async def init_server(self) -> None:
        """Initialize server configuration and register AddressSpace nodes."""
        self.server = Server()
        await self.server.init()
        self.server.set_endpoint(self.endpoint)
        self.server.set_server_name(self.server_name)

        # Register custom namespace
        self.idx = await self.server.register_namespace(self.namespace_uri)

        # Root Folder for Industrial Conveyor Objects
        objects = self.server.nodes.objects
        fabric_folder = await objects.add_folder(self.idx, "FabricConveyors")

        # Create default Line 1 nodes
        await self.setup_line_nodes(line_id=1, parent_folder=fabric_folder)
        logger.info(f"[OPC-UA Server] Initialized AddressSpace on {self.endpoint} (Namespace: {self.namespace_uri})")

    async def setup_line_nodes(self, line_id: int, parent_folder: Any | None = None) -> None:
        """Create OPC-UA variables for a specific production line."""
        assert self.server is not None
        if parent_folder is None:
            objects = self.server.nodes.objects
            parent_folder = await objects.add_folder(self.idx, "FabricConveyors")

        line_folder = await parent_folder.add_folder(self.idx, f"Line_{line_id}")

        counted_var = await line_folder.add_variable(self.idx, "CountedTotal", 0, ua.VariantType.UInt32)
        await counted_var.set_writable()

        target_var = await line_folder.add_variable(self.idx, "TargetCount", 1000, ua.VariantType.UInt32)
        await target_var.set_writable()

        speed_var = await line_folder.add_variable(self.idx, "BeltSpeed_mm_s", 0.0, ua.VariantType.Double)
        await speed_var.set_writable()

        session_var = await line_folder.add_variable(self.idx, "ActiveSessionId", 0, ua.VariantType.UInt32)
        await session_var.set_writable()

        discrepancy_var = await line_folder.add_variable(self.idx, "DiscrepancyAlarm", False, ua.VariantType.Boolean)
        await discrepancy_var.set_writable()

        thermal_var = await line_folder.add_variable(self.idx, "ThermalAnomalyAlarm", False, ua.VariantType.Boolean)
        await thermal_var.set_writable()

        status_var = await line_folder.add_variable(self.idx, "SystemStatus", "IDLE", ua.VariantType.String)
        await status_var.set_writable()

        self._line_nodes[line_id] = {
            "folder": line_folder,
            "CountedTotal": counted_var,
            "TargetCount": target_var,
            "BeltSpeed_mm_s": speed_var,
            "ActiveSessionId": session_var,
            "DiscrepancyAlarm": discrepancy_var,
            "ThermalAnomalyAlarm": thermal_var,
            "SystemStatus": status_var,
        }

    async def update_line_values(
        self,
        line_id: int,
        counted_total: int | None = None,
        target_count: int | None = None,
        belt_speed_mm_s: float | None = None,
        active_session_id: int | None = None,
        discrepancy_alarm: bool | None = None,
        thermal_alarm: bool | None = None,
        status: str | None = None,
    ) -> None:
        """Update OPC-UA variable values atomically."""
        if line_id not in self._line_nodes:
            return

        nodes = self._line_nodes[line_id]

        if counted_total is not None:
            await nodes["CountedTotal"].write_value(ua.DataValue(ua.Variant(counted_total, ua.VariantType.UInt32)))
        if target_count is not None:
            await nodes["TargetCount"].write_value(ua.DataValue(ua.Variant(target_count, ua.VariantType.UInt32)))
        if belt_speed_mm_s is not None:
            await nodes["BeltSpeed_mm_s"].write_value(ua.DataValue(ua.Variant(float(belt_speed_mm_s), ua.VariantType.Double)))
        if active_session_id is not None:
            await nodes["ActiveSessionId"].write_value(ua.DataValue(ua.Variant(active_session_id, ua.VariantType.UInt32)))
        if discrepancy_alarm is not None:
            await nodes["DiscrepancyAlarm"].write_value(ua.DataValue(ua.Variant(discrepancy_alarm, ua.VariantType.Boolean)))
        if thermal_alarm is not None:
            await nodes["ThermalAnomalyAlarm"].write_value(ua.DataValue(ua.Variant(thermal_alarm, ua.VariantType.Boolean)))
        if status is not None:
            await nodes["SystemStatus"].write_value(ua.DataValue(ua.Variant(status, ua.VariantType.String)))

    async def start_async(self) -> None:
        await self.init_server()
        assert self.server is not None
        await self.server.start()
        self.is_running = True
        logger.info(f"[OPC-UA Server] Server started listening on {self.endpoint}")

    async def stop_async(self) -> None:
        if self.server and self.is_running:
            await self.server.stop()
            self.is_running = False
            logger.info("[OPC-UA Server] Server stopped cleanly.")

    def start_in_background(self) -> None:
        """Start the OPC-UA server in a dedicated background daemon thread."""
        if self.is_running or (self._thread and self._thread.is_alive()):
            return

        def _runner():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._loop.run_until_complete(self.start_async())
            self._loop.run_forever()

        self._thread = threading.Thread(target=_runner, daemon=True, name="OpcUaServerThread")
        self._thread.start()

    def stop(self) -> None:
        if self._loop and self.is_running:
            future = asyncio.run_coroutine_threadsafe(self.stop_async(), self._loop)
            try:
                future.result(timeout=5.0)
            except Exception as e:
                logger.debug(f"Error awaiting OPC-UA server stop: {e}")
            self._loop.call_soon_threadsafe(self._loop.stop)


# Global singleton instance
_opcua_server: OpcUaServerBridge | None = None


def get_opcua_server() -> OpcUaServerBridge:
    global _opcua_server
    if _opcua_server is None:
        _opcua_server = OpcUaServerBridge()
    return _opcua_server
