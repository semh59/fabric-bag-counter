"""Unit tests for SAP ECC 6.0 ERP adapter."""

from datetime import datetime, timezone
from pathlib import Path
import pytest
from drivers.erp_sap_ecc.adapter import SapEccErpAdapter
from packages.cs_core.interfaces.erp_adapter import ErpStatusState, SessionPayload


def test_sap_ecc_adapter_formats_bapi_payload(tmp_path):
    adapter = SapEccErpAdapter(file_export_dir=str(tmp_path), plant="1000", storage_location="0001", movement_type="601")

    session = SessionPayload(
        session_id=42,
        line_id=1,
        product_profile_id=10,
        counted_total=500,
        area_estimate_total=498.5,
        erp_material_code="CEM_I_42_5_R",
        opened_at=datetime(2026, 8, 31, 8, 0, 0, tzinfo=timezone.utc),
        closed_at=datetime(2026, 8, 31, 8, 30, 0, tzinfo=timezone.utc),
        metadata={"cryptographic_seal": "test-hmac-seal-42"},
    )

    bapi = adapter.format_bapi_payload(session)
    assert bapi["BAPI_HEADER"]["REF_DOC_NO"] == "SESS-42"
    assert bapi["GOODSMVT_ITEM"][0]["MATERIAL"] == "CEM_I_42_5_R"
    assert bapi["GOODSMVT_ITEM"][0]["ENTRY_QNT"] == 500
    assert bapi["GOODSMVT_ITEM"][0]["MOVE_TYPE"] == "601"
    assert bapi["CRYPTO_PROOF"]["HMAC_SEAL"] == "test-hmac-seal-42"


def test_sap_ecc_adapter_dispatches_file_export(tmp_path):
    adapter = SapEccErpAdapter(file_export_dir=str(tmp_path))

    session = SessionPayload(
        session_id=99,
        line_id=2,
        product_profile_id=10,
        counted_total=120,
        opened_at=datetime.now(timezone.utc),
        closed_at=datetime.now(timezone.utc),
    )

    result = adapter.submit_session(session)
    assert result.success is True
    exported_files = list(tmp_path.glob("SAP_ECC_SESS_99_*.json"))
    assert len(exported_files) == 1
