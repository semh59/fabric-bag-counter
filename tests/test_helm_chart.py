"""Tests for Kubernetes Helm Chart Structure & Deployment Templates (§4.4, §11)."""

from __future__ import annotations

from pathlib import Path
import yaml


def test_helm_chart_metadata():
    chart_file = Path("deploy/helm/fabric-bag-counter/Chart.yaml")
    assert chart_file.exists(), "Chart.yaml does not exist"

    data = yaml.safe_load(chart_file.read_text(encoding="utf-8"))
    assert data["apiVersion"] == "v2"
    assert data["name"] == "fabric-bag-counter"
    assert "version" in data
    assert "appVersion" in data


def test_helm_values_structure():
    values_file = Path("deploy/helm/fabric-bag-counter/values.yaml")
    assert values_file.exists(), "values.yaml does not exist"

    values = yaml.safe_load(values_file.read_text(encoding="utf-8"))
    assert "global" in values
    assert "database" in values
    assert "api" in values
    assert "jobrunner" in values
    assert "erpRelay" in values
    assert "edgeSupervisor" in values

    # Check critical enterprise properties
    assert values["database"]["ssl"]["mode"] in ("require", "verify-ca", "verify-full")
    assert values["api"]["service"]["httpPort"] == 8080
    assert values["edgeSupervisor"]["hostIPC"] is True
    assert values["edgeSupervisor"]["modbus"]["enabled"] is True


def test_helm_templates_exist_and_valid():
    templates_dir = Path("deploy/helm/fabric-bag-counter/templates")
    assert templates_dir.exists()

    expected_templates = [
        "_helpers.tpl",
        "configmap.yaml",
        "deployment-api.yaml",
        "service-api.yaml",
        "ingress.yaml",
        "deployment-jobrunner.yaml",
        "deployment-erp-relay.yaml",
        "daemonset-edge-supervisor.yaml",
        "hpa.yaml",
    ]

    for t_name in expected_templates:
        p = templates_dir / t_name
        assert p.exists(), f"Missing expected Helm template: {t_name}"
        content = p.read_text(encoding="utf-8")
        assert len(content.strip()) > 0
        if t_name.endswith(".yaml"):
            assert "kind:" in content or "{{-" in content
