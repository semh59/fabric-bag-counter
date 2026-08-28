"""Test for the inference worker's process entry point honoring LINE_ID (§4.2, §4.5).

main() previously always constructed InferenceWorker with the default
line_id=1 regardless of environment -- meaning two inference containers
for two different real lines would have silently processed the same
line's frames. Verifies the real env var now reaches the worker.
"""

import services.inference.worker as inference_module


def test_main_honors_line_id_env_var(monkeypatch):
    monkeypatch.setenv("LINE_ID", "7")
    captured = {}

    class FakeTransport:
        pass

    def fake_start_loop(self):
        captured["line_id"] = self.line_id

    monkeypatch.setattr(inference_module, "SharedMemoryTransport", FakeTransport)
    monkeypatch.setattr(inference_module.InferenceWorker, "start_loop", fake_start_loop)

    inference_module.main()

    assert captured["line_id"] == 7


def test_main_defaults_line_id_to_1_when_unset(monkeypatch):
    monkeypatch.delenv("LINE_ID", raising=False)
    captured = {}

    class FakeTransport:
        pass

    def fake_start_loop(self):
        captured["line_id"] = self.line_id

    monkeypatch.setattr(inference_module, "SharedMemoryTransport", FakeTransport)
    monkeypatch.setattr(inference_module.InferenceWorker, "start_loop", fake_start_loop)

    inference_module.main()

    assert captured["line_id"] == 1
