from inference_server.inference_server import bootstrap as bootstrap_module


class FakeSSHClient:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    def run(self, command, timeout=30):
        if self.calls >= len(self._responses):
            raise AssertionError("No more fake responses configured")
        response = self._responses[self.calls]
        self.calls += 1
        return response


def test_wait_for_vllm_ready_remote_succeeds_on_first_ready(monkeypatch):
    ssh_client = FakeSSHClient([(0, "READY\n", "")])
    monkeypatch.setattr(bootstrap_module.time, "sleep", lambda _s: None)
    monkeypatch.setattr(bootstrap_module.time, "time", lambda: 1000.0)

    result = bootstrap_module.wait_for_vllm_ready_remote(
        ssh_client=ssh_client,
        deploy_path="~/inference_deploy",
        timeout=120,
        interval=5,
        callback=lambda _msg: None,
    )

    assert result is True
    assert ssh_client.calls == 1


def test_wait_for_vllm_ready_remote_retries_health_fail_then_succeeds(monkeypatch):
    ssh_client = FakeSSHClient(
        [
            (1, "HEALTH_FAIL:timed out", ""),
            (0, "READY\n", ""),
        ]
    )
    callback_messages = []

    monkeypatch.setattr(bootstrap_module.time, "sleep", lambda _s: None)
    now = {"value": 0.0}

    def fake_time():
        now["value"] += 1.0
        return now["value"]

    monkeypatch.setattr(bootstrap_module.time, "time", fake_time)

    result = bootstrap_module.wait_for_vllm_ready_remote(
        ssh_client=ssh_client,
        deploy_path="~/inference_deploy",
        timeout=120,
        interval=5,
        callback=callback_messages.append,
    )

    assert result is True
    assert ssh_client.calls == 2
    assert any("health endpoint (remote)" in msg for msg in callback_messages)
