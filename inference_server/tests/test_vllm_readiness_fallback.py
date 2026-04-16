from inference_server.inference_server import cli as cli_module


class DummySSHClient:
    pass


def test_wait_for_vllm_readiness_uses_local_check_when_endpoint_reachable(monkeypatch):
    calls = {"local_wait": 0, "remote_wait": 0}

    class FakeHealthChecker:
        def __init__(self, client, timeout, interval):
            self.timeout = timeout
            self.interval = interval

        def wait_for_ready(self, callback=None):
            calls["local_wait"] += 1

    monkeypatch.setattr(cli_module, "_is_endpoint_reachable", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(cli_module, "HealthChecker", FakeHealthChecker)
    monkeypatch.setattr(
        cli_module,
        "wait_for_vllm_ready_remote",
        lambda **_kwargs: calls.__setitem__("remote_wait", calls["remote_wait"] + 1),
    )

    cli_module._wait_for_vllm_readiness(
        ssh_client=DummySSHClient(),
        deploy_path="~/inference_deploy",
        tailscale_ip="100.64.0.1",
        port=8000,
        vllm_api_key="test-key",
        timeout=300,
        interval=5,
        callback=lambda _msg: None,
    )

    assert calls["local_wait"] == 1
    assert calls["remote_wait"] == 0


def test_wait_for_vllm_readiness_falls_back_when_endpoint_unreachable(monkeypatch):
    calls = {"local_wait": 0, "remote_wait": 0, "remote_timeout": None}

    class FakeHealthChecker:
        def __init__(self, client, timeout, interval):
            pass

        def wait_for_ready(self, callback=None):
            calls["local_wait"] += 1

    monkeypatch.setattr(cli_module, "_is_endpoint_reachable", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(cli_module, "HealthChecker", FakeHealthChecker)
    monkeypatch.setattr(cli_module.time, "time", lambda: 1000.0)

    def fake_remote_wait(**kwargs):
        calls["remote_wait"] += 1
        calls["remote_timeout"] = kwargs["timeout"]

    monkeypatch.setattr(cli_module, "wait_for_vllm_ready_remote", fake_remote_wait)

    cli_module._wait_for_vllm_readiness(
        ssh_client=DummySSHClient(),
        deploy_path="~/inference_deploy",
        tailscale_ip="100.64.0.1",
        port=8000,
        vllm_api_key="test-key",
        timeout=300,
        interval=5,
        callback=lambda _msg: None,
    )

    assert calls["local_wait"] == 0
    assert calls["remote_wait"] == 1
    assert calls["remote_timeout"] == 300


def test_wait_for_vllm_readiness_falls_back_after_local_timeout_if_endpoint_drops(monkeypatch):
    calls = {"remote_wait": 0, "remote_timeout": None}

    class FakeHealthChecker:
        def __init__(self, client, timeout, interval):
            pass

        def wait_for_ready(self, callback=None):
            raise TimeoutError("local timeout")

    reachability = iter([True, False])
    monkeypatch.setattr(cli_module, "_is_endpoint_reachable", lambda *_args, **_kwargs: next(reachability))
    monkeypatch.setattr(cli_module, "HealthChecker", FakeHealthChecker)

    timestamps = iter([1000.0, 1300.0])  # 300s local wait elapsed
    monkeypatch.setattr(cli_module.time, "time", lambda: next(timestamps))

    def fake_remote_wait(**kwargs):
        calls["remote_wait"] += 1
        calls["remote_timeout"] = kwargs["timeout"]

    monkeypatch.setattr(cli_module, "wait_for_vllm_ready_remote", fake_remote_wait)

    cli_module._wait_for_vllm_readiness(
        ssh_client=DummySSHClient(),
        deploy_path="~/inference_deploy",
        tailscale_ip="100.64.0.1",
        port=8000,
        vllm_api_key="test-key",
        timeout=900,
        interval=10,
        callback=lambda _msg: None,
    )

    assert calls["remote_wait"] == 1
    assert calls["remote_timeout"] == 600


def test_wait_for_vllm_readiness_preserves_local_timeout_when_endpoint_reachable(monkeypatch):
    calls = {"remote_wait": 0}

    class FakeHealthChecker:
        def __init__(self, client, timeout, interval):
            pass

        def wait_for_ready(self, callback=None):
            raise TimeoutError("local timeout")

    monkeypatch.setattr(cli_module, "_is_endpoint_reachable", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(cli_module, "HealthChecker", FakeHealthChecker)
    monkeypatch.setattr(
        cli_module,
        "wait_for_vllm_ready_remote",
        lambda **_kwargs: calls.__setitem__("remote_wait", calls["remote_wait"] + 1),
    )

    try:
        cli_module._wait_for_vllm_readiness(
            ssh_client=DummySSHClient(),
            deploy_path="~/inference_deploy",
            tailscale_ip="100.64.0.1",
            port=8000,
            vllm_api_key="test-key",
            timeout=300,
            interval=5,
            callback=lambda _msg: None,
        )
        raised = False
    except TimeoutError:
        raised = True

    assert raised is True
    assert calls["remote_wait"] == 0
