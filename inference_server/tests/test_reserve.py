from types import SimpleNamespace

from click.testing import CliRunner

from inference_server.inference_server import cli as cli_module
from inference_server.inference_server.lambda_api import LambdaAPIError, LambdaClient as RealLambdaClient
from inference_server.inference_server.state import InstanceState


class FakeConfig:
    def __init__(self):
        self.models = {"default": "llama31-8b"}
        self.instance = {"gpu": "gpu_1x_a100"}
        self.timeouts = {"ssh_ready": 60}
        self.vllm = {"image_tag": "vllm/vllm-openai:v0.6.4"}

    def get_model_config(self, model_alias=None):
        return {
            "id": "meta-llama/Llama-3.1-8B-Instruct",
            "revision": None,
            "max_model_len": 16384,
        }

    def get_instance_name(self, model_alias=None, custom_name=None):
        if custom_name:
            return custom_name
        return f"research-{model_alias or self.models['default']}"

    def get_gpu_types_to_try(self, primary_gpu=None):
        primary = primary_gpu or self.instance["gpu"]
        return [primary, "gpu_1x_h100_pcie", "gpu_1x_h100_sxm5"]


class FakeStateManager:
    def __init__(self, instances=None):
        self.instances = {}
        for instance in instances or []:
            self.instances[instance.name] = instance

    def get_instance(self, name):
        return self.instances.get(name)

    def set_instance(self, instance):
        self.instances[instance.name] = instance

    def delete_instance(self, name):
        return self.instances.pop(name, None) is not None

    def update_instance(self, name, **kwargs):
        instance = self.instances.get(name)
        if not instance:
            return None
        instance.update(**kwargs)
        return instance


class FakeClock:
    def __init__(self, start=0.0):
        self.now = start

    def monotonic(self):
        return self.now

    def time(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def invoke_reserve(monkeypatch, lambda_client, state_mgr=None, extra_args=None, clock=None):
    state_mgr = state_mgr or FakeStateManager()
    config = FakeConfig()

    class PatchedLambdaClient:
        @staticmethod
        def is_retryable_error(error):
            return RealLambdaClient.is_retryable_error(error)

        def __new__(cls):  # noqa: D401
            return lambda_client

    monkeypatch.setattr(cli_module, "get_config", lambda: config)
    monkeypatch.setattr(cli_module, "get_state_manager", lambda: state_mgr)
    monkeypatch.setattr(cli_module, "LambdaClient", PatchedLambdaClient)

    if clock is not None:
        monkeypatch.setattr(cli_module.time, "monotonic", clock.monotonic)
        monkeypatch.setattr(cli_module.time, "time", clock.time)
        monkeypatch.setattr(cli_module.time, "sleep", clock.sleep)
    else:
        monkeypatch.setattr(cli_module.time, "sleep", lambda _: None)

    runner = CliRunner()
    args = [
        "reserve",
        "--filesystem",
        "my-fs",
        "--ssh-key",
        "my-key",
        "--status-interval",
        "9999",
    ]
    if extra_args:
        args.extend(extra_args)
    result = runner.invoke(cli_module.cli, args)
    return result, state_mgr


class BaseLambdaClientStub:
    def __init__(self):
        self.launch_calls = 0
        self.availability_calls = 0
        self.launch_requests = []

    def get_instance(self, instance_id):
        return None

    def validate_filesystem(self, filesystem_name):
        return filesystem_name == "my-fs"

    def get_filesystem_names(self):
        return ["my-fs"]

    def get_filesystem_region(self, filesystem_name):
        return "us-west-1"

    def get_ssh_key_names(self):
        return ["my-key"]

    def get_available_regions(self, gpu_type):
        self.availability_calls += 1
        return ["us-west-1"]

    def launch_instance(self, instance_type, region, ssh_key_names, filesystem_names, name):
        self.launch_calls += 1
        self.launch_requests.append(
            {
                "instance_type": instance_type,
                "region": region,
                "ssh_key_names": ssh_key_names,
                "filesystem_names": filesystem_names,
                "name": name,
            }
        )
        return "inst-1"

    def wait_for_instance(self, instance_id, target_status, timeout, poll_interval, callback):
        return SimpleNamespace(status="active", ip="198.51.100.7", region="us-west-1")


def test_reserve_exits_success_when_existing_instance_is_already_active(monkeypatch):
    existing = InstanceState(
        instance_id="inst-existing",
        name="research-llama31-8b",
        status="active",
        gpu_type="gpu_1x_a100",
        filesystem="my-fs",
        ssh_key="my-key",
        region="us-west-1",
    )
    state_mgr = FakeStateManager(instances=[existing])

    class Client(BaseLambdaClientStub):
        def get_instance(self, instance_id):
            return SimpleNamespace(
                id=instance_id,
                status="active",
                ip="203.0.113.9",
                region="us-west-1",
                instance_type="gpu_1x_a100",
            )

        def launch_instance(self, *args, **kwargs):  # pragma: no cover - should never be called
            raise AssertionError("launch_instance should not be called when existing instance is active")

    result, _ = invoke_reserve(monkeypatch, Client(), state_mgr=state_mgr)
    assert result.exit_code == 0
    assert "already active" in result.output


def test_reserve_polls_until_capacity_then_launches_once(monkeypatch):
    class Client(BaseLambdaClientStub):
        def __init__(self):
            super().__init__()
            self.responses = [[], [], ["us-west-1"]]

        def get_available_regions(self, gpu_type):
            self.availability_calls += 1
            idx = min(self.availability_calls - 1, len(self.responses) - 1)
            return self.responses[idx]

    client = Client()
    result, _ = invoke_reserve(monkeypatch, client)
    assert result.exit_code == 0
    assert client.availability_calls >= 3
    assert client.launch_calls == 1
    assert "Reservation complete!" in result.output


def test_reserve_retries_on_insufficient_capacity_then_succeeds(monkeypatch):
    class Client(BaseLambdaClientStub):
        def __init__(self):
            super().__init__()
            self.launch_attempts = 0

        def launch_instance(self, *args, **kwargs):
            self.launch_calls += 1
            self.launch_attempts += 1
            if self.launch_attempts == 1:
                raise LambdaAPIError(
                    "API error: Not enough capacity to fulfill launch request.",
                    status_code=400,
                    error_code="instance-operations/launch/insufficient-capacity",
                )
            return "inst-2"

    client = Client()
    result, _ = invoke_reserve(monkeypatch, client)
    assert result.exit_code == 0
    assert client.launch_calls == 2
    assert "retryable error" in result.output


def test_reserve_exits_on_non_retryable_launch_error(monkeypatch):
    class Client(BaseLambdaClientStub):
        def launch_instance(self, *args, **kwargs):
            self.launch_calls += 1
            raise LambdaAPIError(
                "API error: Quota exceeded.",
                status_code=400,
                error_code="global/quota-exceeded",
            )

    client = Client()
    result, _ = invoke_reserve(monkeypatch, client)
    assert result.exit_code == 1
    assert client.launch_calls == 1
    assert "non-retryable error" in result.output


def test_reserve_enforces_minimum_launch_interval(monkeypatch):
    clock = FakeClock()
    launch_times = []

    class Client(BaseLambdaClientStub):
        def __init__(self):
            super().__init__()
            self.launch_attempts = 0

        def launch_instance(self, *args, **kwargs):
            self.launch_calls += 1
            self.launch_attempts += 1
            launch_times.append(clock.monotonic())
            if self.launch_attempts == 1:
                raise LambdaAPIError(
                    "API error: Not enough capacity to fulfill launch request.",
                    status_code=400,
                    error_code="instance-operations/launch/insufficient-capacity",
                )
            return "inst-3"

    client = Client()
    result, _ = invoke_reserve(monkeypatch, client, clock=clock, extra_args=["--poll-interval", "1"])
    assert result.exit_code == 0
    assert len(launch_times) == 2
    assert launch_times[1] - launch_times[0] >= 12.0


def test_reserve_persists_instance_state_on_success(monkeypatch):
    client = BaseLambdaClientStub()
    result, state_mgr = invoke_reserve(monkeypatch, client)
    assert result.exit_code == 0

    saved = state_mgr.get_instance("research-llama31-8b")
    assert saved is not None
    assert saved.instance_id == "inst-1"
    assert saved.status == "active"
    assert saved.gpu_type == "gpu_1x_a100"
    assert saved.region == "us-west-1"
    assert saved.public_ip == "198.51.100.7"
    assert saved.model_alias == "llama31-8b"


def test_reserve_accepts_comma_separated_gpu_priority(monkeypatch):
    class Client(BaseLambdaClientStub):
        def __init__(self):
            super().__init__()
            self.checked_gpu_types = []

        def get_available_regions(self, gpu_type):
            self.availability_calls += 1
            self.checked_gpu_types.append(gpu_type)
            if gpu_type == "gpu_1x_h100_pcie":
                return []
            if gpu_type == "gpu_1x_a100":
                return ["us-west-1"]
            return []

    client = Client()
    result, state_mgr = invoke_reserve(
        monkeypatch,
        client,
        extra_args=["--gpu", "gpu_1x_h100_pcie,gpu_1x_a100"],
    )

    assert result.exit_code == 0
    assert client.checked_gpu_types == ["gpu_1x_h100_pcie", "gpu_1x_a100"]
    saved = state_mgr.get_instance("research-llama31-8b")
    assert saved.gpu_type == "gpu_1x_a100"


def test_reserve_preserves_filesystem_attachment_with_custom_gpu_priority(monkeypatch):
    client = BaseLambdaClientStub()
    result, _ = invoke_reserve(
        monkeypatch,
        client,
        extra_args=["--gpu", "gpu_1x_h100_pcie", "--gpu", "gpu_1x_a100"],
    )

    assert result.exit_code == 0
    assert client.launch_requests
    assert client.launch_requests[0]["filesystem_names"] == ["my-fs"]
