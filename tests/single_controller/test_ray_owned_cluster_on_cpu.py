"""Check the process-owned Ray lifecycle used by the maze launch scripts."""

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

import psutil
import ray

DRIVER = """
import json
import sys
from pathlib import Path

import psutil
import ray

context = ray.init(
    num_cpus=1,
    num_gpus=0,
    object_store_memory=128 * 1024**2,
    include_dashboard=False,
    _temp_dir=sys.argv[1],
)
assert ray.get(ray.remote(lambda: 42).remote()) == 42
ready_file = Path(sys.argv[1], "ready.tmp")
ready_file.write_text(json.dumps({
    "address": context.address_info["gcs_address"],
    "pids": [process.pid for process in psutil.Process().children(recursive=True)],
}))
ready_file.replace(Path(sys.argv[1], "ready.json"))
if sys.stdin.readline().strip() == "error":
    raise RuntimeError("intentional training failure")
"""


def check_owned_cluster_exit(unrelated_cluster, exit_mode):
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "RAY_ADDRESS": "local", "RAY_USAGE_STATS_ENABLED": "0"}
    with tempfile.TemporaryDirectory(prefix="ray-owned-") as ray_dir:
        with subprocess.Popen(
            [sys.executable, "-c", DRIVER, ray_dir],
            env=environment,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        ) as driver:
            try:
                ready_file = Path(ray_dir, "ready.json")
                deadline = time.monotonic() + 60
                while not ready_file.exists():
                    if driver.poll() is not None:
                        raise AssertionError(f"Ray driver failed to start: {driver.communicate()[0]}")
                    if time.monotonic() >= deadline:
                        raise AssertionError("Ray driver did not become ready within 60 seconds")
                    time.sleep(0.1)

                details = json.loads(ready_file.read_text())
                assert details["address"] != unrelated_cluster
                owned_processes = [psutil.Process(pid) for pid in details["pids"] if psutil.pid_exists(pid)]
                assert owned_processes

                if exit_mode == "sigterm":
                    driver.send_signal(signal.SIGTERM)
                elif exit_mode == "sigint":
                    driver.send_signal(signal.SIGINT)
                output, _ = driver.communicate(input=f"{exit_mode}\n", timeout=30)
                if exit_mode == "success":
                    assert driver.returncode == 0, output
                else:
                    assert driver.returncode != 0, output
                if exit_mode == "error":
                    assert "intentional training failure" in output

                _, alive = psutil.wait_procs(owned_processes, timeout=10)
                assert not [process.pid for process in alive if process.status() != psutil.STATUS_ZOMBIE], output
                assert ray.is_initialized()
                assert ray.get(ray.remote(lambda: 42).remote(), timeout=15) == 42
            finally:
                if driver.poll() is None:
                    driver.terminate()
                    driver.wait(timeout=30)


class TestRayOwnedCluster(unittest.TestCase):
    def test_exit_preserves_unrelated_cluster(self):
        with tempfile.TemporaryDirectory(prefix="ray-other-") as ray_dir:
            context = ray.init(
                address="local",
                num_cpus=1,
                num_gpus=0,
                object_store_memory=128 * 1024**2,
                include_dashboard=False,
                _temp_dir=ray_dir,
            )
            try:
                for exit_mode in ["success", "error", "sigint", "sigterm"]:
                    with self.subTest(exit_mode=exit_mode):
                        check_owned_cluster_exit(context.address_info["gcs_address"], exit_mode)
            finally:
                ray.shutdown()
