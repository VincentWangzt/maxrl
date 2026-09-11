"""Queue eight exact-population runs on two idle physical GPUs on the server.

GPU auto-selection is explicitly requested for this sweep. A slot keeps its
assigned GPU and checks it is free before every child. The queue can start with
one idle GPU and wait for a second; occupied devices are never shared.
"""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


def idle_gpus():
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
        text=True,
    )
    free = []
    for line in output.splitlines():
        gpu, memory, utilization = (int(item.strip()) for item in line.split(","))
        if memory > 128 or utilization != 0:
            continue
        processes = subprocess.check_output(
            ["nvidia-smi", f"--id={gpu}", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
            text=True,
        )
        if not processes.strip():
            free.append(gpu)
    return free


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--maxrl-tau", type=float, default=0.1)
    args = parser.parse_args()
    if (
        not args.name
        or any(
            character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
            for character in args.name
        )
        or args.name in (".", "..")
    ):
        raise ValueError("Sweep name must be a simple directory name")
    root = Path(__file__).resolve().parents[1]
    logs = root / "noisy-regression" / "logs" / args.name
    jobs = []
    for method, degree in (("grpo", None), ("rloo", None), ("maxrl", 256), ("maxrl", "inf")):
        for sigma in ("0p001", "0p1"):
            identity = f"d2_n64_10m_sep_eoo_range4_sigma{sigma}"
            data = root / "noisy-regression" / "data" / f"fixed_{identity}"
            metadata = json.loads((data / "metadata.json").read_text())["config"]
            expected = {
                "dimension": 2,
                "observations": 64,
                "sigma": float(sigma.replace("p", ".")),
                "train_count": 10_000_000,
                "eval_count": 1024,
            }
            if any(metadata[key] != value for key, value in expected.items()):
                raise ValueError(f"Unexpected dataset: {data}")
            label = method if degree is None else f"maxrl_deg{degree}_tau{args.maxrl_tau}"
            name = f"{args.name}_{label}_{identity}_bs1024_lr1e-4"
            output = root / "noisy-regression" / "checkpoints" / name
            if output.exists():
                raise FileExistsError(output)
            command = [
                "bash",
                str(root / "noisy-regression" / f"{method}.sh"),
                "--data-dir",
                str(data),
                "--run-name",
                name,
                "--output-dir",
                str(output),
                "--batch-size",
                "1024",
                "--micro-batch-size",
                "256",
                "--learning-rate",
                "1e-4",
                "--max-steps",
                "20000",
            ]
            if degree is not None:
                command += ["--maxrl-degree", str(degree), "--maxrl-tau", str(args.maxrl_tau)]
            jobs.append(
                {
                    "name": name,
                    "method": method,
                    "degree": degree,
                    "sigma": sigma,
                    "command": command,
                    "state": "queued",
                }
            )
    logs.mkdir(parents=True, exist_ok=False)
    (logs / "git_commit.txt").write_text(
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True)
    )
    environment = {**os.environ, "WANDB_RUN_GROUP": args.name}
    selected, active = [], {}

    def save():
        temporary = logs / "runs.json.tmp"
        temporary.write_text(json.dumps({"pid": os.getpid(), "gpus": selected, "jobs": jobs}, indent=2))
        temporary.replace(logs / "runs.json")

    save()
    while active or any(job["state"] == "queued" for job in jobs):
        for gpu, (process, stream, job) in list(active.items()):
            code = process.poll()
            if code is not None:
                stream.close()
                job.update(
                    state="complete" if code == 0 else "failed", exit_code=code, finished_at=time.time()
                )
                del active[gpu]
                print(f"GPU {gpu}: {job['name']} exited {code}", flush=True)
                save()
        free = idle_gpus()
        for gpu in free:
            if len(selected) < 2 and gpu not in selected:
                selected.append(gpu)
        for gpu in selected:
            if gpu in active or gpu not in free:
                continue
            job = next((item for item in jobs if item["state"] == "queued"), None)
            if job is None:
                break
            stream = (logs / f"{job['name']}.log").open("x")
            command = [*job["command"], "--gpu-id", str(gpu)]
            process = subprocess.Popen(
                command,
                cwd=root,
                env=environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
            )
            job.update(state="running", gpu=gpu, pid=process.pid, started_at=time.time(), command=command)
            active[gpu] = process, stream, job
            print(f"GPU {gpu}: started {job['name']} (PID {process.pid})", flush=True)
            save()
        time.sleep(15)
    save()
    if any(job["state"] == "failed" for job in jobs):
        raise SystemExit("One or more population runs failed; see runs.json and child logs")


if __name__ == "__main__":
    main()
