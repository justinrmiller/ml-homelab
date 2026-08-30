# Multi-Machine Ray Cluster

Run Ray across several machines on your LAN, with this repo checked out on each
one.

## Why this is a separate topology

`make start` runs Ray inside a Kind cluster on a single machine. That path
cannot accept workers from other machines: Kind publishes its ports on
loopback, and Ray needs a wide, bidirectional port range open between every
node. So the multi-machine setup skips Kubernetes entirely — a native Ray head
on one machine, native Ray workers on the others.

The two topologies are independent. Use one or the other, not both at once:
they would fight over ports 6379 and 8265.

| | `make start` (Kind) | `make head-start` (native) |
|---|---|---|
| Machines | one | many |
| Kubernetes | Kind + KubeRay | none |
| Monitoring | Prometheus + Grafana wired up | Ray dashboard only |
| Floci S3 | yes, via Compose | run it separately if you want it |

## How version parity is guaranteed

Ray refuses to join a cluster whose Ray version differs from the head's, and
the Python versions have to match too. Rather than each machine getting that
right by hand, every node syncs the *same pinned environment* from this repo:

- `.python-version` pins the interpreter to 3.12.
- `uv.lock` pins `ray` to one exact build.
- `scripts/ray-node-setup.sh` runs `uv sync --frozen`, so uv installs precisely
  what the lockfile says instead of re-resolving.

**Two machines on the same commit of this repo therefore agree by
construction.** `scripts/ray-node-setup.sh` prints the Ray version, Ray commit,
and repo commit so you can confirm; `scripts/ray-worker-start.sh` re-checks
against the head's dashboard before trying to join, and names the fix if they
differ.

PyTorch is the one thing that intentionally differs per machine:
`scripts/torch-extra.sh` picks the CUDA build on a Linux box with an NVIDIA GPU
and the CPU/MPS build everywhere else. That is a different wheel, not a
different version, so it does not affect cluster compatibility.

Mixed architectures are fine — an x86_64 Linux head with Apple Silicon and
arm64 Linux workers all interoperate, as long as the Ray version matches.

### macOS nodes

Ray refuses to form a multi-node cluster on macOS unless
`RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER=1` is set, failing with *"Multi-node Ray
clusters are not supported on Windows and OSX."* The head and worker scripts
set it automatically on Darwin and print a warning.

It does work over a LAN, but this is Ray's own caveat, not ours: the
configuration is unsupported upstream and gets far less testing. If you have a
choice, make the Linux machines the cluster and leave macOS as the client that
submits jobs.

## Setup

### 1. On every machine

Clone the repo at the *same commit* and sync:

```bash
git clone <your-fork-url> ml-homelab && cd ml-homelab && scripts/ray-node-setup.sh
```

The script checks for `uv` (printing the install command if it is missing),
syncs the locked environment, and prints the version report. Supported:
Linux x86_64, Linux arm64, and macOS on Apple Silicon.

### 2. On the head machine

```bash
make head-start
```

It prints the GCS address to give the workers, e.g. `192.168.1.10:6379`.

### 3. On each worker machine

```bash
make worker-start HEAD=192.168.1.10
```

Or put `RAY_HEAD_ADDRESS=192.168.1.10` in that machine's `.env` and run
`make worker-start`.

### 4. Confirm

From any node:

```bash
make node-status
```

Every machine should appear under "Node status", with the CPUs and memory of
the cluster summed under "Resources".

## Ports

The scripts pin these rather than letting Ray pick at random, so a firewall
rule can be written once. All of them need to be open **both ways** between
every pair of nodes.

| Port | Purpose | Where |
|---|---|---|
| 6379 | GCS (`RAY_HEAD_PORT`) | head |
| 6380 | Node manager (`RAY_NODE_MANAGER_PORT`) | every node |
| 6381 | Object manager (`RAY_OBJECT_MANAGER_PORT`) | every node |
| 8265 | Dashboard + Jobs API (`DASHBOARD_PORT`) | head |
| 10001 | Ray Client (`HEAD_NODE_PORT`) | head |
| 8080 | Prometheus metrics (`METRICS_EXPORT_PORT`) | every node |
| 11000-11999 | Ray worker processes (`RAY_MIN_WORKER_PORT` / `RAY_MAX_WORKER_PORT`) | every node |

The worker range is pinned deliberately. Ray's own default is 10002-19999,
which is wide enough to swallow the Ray Client port and makes the rule above
impossible to state precisely; 1000 ports is ample for a homelab. Raise
`RAY_MAX_WORKER_PORT` if you run more than that many worker processes on one
machine.

## Security

**Ray has no authentication.** Anything that can reach the GCS port can submit
arbitrary code to the cluster and it will run. `make head-start` binds
`0.0.0.0` because remote workers must reach it, and warns about this on
startup. Run it only on a network you trust, and never forward these ports
through a router.

This is the same reason the Floci S3 container is published on loopback only
(see `docker-compose.yaml`). If your Ray workers need to reach Floci, set
`FLOCI_BIND_HOST=0.0.0.0` deliberately, with the same caveat.

## Stopping

Per machine:

```bash
make node-stop
```

Stopping a worker drains it from the cluster. Stopping the head tears the
cluster down — do the workers first for a clean shutdown.

## Troubleshooting

**`Port 6379 is already in use`** — usually a leftover `kubectl port-forward`
from the Kind topology, or a previous head. Find it with
`lsof -nP -iTCP:6379 -sTCP:LISTEN`, and run `make node-stop` for a previous
head.

**`Version mismatch — this worker cannot join`** — the machines are on
different commits. `git log -1 --format=%H` on each, check out the same one,
and re-run `scripts/ray-node-setup.sh` on the worker.

**Worker joins, then disappears** — almost always the return path: the head
can reach the worker's GCS but not its node manager or object manager. Confirm
6380 and 6381 are open *from the head to the worker*, not just the reverse.

**`Could not reach the head dashboard`** — only a warning. The pre-flight
version check is skipped; Ray still enforces the version match itself.
