"""Export Ray's stock Grafana dashboards into the Grafana provisioning directory.

Ray builds these dashboards from code in ``ray.dashboard.modules.metrics`` and
writes them into the session directory at runtime. Committing them lets Grafana
provision them without a running cluster — but it also means they go stale
whenever Ray is upgraded, so re-run this after every Ray version bump:

    make dashboards

The filenames match the ones Ray's own ``metrics_head`` writes, so a dashboard
provisioned from this directory is the same one the Ray dashboard links to.
"""

import argparse
from collections.abc import Callable
from pathlib import Path

import ray
from ray.dashboard.modules.metrics import grafana_dashboard_factory as factory

OUTPUT_DIR = Path("config/grafana/provisioning/dashboards/json")

# Filename -> generator, mirroring what ray's metrics_head writes on startup.
# Each generator returns ``(dashboard_json, uid)``.
DASHBOARDS: dict[str, Callable[[], tuple[str, str]]] = {
    "default_grafana_dashboard.json": factory.generate_default_grafana_dashboard,
    "data_grafana_dashboard.json": factory.generate_data_grafana_dashboard,
    "data_llm_grafana_dashboard.json": factory.generate_data_llm_grafana_dashboard,
    "serve_grafana_dashboard.json": factory.generate_serve_grafana_dashboard,
    "serve_deployment_grafana_dashboard.json": (
        factory.generate_serve_deployment_grafana_dashboard
    ),
    "serve_llm_grafana_dashboard.json": factory.generate_serve_llm_grafana_dashboard,
    "serve_llm_sglang_grafana_dashboard.json": (
        factory.generate_serve_llm_sglang_grafana_dashboard
    ),
    "train_grafana_dashboard.json": factory.generate_train_grafana_dashboard,
}


def export(output_dir: Path = OUTPUT_DIR) -> dict[str, str]:
    """Write every dashboard to ``output_dir``.

    Args:
        output_dir: Directory to write the dashboard JSON into. Created if
            it does not already exist.

    Returns:
        A mapping of filename to the dashboard's Grafana uid.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    uids = {}
    for filename, generate in DASHBOARDS.items():
        content, uid = generate()
        # Ray emits no trailing newline; add one so end-of-file-fixer does not
        # rewrite every file the moment it is exported.
        if not content.endswith("\n"):
            content += "\n"
        (output_dir / filename).write_text(content)
        uids[filename] = uid
    return uids


def main(argv: list[str] | None = None) -> None:
    """Export the dashboards and report what was written."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Where to write the dashboard JSON",
    )
    args = parser.parse_args(argv)

    uids = export(args.output_dir)
    print(f"Exported {len(uids)} dashboards from Ray {ray.__version__}:")
    for filename, uid in uids.items():
        print(f"  {filename}  (uid: {uid})")


if __name__ == "__main__":
    main()
