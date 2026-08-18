"""Tests for the Grafana dashboard exporter.

These guard against the failure mode that prompted the exporter: dashboards
drifting out of sync with the installed Ray version without anyone noticing.
"""

import json

import pytest
import ray

from scripts import export_grafana_dashboards as exporter

COMMITTED_DIR = exporter.OUTPUT_DIR


def test_every_generator_is_callable_on_the_installed_ray():
    """A Ray upgrade that renames a generator must fail loudly, not silently."""
    for filename, generate in exporter.DASHBOARDS.items():
        content, uid = generate()
        assert uid, f"{filename} produced no uid"
        assert json.loads(content)["panels"], f"{filename} produced no panels"


def test_export_writes_every_dashboard(tmp_path):
    uids = exporter.export(tmp_path)

    written = {path.name for path in tmp_path.glob("*.json")}
    assert written == set(exporter.DASHBOARDS)
    assert set(uids) == set(exporter.DASHBOARDS)


def test_export_creates_a_missing_output_directory(tmp_path):
    target = tmp_path / "nested" / "json"
    exporter.export(target)
    assert (target / "default_grafana_dashboard.json").is_file()


def test_exported_dashboards_have_unique_uids(tmp_path):
    uids = exporter.export(tmp_path)
    assert len(set(uids.values())) == len(uids)


def test_committed_dashboards_match_the_installed_ray_version(tmp_path):
    """The checked-in JSON must be what this Ray version generates."""
    exporter.export(tmp_path)

    stale = [
        name
        for name in exporter.DASHBOARDS
        if json.loads((tmp_path / name).read_text())
        != json.loads((COMMITTED_DIR / name).read_text())
    ]

    assert not stale, (
        f"Dashboards are stale for Ray {ray.__version__}: {stale}. Run `make dashboards`."
    )


def test_main_reports_what_it_wrote(tmp_path, capsys):
    exporter.main(["--output-dir", str(tmp_path)])

    out = capsys.readouterr().out
    assert (
        f"Exported {len(exporter.DASHBOARDS)} dashboards from Ray {ray.__version__}"
        in out
    )
    assert "rayDefaultDashboard" in out


@pytest.mark.parametrize("filename", sorted(exporter.DASHBOARDS))
def test_committed_dashboard_is_valid_json(filename):
    dashboard = json.loads((COMMITTED_DIR / filename).read_text())
    assert dashboard["panels"]
