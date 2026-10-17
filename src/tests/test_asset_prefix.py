from pathlib import Path

import pytest

import duobench


@pytest.fixture()
def downloads(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    calls: list[Path] = []

    def fake_download(_version: str, prefix: Path, _url: str, assets_dirname: str) -> None:
        calls.append(prefix)
        (prefix / assets_dirname).mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(duobench, "download_assets", fake_download)
    return calls


def test_assets_downloaded_once(tmp_path: Path, downloads: list[Path]):
    prefix = tmp_path / "prefix"
    for _ in range(3):
        assert duobench._get_duobench_prefix(str(prefix), tmp_path / "pkg", tmp_path / "default") == str(prefix)
    assert downloads == [prefix]
    assert (prefix / duobench.ASSETS_VERSION_FILE).read_text() == duobench.__version__


def test_default_prefix_used_without_env(tmp_path: Path, downloads: list[Path]):
    default = tmp_path / "default"
    for _ in range(2):
        assert duobench._get_duobench_prefix(None, tmp_path / "pkg", default) == str(default)
    assert downloads == [default]


def test_assets_redownloaded_on_version_change(tmp_path: Path, downloads: list[Path]):
    prefix = tmp_path / "prefix"
    (prefix / "assets").mkdir(parents=True)
    (prefix / duobench.ASSETS_VERSION_FILE).write_text("0.0.0")
    duobench._get_duobench_prefix(str(prefix), tmp_path / "pkg", tmp_path / "default")
    assert downloads == [prefix]
    assert (prefix / duobench.ASSETS_VERSION_FILE).read_text() == duobench.__version__


def test_editable_install_skips_download(tmp_path: Path, downloads: list[Path]):
    editable = tmp_path / "pkg"
    (editable / "assets").mkdir(parents=True)
    assert duobench._get_duobench_prefix(None, editable, tmp_path / "default") == str(editable)
    assert downloads == []
