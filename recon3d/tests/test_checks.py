"""pytest がある環境向けの薄いラッパ (本体は check_*.py)

ホストには pytest が入っていないので、実体は素のスクリプトに置いてある。
  python -m recon3d.tests.check_imports
  RECON3D_FIXTURE=<入力dir> python -m recon3d.tests.check_paths
"""
import pytest

from recon3d.tests import check_imports, check_paths
from recon3d.tests.conftest import fixture_dir


def test_import_smoke():
    bad = check_imports.smoke()
    assert not bad, "\n".join(f"{m}: {t}: {e}" for m, t, e in bad)


def test_paths_has_no_default():
    bad = check_imports.paths_with_default()
    assert not bad, "\n".join(f"{f}:{ln} {fn}" for f, ln, fn in bad)


def test_entry_signatures():
    assert not check_imports.entry_signatures()


def test_fixture_paths():
    d = fixture_dir()
    if d is None:
        pytest.skip("RECON3D_FIXTURE 未設定")
    assert not check_paths.failures(d)
