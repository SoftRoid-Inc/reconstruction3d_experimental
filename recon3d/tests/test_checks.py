"""pytest がある環境向けの薄いラッパ (本体は check_*.py)

ホストには pytest が入っていないので、実体は素のスクリプトに置いてある。
  python -m recon3d.tests.check_config
  python -m recon3d.tests.check_imports
"""
import pytest

from recon3d.tests import check_config, check_imports
from recon3d.tests.conftest import known_sessions


def test_config_snapshot():
    d = check_config.diff_snapshot()
    if d is None:
        pytest.skip("ベースライン未作成 (check_config --update)")
    assert not d, f"config の凍結値が変わった: {d}"


def test_mm_per_px_fixed():
    assert not check_config.mm_per_px_mismatch()


def test_clean_imports():
    assert not check_imports.clean_imports()


@pytest.mark.parametrize("session", check_imports.DEFAULT_SESSIONS)
def test_import_smoke(session):
    if session not in known_sessions():
        pytest.skip(f"{session} が未登録 (ext/ 未配置)")
    bad = check_imports.smoke(session)
    assert not bad, "\n".join(f"{m}: {t}: {e}" for m, t, e in bad)
