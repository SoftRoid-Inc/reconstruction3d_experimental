"""pytest がある環境向けの薄いラッパ (本体は check_*.py)

ホストには pytest が入っていないので、実体は素のスクリプトに置いてある。
  python -m img_align.tests.check_config
  python -m img_align.tests.check_imports
"""
import pytest

from img_align.tests import check_config, check_imports
from img_align.tests.conftest import known_sessions


def test_config_snapshot():
    d = check_config.diff_snapshot()
    if d is None:
        pytest.skip("ベースライン未作成 (check_config --update)")
    assert not d, f"config の凍結値が変わった: {d}"


def test_mm_per_px_fixed():
    assert not check_config.mm_per_px_mismatch()


@pytest.mark.parametrize("session", check_imports.DEFAULT_SESSIONS)
def test_import_smoke(session):
    if session not in known_sessions():
        pytest.skip(f"{session} が未登録 (ext/ 未配置)")
    bad = check_imports.smoke(session)
    assert not bad, "\n".join(f"{m}: {t}: {e}" for m, t, e in bad)
