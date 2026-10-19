"""全モジュールが import できることを確認する

`eval/gt_traj.py` は import 時に図面を読む。そこで `TypeError`/`ValueError`
が出ると、これを top-level import する20モジュール + `postproc/plan_icp`
経由の二次伝播16モジュール (= アンカー経路まるごと) が **import 不能**に
なる。だから「import が通る」こと自体が守るべき不変条件。

  python -m img_align.tests.check_imports            # 既定3セッション
  python -m img_align.tests.check_imports 0616 0925
"""
import pathlib
import sys

from img_align.tests.conftest import ROOT, known_sessions, run_json

DEFAULT_SESSIONS = ("0828", "0912", "e11196")

# GPU / 別コンテナの依存で、ホストでは import できないのが正常なもの
SKIP_PARTS = {"legacy", "tests", "vendor", "__pycache__"}
SKIP_MODULES = {
    "img_align.core.panovggt",      # PanoVGGT 本体 (GPU)
    "img_align.postproc.mapany",    # mapanything コンテナ
    "img_align.seg_ade20k",         # transformers + GPU
}
SKIP_PREFIX = ("img_align.gs.",)    # gsplat コンテナ

# ホストに無くて当然のサードパーティ (コンテナ側にだけある)。これらの
# ModuleNotFoundError は環境差であって退行ではないので失敗に数えない
EXTERNAL_DEPS = {"shapely", "torch", "mapanything", "transformers", "gsplat",
                 "open3d", "pycolmap", "trimesh", "cv2"}

_CODE = """
import importlib, json
from img_align.tests.check_imports import module_names
bad = []
for m in module_names():
    try:
        importlib.import_module(m)
    except Exception as e:
        bad.append([m, type(e).__name__, str(e)[:300],
                    getattr(e, "name", None)])
print(json.dumps(bad, ensure_ascii=False))
"""


def module_names():
    base = pathlib.Path(ROOT) / "img_align"
    names = []
    for p in sorted(base.rglob("*.py")):
        if p.name == "__init__.py" or SKIP_PARTS & set(p.parts):
            continue
        m = "img_align." + ".".join(p.relative_to(base).with_suffix("").parts)
        if m in SKIP_MODULES or m.startswith(SKIP_PREFIX):
            continue
        names.append(m)
    return names


def _is_env(rec):
    """環境差 (ホストに無いサードパーティ) による失敗か"""
    _, typ, _, name = rec
    return typ == "ModuleNotFoundError" and (name or "").split(".")[0] \
        in EXTERNAL_DEPS


def smoke(session):
    """退行とみなす失敗だけ [(module, exc_type, message), ...] で返す"""
    raw = run_json(_CODE, session=session, timeout=600)
    return [(m, t, e) for m, t, e, _ in raw if not _is_env((m, t, e, _))]


def smoke_all(session):
    """環境差も含めた生の結果"""
    return run_json(_CODE, session=session, timeout=600)


def main():
    avail = known_sessions()
    todo = [s for s in (sys.argv[1:] or DEFAULT_SESSIONS)
            if s in avail or "/" in s]
    n = len(module_names())
    rc = 0
    for s in todo:
        raw = smoke_all(s)
        bad = [r for r in raw if not _is_env(r)]
        env = [r for r in raw if _is_env(r)]
        print(f"[{'OK' if not bad else 'NG'}] {s}: 失敗 {len(bad)} / 全 {n}"
              + (f" (環境差 {len(env)}件は除外)" if env else ""))
        for m, t, e, _ in bad:
            print(f"    {m}: {t}: {e}")
        rc |= bool(bad)
    return rc


if __name__ == "__main__":
    sys.exit(main())
