"""全モジュールが import できることを確認する

`eval/gt_traj.py` は import 時に図面を読む。そこで `TypeError`/`ValueError`
が出ると、これを top-level import する20モジュール + `postproc/plan_icp`
経由の二次伝播16モジュール (= アンカー経路まるごと) が **import 不能**に
なる。だから「import が通る」こと自体が守るべき不変条件。

  python -m recon3d.tests.check_imports            # 既定3セッション
  python -m recon3d.tests.check_imports 0616 0925
"""
import pathlib
import sys

from recon3d.tests.conftest import ROOT, known_sessions, run_json

DEFAULT_SESSIONS = ("0828", "0912", "e11196")

# GPU / 別コンテナの依存で、ホストでは import できないのが正常なもの
SKIP_PARTS = {"tests", "__pycache__"}
SKIP_MODULES = {
    "recon3d.core.panovggt",      # PanoVGGT 本体 (GPU)
    "recon3d.postproc.mapany",    # mapanything コンテナ
}
SKIP_PREFIX = ()

# ホストに無くて当然のサードパーティ (コンテナ側にだけある)。これらの
# ModuleNotFoundError は環境差であって退行ではないので失敗に数えない
EXTERNAL_DEPS = {"shapely", "torch", "mapanything", "transformers", "gsplat",
                 "open3d", "pycolmap", "trimesh", "cv2"}

_CODE = """
import importlib, json
from recon3d.tests.check_imports import module_names
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
    base = pathlib.Path(ROOT) / "recon3d"
    names = []
    for p in sorted(base.rglob("*.py")):
        if p.name == "__init__.py" or SKIP_PARTS & set(p.parts):
            continue
        m = "recon3d." + ".".join(p.relative_to(base).with_suffix("").parts)
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


# ---------------------------------------------------------------- import衛生
# 入口のモジュールは **import しただけで config を凍結してはいけない**。
# 凍結すると `IMG_SESSION` 未設定のとき既定 "0828" に落ち、新しい物件を
# 28-AAAAA の図面・GT で処理して**例外なしに完走する** (2026-10-09に踏んだ形)。
# api.run() 側の `_freeze` がこれを検出できるのも、api を import した時点で
# まだ config が読まれていないからこそ。
CLEAN_IMPORTS = ("recon3d.api", "recon3d.steps", "recon3d.tools.recon",
                 "recon3d.tools._hydra")

_CLEAN_CODE = """
import importlib, importlib.util, json, sys, pathlib
bad = []
for m in %r:
    try:
        importlib.import_module(m)
    except Exception as e:
        bad.append([m, f"import 自体が失敗: {type(e).__name__}: {e}"])
        continue
    if "recon3d.config" in sys.modules:
        bad.append([m, "import しただけで config が凍結した"])
# ポッドの ENTRYPOINT。パッケージ外なのでファイルから読み込んで確かめる
p = pathlib.Path(%r) / "main.py"
spec = importlib.util.spec_from_file_location("__entry_probe__", p)
mod = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(mod)
except Exception as e:
    bad.append(["main.py", f"import 自体が失敗: {type(e).__name__}: {e}"])
else:
    if "recon3d.config" in sys.modules:
        bad.append(["main.py", "import しただけで config が凍結した"])
print(json.dumps(bad, ensure_ascii=False))
"""


def clean_imports():
    """config を巻き込む入口モジュールを返す ([] なら健全)

    **`IMG_SESSION` を外した子プロセス**で見るのが要点。事故の形が
    「env 未設定のまま凍結」なので、env を立てたまま検査すると意味がない。
    """
    return [tuple(r) for r in
            run_json(_CLEAN_CODE % (list(CLEAN_IMPORTS), str(ROOT)),
                     session=None)]


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

    dirty = clean_imports()
    print(f"[{'OK' if not dirty else 'NG'}] import衛生: "
          f"{len(CLEAN_IMPORTS) + 1}入口 (+ main.py) が config を巻き込まないこと")
    for m, why in dirty:
        print(f"    {m}: {why}")
    rc |= bool(dirty)
    return rc


if __name__ == "__main__":
    sys.exit(main())
