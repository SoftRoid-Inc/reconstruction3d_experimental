"""全モジュールが import できること + 「paths を省けない」ことを確認する

`eval/gt_traj.py` は以前 import 時に図面を読んでいた。そこで例外が出ると、
これを top-level import する20モジュール + `postproc/plan_icp` 経由の
二次伝播16モジュール (= アンカー経路まるごと) が **import 不能**になる。
だから「import が通る」こと自体が守るべき不変条件。

  python -m recon3d.tests.check_imports
"""
import ast
import pathlib
import sys

from recon3d.tests.conftest import ROOT, run_json

# GPU / 別コンテナの依存で、ホストでは import できないのが正常なもの
SKIP_PARTS = {"tests", "__pycache__"}
SKIP_MODULES = {
    "recon3d.core.panovggt",      # PanoVGGT 本体 (GPU)
    "recon3d.postproc.mapany",    # mapanything コンテナ
}

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
        if m in SKIP_MODULES:
            continue
        names.append(m)
    return names


def _is_env(rec):
    """環境差 (ホストに無いサードパーティ) による失敗か"""
    _, typ, _, name = rec
    return typ == "ModuleNotFoundError" and (name or "").split(".")[0] \
        in EXTERNAL_DEPS


def smoke():
    """退行とみなす失敗だけ [(module, exc_type, message), ...] で返す"""
    raw = run_json(_CODE, timeout=600)
    return [(m, t, e) for m, t, e, _ in raw if not _is_env((m, t, e, _))]


def smoke_all():
    """環境差も含めた生の結果"""
    return run_json(_CODE, timeout=600)


# ------------------------------------------------------- paths を省けないこと
# かつては `IMG_SESSION` 未設定が既定セッション "0828" に落ち、新しい物件を
# 28-AAAAA の図面・GT で処理して**例外なしに完走した** (2026-10-09)。
# いまは所在を `Paths` に持たせて引数で配るので、渡し忘れは `TypeError` に
# なる — ただしそれは **`paths=None` のような既定値を置かない限り**。
# 既定値を一つ置いた瞬間に昔と同じ「黙って別物を読む」形が戻るので、
# 静的に禁じておく。

def paths_with_default():
    """第1引数 `paths` に既定値を与えている関数を返す ([] なら健全)"""
    bad = []
    base = pathlib.Path(ROOT) / "recon3d"
    for p in sorted(base.rglob("*.py")):
        if "__pycache__" in p.parts:
            continue
        tree = ast.parse(p.read_text())
        for n in ast.walk(tree):
            if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            a = n.args
            pos = [x for x in a.posonlyargs + a.args
                   if x.arg not in ("self", "cls")]
            if not pos or pos[0].arg != "paths":
                continue
            # 既定値は末尾から詰まる。`paths` まで届いていたら NG
            if len(a.defaults) >= len(pos):
                bad.append((str(p.relative_to(ROOT)), n.lineno, n.name))
    return bad


# 本番経路の要。`paths` を受け取らなくなったら、どこかに所在が固定されている
MUST_TAKE_PATHS = [
    ("recon3d.steps", "plan"),
    ("recon3d.steps", "execute"),
    ("recon3d.core.plan_grid", "PlanGrid"),
    ("recon3d.eval.gt_traj", "load_gt"),
    ("recon3d.out.export", "export"),
    ("recon3d.plan2model.build", "build_from_paths"),
]

_SIG_CODE = """
import importlib, inspect, json
EXTERNAL = %r
bad = []
for mod, name in %r:
    try:
        obj = getattr(importlib.import_module(mod), name)
    except ModuleNotFoundError as e:
        if (e.name or "").split(".")[0] in EXTERNAL:
            continue                      # ホストに無いだけ (環境差)
        bad.append([f"{mod}.{name}", f"取得できない: {e}"])
        continue
    except Exception as e:
        bad.append([f"{mod}.{name}", f"取得できない: {type(e).__name__}: {e}"])
        continue
    try:
        params = list(inspect.signature(obj).parameters)
    except (TypeError, ValueError) as e:
        bad.append([f"{mod}.{name}", f"signature 不可: {e}"]); continue
    if not params or params[0] != "paths":
        bad.append([f"{mod}.{name}", f"第1引数が paths でない: {params[:2]}"])
print(json.dumps(bad, ensure_ascii=False))
"""


def entry_signatures():
    """本番入口が `paths` を受けていることを確かめる ([] なら健全)"""
    return [tuple(r) for r in run_json(_SIG_CODE % (sorted(EXTERNAL_DEPS), MUST_TAKE_PATHS))]


def main():
    n = len(module_names())
    raw = smoke_all()
    bad = [r for r in raw if not _is_env(r)]
    env = [r for r in raw if _is_env(r)]
    print(f"[{'OK' if not bad else 'NG'}] import: 失敗 {len(bad)} / 全 {n}"
          + (f" (環境差 {len(env)}件は除外)" if env else ""))
    for m, t, e, _ in bad:
        print(f"    {m}: {t}: {e}")
    rc = bool(bad)

    dflt = paths_with_default()
    print(f"[{'OK' if not dflt else 'NG'}] paths に既定値を置かないこと")
    for f, ln, fn in dflt:
        print(f"    {f}:{ln} {fn}")
    rc |= bool(dflt)

    sig = entry_signatures()
    print(f"[{'OK' if not sig else 'NG'}] 本番入口 {len(MUST_TAKE_PATHS)}件が "
          "paths を受けること")
    for m, why in sig:
        print(f"    {m}: {why}")
    rc |= bool(sig)
    return rc


if __name__ == "__main__":
    sys.exit(main())
