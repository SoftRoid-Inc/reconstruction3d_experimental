"""メッシュ(TSDF / Poisson)+ 点群オーバーレイの単体HTMLビューア

collision_viewer.py と同じWebGLテンプレートを流用し、表示グループだけ差し替える:
  structure -> Poissonメッシュ (半透明)
  obstacle  -> TSDFメッシュ (不透明)
  points    -> 位置合わせ済み点群

実行: IMG_SESSION=0312 conda run -n panovggt python -m img_align.mesh.mesh_viewer
出力: results/backend/mesh_viewer_<sess>.html
"""
import base64
import json

import numpy as np
import open3d as o3d

from img_align.mesh.collision_mesh import CURTAIN_COLORS, OUT_DIR, PLY_NAME
from img_align.mesh.collision_viewer import TEMPLATE
from img_align.config import SESSION
from img_align.mesh.meshing import SFX
from img_align.core.io_ply import read_ply_xyzrgb

VIEW_TRIS = 260_000      # ブラウザ用にメッシュを間引く目標三角形数
MAX_POINTS = 420_000
GROUP_COLOR = {"poisson": (150, 165, 190), "tsdf": (240, 150, 60)}
SLOT = {"poisson": "structure", "tsdf": "obstacle"}


def load_mesh(kind):
    p = OUT_DIR / f"mesh_{kind}{SFX}.ply"
    if not p.exists():
        return None
    m = o3d.io.read_triangle_mesh(str(p))
    if len(m.triangles) > VIEW_TRIS:
        m = m.simplify_quadric_decimation(VIEW_TRIS)
    v = np.asarray(m.vertices)
    t = np.asarray(m.triangles)
    return v[t].reshape(-1, 3)      # 三角形スープ (flat shading)


def main():
    payload, lo, hi = {}, np.full(3, np.inf), np.full(3, -np.inf)
    for kind in ("poisson", "tsdf"):
        tri = load_mesh(kind)
        if tri is None:
            continue
        payload[SLOT[kind]] = (tri, GROUP_COLOR[kind])
        lo = np.minimum(lo, tri.min(axis=0))
        hi = np.maximum(hi, tri.max(axis=0))
        print(f"{kind}: {len(tri)//3:,} tris")

    xyz, rgb = read_ply_xyzrgb(OUT_DIR / PLY_NAME[SESSION])
    keep = np.ones(len(xyz), bool)
    for c in CURTAIN_COLORS:
        keep &= ~np.all(rgb == np.array(c, np.uint8), axis=1)
    xyz, rgb = xyz[keep], rgb[keep]
    if len(xyz) > MAX_POINTS:
        step = len(xyz) // MAX_POINTS + 1
        xyz, rgb = xyz[::step], rgb[::step]
    lo = np.minimum(lo, xyz.min(axis=0))
    hi = np.maximum(hi, xyz.max(axis=0))

    meta = {"lo": lo.tolist(), "hi": hi.tolist(), "groups": {}}
    blob = b""
    for slot, (tri, color) in payload.items():
        q = np.round((tri - lo) / (hi - lo) * 65535).astype(np.uint16)
        meta["groups"][slot] = {"n": len(tri), "offset": len(blob),
                                "color": list(color)}
        blob += q.tobytes()
    qp = np.round((xyz - lo) / (hi - lo) * 65535).astype(np.uint16)
    meta["points"] = {"n": len(xyz), "offset": len(blob)}
    blob += qp.tobytes()
    meta["points"]["rgb_offset"] = len(blob)
    blob += rgb.astype(np.uint8).tobytes()

    html = TEMPLATE.replace("/*DATA*/",
                            f"const META={json.dumps(meta)};\n"
                            f"const B64=\"{base64.b64encode(blob).decode()}\";")
    html = html.replace("__TITLE__", f"mesh {SESSION} (Poisson / TSDF)")
    html = (html.replace("構造 (図面由来)", "Poissonメッシュ")
                .replace("障害物 (点群由来)", "TSDFメッシュ")
                .replace("スラブ", "(未使用)"))
    out = OUT_DIR / f"mesh_viewer{SFX}.html"
    out.write_text(html)
    print(f"-> {out} ({len(xyz):,} pts overlay)")


if __name__ == "__main__":
    main()
