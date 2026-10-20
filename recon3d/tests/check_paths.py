"""入力契約 (`Paths`) が実物のディレクトリを正しく読めるか

`RECON3D_FIXTURE` に入力ディレクトリを指して実行する:

  RECON3D_FIXTURE=/path/to/recon_0912 python -m recon3d.tests.check_paths

入力の所在は `Paths` ひとつが決めるので、ここが狂うと全段が静かに
別の場所を読む。実物で測れる性質 (frame の桁数、meta の型、出力が
入力の下に来ること) だけを見る。
"""
import sys

from recon3d.paths import from_input
from recon3d.tests.conftest import fixture_dir


def checks(paths):
    """(名前, 真偽, 実測値) のリスト"""
    out = [
        ("images/ がある", paths.images.is_dir(), str(paths.images)),
        ("フレームが1枚以上", paths.n_frames > 0,
         f"{len(paths.frame_indices)}枚 / 上限 {paths.n_frames}"),
        ("ゼロ詰め桁数が実物と一致",
         (paths.images / paths.frame_name(paths.frame_indices[0])).is_file()
         if paths.frame_indices else False,
         f"pad={paths.frame_pad} -> "
         f"{paths.frame_name(paths.frame_indices[0]) if paths.frame_indices else '-'}"),
        ("meta は dict", isinstance(paths.meta, dict), str(paths.meta)[:80]),
        ("出力は入力の下", paths.results.parent == paths.root,
         str(paths.results)),
        ("出力名が二重化していない",
         paths.results.name == "results", paths.results.name),
        ("タグ接頭辞 = 入力ディレクトリ名", paths.name == paths.root.name,
         paths.name),
    ]
    if paths.known_width_mm is not None:
        out.append(("known_width_mm が正の数", _positive(paths.known_width_mm),
                    str(paths.known_width_mm)))
    return out


def _positive(kw):
    vals = kw.values() if isinstance(kw, dict) else [kw]
    return all(isinstance(v, (int, float)) and v > 0 for v in vals)


def failures(root):
    return [(n, v) for n, ok, v in checks(from_input(root)) if not ok]


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else fixture_dir()
    if root is None:
        print("[SKIP] RECON3D_FIXTURE 未設定 (入力ディレクトリを指すこと)")
        return 0
    paths = from_input(root)
    rc = 0
    print(f"入力: {paths.root}")
    for name, ok, val in checks(paths):
        print(f"  [{'OK' if ok else 'NG'}] {name}: {val}")
        rc |= not ok
    return rc


if __name__ == "__main__":
    sys.exit(main())
