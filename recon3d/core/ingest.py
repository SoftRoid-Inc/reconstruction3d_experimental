"""入力ディレクトリの外部生成物を results 側へ取り込む

IMU派生物 (transitions.json / idle_spans.json など) は別パイプラインが作り、
入力の `imu/` に置かれる。一方 `config.IMU_DIR` は**読み書き共用**で、
11モジュール14種類のファイルがここへ書く (`imu_raw_traj.npz` は数十MB)。

したがって:
  - `IMU_DIR` を入力側へ向ける … 入力を汚染するので不可
  - ディレクトリごと symlink    … 同上
  - ファイル単位 symlink        … docker (`/code2` マウント) から dangling

残るのは**コピー**。ただし `core/chunking.imu_segments` は `exists()` しか
見ないので、古いコピーが残ると入力を差し替えても永久に古い区間が使われる。
そこで取り込んだファイル名を `_materialized.json` に記録し、毎回それを
消してから張り直す。
"""
import json
import shutil

from recon3d.config import FRAMES_DIR, IMU_DIR, PROP

LEDGER = "_materialized.json"
PATTERNS = ("*.json", "*.npz")


def source_dir():
    """入力側の imu/ (無ければ None)

    新しい入力契約では images/ と imu/ が並ぶので `FRAMES_DIR.parent/imu`。
    28-BBBBB のように1つの root に複数セッションがぶら下がる物件でも、
    これならセッション別の imu/ を指す (root/imu だと全セッションで共用に
    なってしまう)。
    """
    for d in (FRAMES_DIR.parent / "imu", PROP.root / "imu"):
        if d.is_dir() and d.resolve() != IMU_DIR.resolve():
            return d
    return None


def materialize_imu(verbose=True):
    """入力の imu/* を IMU_DIR へ取り込む。取り込んだ名前のリストを返す"""
    src = source_dir()
    if src is None:
        return []
    IMU_DIR.mkdir(parents=True, exist_ok=True)
    ledger = IMU_DIR / LEDGER
    # 前回の取り込み分を先に消す (入力から消えたファイルを残さない)
    if ledger.exists():
        try:
            for name in json.loads(ledger.read_text()).get("files", []):
                (IMU_DIR / name).unlink(missing_ok=True)
        except Exception as e:                      # noqa: BLE001
            print(f"[ingest] {LEDGER} を読めない ({e}) -> 全部張り直す")
    names = []
    for pat in PATTERNS:
        for p in sorted(src.glob(pat)):
            if p.name == LEDGER:
                continue
            shutil.copy2(p, IMU_DIR / p.name)
            names.append(p.name)
    ledger.write_text(json.dumps({"src": str(src), "files": names}, indent=1))
    if verbose:
        print(f"[ingest] imu: {len(names)}件を {IMU_DIR} へ取り込み"
              + (f" ({', '.join(names)})" if names else ""))
    return names


def main():
    materialize_imu()


if __name__ == "__main__":
    main()
