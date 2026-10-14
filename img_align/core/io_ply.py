"""バイナリPLY (x,y,z float32 + r,g,b uint8) の読み書き (全系統共通)"""
import numpy as np


def write_ply(path, xyz, rgb):
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {len(xyz)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\n"
              "end_header\n")
    data = np.empty(len(xyz), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                     ("r", "u1"), ("g", "u1"), ("b", "u1")])
    data["x"], data["y"], data["z"] = xyz.T.astype(np.float32)
    data["r"], data["g"], data["b"] = rgb.T
    with open(path, "wb") as f:
        f.write(header.encode())
        f.write(data.tobytes())


def read_ply_xyzrgb(path):
    """PanoVGGT inference.py が書く形式のバイナリPLYを読む"""
    with open(path, "rb") as f:
        header = b""
        while not header.endswith(b"end_header\n"):
            header += f.readline()
        n = int([l for l in header.decode().splitlines()
                 if l.startswith("element vertex")][0].split()[-1])
        dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                          ("r", "u1"), ("g", "u1"), ("b", "u1")])
        data = np.frombuffer(f.read(n * dtype.itemsize), dtype=dtype)
    xyz = np.stack([data["x"], data["y"], data["z"]], axis=1)
    rgb = np.stack([data["r"], data["g"], data["b"]], axis=1)
    return xyz, rgb
