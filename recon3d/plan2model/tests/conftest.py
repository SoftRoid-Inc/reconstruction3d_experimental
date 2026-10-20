import json

import pytest

GT_DIR = "/code2/reconstruction3d_experimental/28-AAAAA/zumen"


@pytest.fixture
def tiny_labelme(tmp_path):
    """壁2(縦横)・柱1・窓1・ドア1の最小labelme。画像100x80px想定。"""
    data = {
        "imageWidth": 100, "imageHeight": 80,
        "shapes": [
            {"label": "wall", "shape_type": "rectangle",
             "points": [[10.0, 10.0], [90.0, 14.0]]},          # 横壁
            {"label": "wall", "shape_type": "rectangle",
             "points": [[90.0, 70.0], [86.0, 10.0]]},          # 縦壁(点順が逆)
            {"label": "column", "shape_type": "rectangle",
             "points": [[10.0, 10.0], [14.0, 14.0]]},
            {"label": "window", "shape_type": "rectangle",
             "points": [[30.0, 9.0], [50.0, 15.0]]},
            {"label": "door", "shape_type": "rectangle",
             "points": [[60.0, 9.5], [75.0, 14.5]]},
        ],
    }
    p = tmp_path / "t.json"
    p.write_text(json.dumps(data))
    return str(p)


@pytest.fixture
def tiny_prediction(tmp_path):
    data = {
        "image_size": [100, 80],
        "walls": [
            {"id": 0, "polygon": [[10, 10], [90, 10], [90, 14], [10, 14]],
             "is_column": False},
            {"id": 1, "polygon": [[86, 10], [90, 10], [90, 14], [86, 14]],
             "is_column": True},
        ],
        "windows": [{"polygon": [[30, 9], [50, 9], [50, 15], [30, 15]]}],
        "doors": [],
        "rooms": [],
    }
    p = tmp_path / "p.json"
    p.write_text(json.dumps(data))
    return str(p)
