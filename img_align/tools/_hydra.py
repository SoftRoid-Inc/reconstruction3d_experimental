"""hydra 設定読み込みの共通部 (tools/* で共有)"""
from pathlib import Path
from typing import List, Optional

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

ROOT = Path(__file__).resolve().parents[2]
CONF_DIR = ROOT / "img_align" / "conf"


def load_cfg(overrides: Optional[List[str]] = None,
             name: str = "recon") -> DictConfig:
    """hydra設定を読む。overrides は `key=value` の列"""
    with initialize_config_dir(config_dir=str(CONF_DIR), version_base=None):
        return compose(config_name=name, overrides=list(overrides or []))


def to_plain(cfg) -> dict:
    """DictConfig -> 素の dict (steps.py へ渡すため。null は None のまま)"""
    return OmegaConf.to_container(cfg, resolve=True)


def abs_path(value) -> Path:
    """相対パスはリポジトリルート基準で解決する"""
    p = Path(value).expanduser()
    return p if p.is_absolute() else ROOT / p
