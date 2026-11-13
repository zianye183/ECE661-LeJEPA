from ..backbones import build_backbone
from .base import SSLMethod
from .lejepa import LeJEPA
from .simclr import SimCLR
from .simsiam import SimSiam

METHODS = {"lejepa": LeJEPA, "simclr": SimCLR, "simsiam": SimSiam}


def build_method(cfg: dict) -> SSLMethod:
    backbone, feat_dim = build_backbone(cfg["backbone"]["name"])
    return METHODS[cfg["method"]["name"]](backbone, feat_dim, cfg["method"])
