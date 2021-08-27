from . import utils
from .probav import PROBAV
from .etci2021 import ETCI2021
from .rsvqa import RSVQALR, RSVQAxBEN
from .eurosat import EuroSATRGB, EuroSATMS
from .resisc45 import RESISC45
from .rsicd import RSICD
from .oscd import OSCD
from .s2looking import S2Looking
from .levircd import LEVIRCDPlus
from .fair1m import FAIR1M
from .sydney_captions import SydneyCaptions
from .ucm_captions import UCMCaptions
from .s2mtcp import S2MTCP
from .advance import ADVANCE
from .sat import SAT4, SAT6
from .hrscd import HRSCD
from .inria_ail import InriaAIL
from .tiselac import Tiselac
from .gid15 import GID15


__all__ = [
    "PROBAV", "ETCI2021", "RSVQALR", "RSVQAxBEN", "EuroSATRGB", "EuroSATMS",
    "RESISC45", "RSICD", "OSCD", "S2Looking", "LEVIRCDPlus", "FAIR1M",
    "SydneyCaptions", "UCMCaptions", "S2MTCP", "ADVANCE", "SAT4", "SAT6",
    "HRSCD", "InriaAIL", "Tiselac", "GID15"
]
