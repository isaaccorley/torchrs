from .base import BaseDataModule
from .probav import PROBAVDataModule
from .etci2021 import ETCI2021DataModule
from .rsvqa import RSVQALRDataModule, RSVQAxBENDataModule
from .eurosat import EuroSATRGBDataModule, EuroSATMSDataModule
from .resisc45 import RESISC45DataModule
from .rsicd import RSICDDataModule
from .oscd import OSCDDataModule
from .s2looking import S2LookingDataModule
from .levircd import LEVIRCDPlusDataModule
from .fair1m import FAIR1MDataModule
from .sydney_captions import SydneyCaptionsDataModule
from .ucm_captions import UCMCaptionsDataModule
from .s2mtcp import S2MTCPDataModule
from .advance import ADVANCEDataModule
from .sat import SAT4DataModule, SAT6DataModule
from .hrscd import HRSCDDataModule
from .inria_ail import InriaAILDataModule
from .tiselac import TiselacDataModule
from .gid15 import GID15DataModule
from .zuericrop import ZueriCropDataModule
from .aid import AIDDataModule


__all__ = [
    "BaseDataModule", "PROBAVDataModule", "ETCI2021DataModule", "RSVQALRDataModule",
    "RSVQAxBENDataModule", "EuroSATRGBDataModule", "EuroSATMSDataModule", "RESISC45DataModule",
    "RSICDDataModule", "OSCDDataModule", "S2LookingDataModule", "LEVIRCDPlusDataModule",
    "FAIR1MDataModule", "SydneyCaptionsDataModule", "UCMCaptionsDataModule", "S2MTCPDataModule",
    "ADVANCEDataModule", "SAT4DataModule", "SAT6DataModule", "HRSCDDataModule", "InriaAILDataModule",
    "TiselacDataModule", "GID15DataModule", "ZueriCropDataModule", "AIDDataModule"
]
