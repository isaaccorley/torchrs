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


__all__ = [
    "PROBAVDataModule", "ETCI2021DataModule", "RSVQALRDataModule", "RSVQAxBENDataModule",
    "EuroSATRGBDataModule", "EuroSATMSDataModule", "RESISC45DataModule", "RSICDDataModule",
    "OSCDDataModule", "S2LookingDataModule", "LEVIRCDPlusDataModule", "FAIR1MDataModule"
]
