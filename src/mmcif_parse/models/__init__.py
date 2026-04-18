"""Data models for mmCIF parsing."""

from .complex import AntibodyAntigenComplexRecord, TcrPmhcComplexRecord
from .contact import DimerInterfaceRecord
from .multimer import TightMultimerRecord
from .structure import ChainRecord, StructureSummary

__all__ = [
    "AntibodyAntigenComplexRecord",
    "ChainRecord",
    "DimerInterfaceRecord",
    "StructureSummary",
    "TightMultimerRecord",
    "TcrPmhcComplexRecord",
]
