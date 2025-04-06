from __future__ import annotations

import sys
import typing
from dataclasses import dataclass

import numpy.typing as npt

if sys.version_info < (3, 10):
    import typing_extensions as typing
else:
    import typing

# `slots` feature added to of `dataclass` in 3.10
# see: https://docs.python.org/3.12/library/dataclasses.html#dataclasses.dataclass
if sys.version_info < (3, 10):
    dataclass_kwargs = {}
else:
    dataclass_kwargs = {"slots": True}
