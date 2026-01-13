"""Deprecated compatibility shim.

This module used to contain processor utilities. It has been renamed to
`pron_inpaint.utils`. Importing from here will emit a DeprecationWarning and
re-export the useful symbols for backward compatibility.
"""

import warnings

warnings.warn(
    "pron_inpaint.processor is deprecated; import from pron_inpaint.utils instead",
    DeprecationWarning,
)

from .utils import *  # re-export everything for compatibility

__all__ = [s for s in dir() if not s.startswith("_")]
