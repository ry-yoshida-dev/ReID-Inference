"""Shared NumPy array type aliases for ReID inference."""

from __future__ import annotations

from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

NumericArray: TypeAlias = NDArray[np.integer[Any] | np.floating[Any]]
FloatArray: TypeAlias = NDArray[np.floating[Any]]
