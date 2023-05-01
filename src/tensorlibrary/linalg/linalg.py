"""
	Linear algebra helper functions
"""

import numpy as np
import tensorly as tl
from typing import Any, List, Optional, Text, Type, Union, Dict, Sequence


def truncated_svd(
		mat,
		max_rank: int = np.inf,
		max_trunc_error: Optional[float] = 0.0,
		relative: Optional[bool] = False,
):
	u, s, vh = tl.svd(mat, full_matrices=False)
	if np.isinf(max_rank) and max_trunc_error == 0.0:
		return u, s, vh, 0.0

	if max_trunc_error != 0.0:
		err = 0.0
		k = len(s) - 1
		if relative:
			max_trunc_error = max_trunc_error * np.max(s)
		while err <= max_trunc_error:
			err = err + s[k]
			k -= 1
		max_rank = min([k + 2, max_rank])

	max_rank = int(min([max_rank, len(s)]))
	err = tl.norm(s[max_rank:])
	u = u[:, :max_rank]
	s = s[:max_rank]
	vh = vh[:max_rank, :]

	return u, s, vh, err
