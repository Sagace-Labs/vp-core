"""The manifest signature names even an empty variable-length result."""

from __future__ import annotations

import numpy as np
import pytest

from vp_core.registry import Version


def test_empty_frame_preserves_declared_column(tmp_path):
    version = Version("som", "v1", tmp_path, {"signature": {"outputs": [{"name": "som_score"}]}})
    assert version.frame(np.empty((0, 1))).columns.tolist() == ["som_score"]
    with pytest.raises(ValueError, match="produced 2 columns"):
        version.frame(np.zeros((3, 2)))
