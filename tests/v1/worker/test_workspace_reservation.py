# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

import vllm.v1.worker.workspace as workspace_module
from vllm.v1.worker.workspace import WorkspaceManager


@pytest.mark.parametrize("num_ubatches", [1, 2, 3])
def test_reserve_all_ubatches_prevents_growth_after_lock(
    monkeypatch, num_ubatches: int
):
    manager = WorkspaceManager(torch.device("cpu"), num_ubatches=num_ubatches)
    shapes_and_dtypes = (
        ((4,), torch.float32),
        ((3,), torch.float16),
    )

    manager.reserve_all_ubatches(*shapes_and_dtypes)
    reserved_ptrs = [workspace.data_ptr() for workspace in manager._current_workspaces]
    assert len(set(reserved_ptrs)) == num_ubatches
    manager.lock()

    view_ptrs = []
    for ubatch_id in range(num_ubatches):
        monkeypatch.setattr(
            workspace_module,
            "dbo_current_ubatch_id",
            lambda ubatch_id=ubatch_id: ubatch_id,
        )
        views = manager.get_simultaneous(*shapes_and_dtypes)
        assert len(views) == 2
        view_ptrs.append(views[0].data_ptr())
        assert (
            manager._current_workspaces[ubatch_id].data_ptr()
            == reserved_ptrs[ubatch_id]
        )
    assert view_ptrs == reserved_ptrs

    monkeypatch.setattr(workspace_module, "dbo_current_ubatch_id", lambda: 0)
    with pytest.raises(AssertionError, match="Workspace is locked"):
        manager.get_simultaneous(((1024,), torch.uint8))
