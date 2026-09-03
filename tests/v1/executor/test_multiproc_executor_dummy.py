# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock

import pytest

from vllm.v1.executor.multiproc_executor import MultiprocExecutor


@pytest.mark.skip_global_cleanup
def test_execute_dummy_batch_waits_for_all_workers() -> None:
    executor = object.__new__(MultiprocExecutor)
    executor.collective_rpc = MagicMock()

    executor.execute_dummy_batch()

    executor.collective_rpc.assert_called_once_with("execute_dummy_batch")
