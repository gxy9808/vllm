# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest import mock

import pytest
import torch

import vllm.v1.spec_decode.llm_base_proposer as proposer_module
from vllm.config import CUDAGraphMode
from vllm.forward_context import BatchDescriptor
from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


class _Dispatcher:
    @staticmethod
    def dispatch(num_tokens, **kwargs):
        del kwargs
        return CUDAGraphMode.NONE, BatchDescriptor(num_tokens=num_tokens)


def _make_mtp_proposer(*, is_dense_mtp: bool) -> SpecDecodeBaseProposer:
    proposer = object.__new__(SpecDecodeBaseProposer)
    proposer.method = "mtp"
    proposer.model = SimpleNamespace(is_dense_mtp=is_dense_mtp)
    proposer.dp_rank = 0
    proposer.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            data_parallel_size=2,
            data_parallel_rank=0,
        )
    )
    proposer.cudagraph_dispatcher = _Dispatcher()
    return proposer


def test_dense_mtp_uses_local_shape_without_draft_all_reduce(monkeypatch):
    proposer = _make_mtp_proposer(is_dense_mtp=True)

    def unexpected_dp_sync(*args, **kwargs):
        raise AssertionError("dense MTP draft must not synchronize across DP")

    monkeypatch.setattr(
        proposer_module,
        "coordinate_batch_across_dp",
        unexpected_dp_sync,
    )

    cudagraph_mode, num_tokens_padded, num_tokens_across_dp = (
        proposer._determine_batch_execution_and_padding(
            num_tokens=5,
            use_cudagraphs=False,
        )
    )

    assert cudagraph_mode == CUDAGraphMode.NONE
    assert num_tokens_padded == 5
    assert num_tokens_across_dp is not None
    assert num_tokens_across_dp.tolist() == [5, 5]


def test_moe_mtp_keeps_draft_dp_coordination(monkeypatch):
    proposer = _make_mtp_proposer(is_dense_mtp=False)
    coordinate = mock.Mock(
        return_value=(
            False,
            torch.tensor([5, 5], dtype=torch.int32),
            CUDAGraphMode.NONE.value,
        )
    )
    monkeypatch.setattr(
        proposer_module,
        "coordinate_batch_across_dp",
        coordinate,
    )

    proposer._determine_batch_execution_and_padding(
        num_tokens=5,
        use_cudagraphs=False,
    )

    coordinate.assert_called_once()
