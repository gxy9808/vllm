# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.spec_decode.autoregressive import speculator as spec_module
from vllm.v1.worker.gpu.spec_decode.autoregressive.speculator import (
    AutoRegressiveSpeculator,
)
from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator


class _TestSpeculator(AutoRegressiveSpeculator):
    def load_draft_model(self, target_model, target_attn_layer_names):
        raise NotImplementedError


class _TestBaseSpeculator(DraftModelSpeculator):
    def init_cudagraph_manager(self, cudagraph_mode):
        raise NotImplementedError

    def capture(self):
        raise NotImplementedError

    def load_draft_model(self, target_model, target_attn_layer_names):
        raise NotImplementedError

    def propose(self, *args, **kwargs):
        raise NotImplementedError


class _DraftModel(torch.nn.Module):
    def __init__(self, output: torch.Tensor | tuple[torch.Tensor, torch.Tensor]):
        super().__init__()
        self.output = output

    def forward(self, **kwargs):
        return self.output


def _make_speculator(
    monkeypatch,
    output: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
) -> _TestSpeculator:
    monkeypatch.setattr(
        spec_module,
        "set_forward_context",
        lambda *args, **kwargs: nullcontext(),
    )

    speculator = object.__new__(_TestSpeculator)
    speculator.supports_mm_inputs = False
    speculator.vllm_config = None
    speculator.input_buffers = SimpleNamespace(
        input_ids=torch.arange(4),
        positions=torch.arange(4),
    )
    speculator.hidden_states = torch.zeros(4, 3)
    speculator.model = _DraftModel(output)
    return speculator


def test_run_model_unpacks_tuple_return_for_mtp(monkeypatch):
    logits_hidden = torch.full((4, 3), 1.0)
    feedback_hidden = torch.full((4, 3), 2.0)
    speculator = _make_speculator(monkeypatch, (logits_hidden, feedback_hidden))

    actual_logits_hidden, actual_feedback_hidden = speculator._run_model(
        4,
        attn_metadata=None,
        slot_mappings=None,
        num_tokens_across_dp=None,
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
    )

    assert actual_logits_hidden is logits_hidden
    assert actual_feedback_hidden is feedback_hidden


def test_run_model_reuses_tensor_return_for_mtp(monkeypatch):
    hidden = torch.full((4, 3), 1.0)
    speculator = _make_speculator(monkeypatch, hidden)

    actual_logits_hidden, actual_feedback_hidden = speculator._run_model(
        4,
        attn_metadata=None,
        slot_mappings=None,
        num_tokens_across_dp=None,
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
    )

    assert actual_logits_hidden is hidden
    assert actual_feedback_hidden is hidden


def test_probabilistic_draft_buffer_uses_effective_vocab_size():
    draft_model_config = SimpleNamespace(
        get_hidden_size=lambda: 4,
        get_vocab_size=lambda: 8,
        get_valid_vocab_size=lambda: 5,
        hf_config=SimpleNamespace(hc_mult=1),
    )
    vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            method="mtp",
            num_speculative_tokens=2,
            draft_model_config=draft_model_config,
            draft_sample_method="probabilistic",
            use_local_argmax_reduction=False,
        ),
        scheduler_config=SimpleNamespace(
            max_num_seqs=3,
            max_num_batched_tokens=8,
        ),
        model_config=SimpleNamespace(
            max_model_len=16,
            dtype=torch.float32,
            use_fp64_gumbel=False,
        ),
        parallel_config=SimpleNamespace(
            data_parallel_size=1,
            data_parallel_rank=0,
        ),
    )

    speculator = _TestBaseSpeculator(vllm_config, torch.device("cpu"))

    assert speculator.vocab_size == 5
    assert speculator.draft_logits is not None
    assert speculator.draft_logits.shape == (3, 2, 5)


def test_probabilistic_draft_rejects_mismatched_logits_buffer():
    speculator = object.__new__(_TestBaseSpeculator)
    speculator.model = SimpleNamespace(
        compute_logits=lambda hidden_states: hidden_states.new_zeros(2, 5)
    )

    with pytest.raises(ValueError, match="must match computed logits"):
        speculator.sample_draft(
            hidden_states=torch.zeros(2, 4),
            positions=torch.zeros(2, dtype=torch.int64),
            idx_mapping=torch.arange(2, dtype=torch.int32),
            temperature=torch.ones(2),
            seeds=torch.zeros(2, dtype=torch.int64),
            draft_step=torch.tensor(0),
            draft_logits=torch.zeros(2, 2, 8),
        )
