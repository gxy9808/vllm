# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import torch

from vllm.v1.spec_decode.step3p5 import Step3p5MTPProposer


def test_step_mtp_local_argmax_receives_spec_step_idx():
    hidden_states = torch.ones(2, 3)
    expected = torch.tensor([4, 5])
    model = SimpleNamespace(get_top_tokens=Mock(return_value=expected))
    proposer = SimpleNamespace(
        _enable_probabilistic_draft_probs=False,
        use_local_argmax_reduction=True,
        model=model,
    )

    token_ids, draft_probs = Step3p5MTPProposer._sample_draft_tokens_for_step(
        proposer,
        hidden_states,
        SimpleNamespace(all_greedy=True),
        spec_step_idx=2,
    )

    assert token_ids is expected
    assert draft_probs is None
    model.get_top_tokens.assert_called_once_with(
        hidden_states,
        spec_step_idx=2,
    )
