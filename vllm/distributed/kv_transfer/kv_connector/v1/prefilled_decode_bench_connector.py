# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PreFilledDecodeBenchConnector: A KV Connector for pure decode performance testing.

Unlike DecodeBenchConnector which fills KV cache per-request (causing CPU
bottlenecks on models with many layers like Step4), this connector fills the
ENTIRE KV cache pool once during initialization. Subsequent requests simply
claim pre-filled blocks without any per-request fill overhead.

Usage:
    vllm serve <model> --kv-transfer-config '{
        "kv_connector": "PreFilledDecodeBenchConnector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {
            "fill_mean": 0.015,
            "fill_std": 0.0
        }
    }'

    Then run your benchmark:
    vllm bench serve --base-url http://127.0.0.1:8000 --model <model> \\
        --dataset-name random --random-input-len 40000 \\
        --random-output-len 100 --max-concurrency 10
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from vllm.distributed.kv_transfer.kv_connector.v1 import (
    KVConnectorBase_V1,
    KVConnectorRole,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorMetadata,
    SupportsHMA,
)
from vllm.logger import init_logger
from vllm.v1.attention.backend import AttentionMetadata

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)

_FLOAT8_DTYPES = {
    dtype
    for dtype_name in (
        "float8_e4m3fn",
        "float8_e5m2",
        "float8_e4m3fnuz",
        "float8_e5m2fnuz",
    )
    if (dtype := getattr(torch, dtype_name, None)) is not None
}


@dataclass
class PreFilledDecodeBenchConnectorMetadata(KVConnectorMetadata):
    """Metadata for PreFilledDecodeBenchConnector (empty, no per-req fill)."""
    pass


class PreFilledDecodeBenchConnector(KVConnectorBase_V1, SupportsHMA):
    """A KV Connector that pre-fills the entire KV cache pool at init.

    Fills all KV caches with dummy values once during register_kv_caches.
    Subsequent requests claim pre-filled blocks without any fill overhead.
    """

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig",
    ):
        super().__init__(vllm_config, role, kv_cache_config)

        self.connector_scheduler = None
        self.connector_worker = None

        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler = _Scheduler(vllm_config)
        elif role == KVConnectorRole.WORKER:
            self.connector_worker = _Worker(vllm_config)

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        assert self.connector_worker is not None
        self.connector_worker.register_kv_caches(kv_caches)

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        pass

    def wait_for_layer_load(self, layer_name: str) -> None:
        pass

    def save_kv_layer(self, layer_name: str, kv_layer: torch.Tensor,
                      attn_metadata: AttentionMetadata, **kwargs: Any) -> None:
        pass

    def wait_for_save(self):
        pass

    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.get_num_new_matched_tokens(
            request, num_computed_tokens
        )

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        assert self.connector_scheduler is not None
        return self.connector_scheduler.update_state_after_alloc(
            request, blocks, num_external_tokens
        )

    def build_connector_meta(
        self, scheduler_output: "SchedulerOutput"
    ) -> KVConnectorMetadata:
        return PreFilledDecodeBenchConnectorMetadata()

    def request_finished(
        self, request: "Request", block_ids: list[int]
    ) -> tuple[bool, dict[str, Any] | None]:
        assert self.connector_scheduler is not None
        self.connector_scheduler.request_finished(request)
        return False, None

    def request_finished_all_groups(
        self, request: "Request", block_ids: tuple[list[int], ...]
    ) -> tuple[bool, dict[str, Any] | None]:
        assert self.connector_scheduler is not None
        self.connector_scheduler.request_finished(request)
        return False, None


class _Scheduler:
    def __init__(self, vllm_config: "VllmConfig"):
        self._filled_requests: set[str] = set()

    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int
    ) -> tuple[int, bool]:
        req_id = request.request_id
        if req_id in self._filled_requests:
            return 0, False
        num_uncomputed = request.num_tokens - num_computed_tokens
        num_to_fill = max(0, num_uncomputed - 1)
        return num_to_fill, False

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        if num_external_tokens > 0:
            self._filled_requests.add(request.request_id)

    def request_finished(self, request: "Request"):
        self._filled_requests.discard(request.request_id)


class _Worker:
    def __init__(self, vllm_config: "VllmConfig"):
        kv_transfer_config = vllm_config.kv_transfer_config
        assert kv_transfer_config is not None
        self.fill_mean = kv_transfer_config.get_from_extra_config("fill_mean", 0.015)
        self.fill_std = kv_transfer_config.get_from_extra_config("fill_std", 0.0)
        self.kv_caches: dict[str, torch.Tensor] | None = None
        self._pool_filled = False

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        self.kv_caches = kv_caches
        if not self._pool_filled:
            self._fill_entire_pool()

    def _make_fill_values(self, shape, dtype, device):
        gen_dtype = torch.float16 if dtype in _FLOAT8_DTYPES else dtype
        if self.fill_std > 0:
            values = torch.normal(
                mean=self.fill_mean, std=self.fill_std,
                size=shape, dtype=gen_dtype, device=device,
            )
        else:
            values = torch.full(shape, self.fill_mean, dtype=gen_dtype, device=device)
        return values.to(dtype) if gen_dtype != dtype else values

    def _fill_entire_pool(self):
        assert self.kv_caches is not None
        logger.info(
            "PreFilledDecodeBenchConnector: Pre-filling entire KV cache pool "
            "(%d layers, mean=%.6f, std=%.6f)",
            len(self.kv_caches), self.fill_mean, self.fill_std,
        )
        for layer_name, kv_cache in self.kv_caches.items():
            if isinstance(kv_cache, torch.Tensor):
                dtype = kv_cache.dtype
                if dtype in _FLOAT8_DTYPES:
                    fill = self._make_fill_values(kv_cache.shape, dtype, kv_cache.device)
                    kv_cache.view(torch.uint8).copy_(fill.view(torch.uint8))
                else:
                    if self.fill_std > 0:
                        kv_cache.normal_(mean=self.fill_mean, std=self.fill_std)
                    else:
                        kv_cache.fill_(self.fill_mean)
            elif isinstance(kv_cache, (list, tuple)):
                for t in kv_cache:
                    if isinstance(t, torch.Tensor):
                        if t.dtype in _FLOAT8_DTYPES:
                            fill = self._make_fill_values(t.shape, t.dtype, t.device)
                            t.view(torch.uint8).copy_(fill.view(torch.uint8))
                        elif self.fill_std > 0:
                            t.normal_(mean=self.fill_mean, std=self.fill_std)
                        else:
                            t.fill_(self.fill_mean)
        self._pool_filled = True
        logger.info("PreFilledDecodeBenchConnector: KV cache pool pre-fill complete.")
