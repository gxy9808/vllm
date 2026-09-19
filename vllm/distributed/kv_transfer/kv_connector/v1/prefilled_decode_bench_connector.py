# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PreFilledDecodeBenchConnector: A KV Connector for pure decode performance testing.

Unlike DecodeBenchConnector which fills KV cache per-request (causing CPU
bottlenecks on models with many layers like Step4), this connector fills the
ENTIRE KV cache pool once during initialization. Subsequent requests simply
claim pre-filled blocks without any per-request fill overhead.

It reuses DecodeBenchConnector's scheduler logic (which is proven to correctly
skip prefill) and only replaces the worker's fill strategy.

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
from vllm.distributed.kv_transfer.kv_connector.v1.decode_bench_connector import (
    DecodeBenchConnector,
    DecodeBenchConnectorScheduler,
)
from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv
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


class PreFilledDecodeBenchConnector(DecodeBenchConnector):
    """A KV Connector that pre-fills the entire KV cache pool at init.

    Inherits all scheduler logic from DecodeBenchConnector (proven to correctly
    skip prefill). Only overrides the worker to pre-fill the entire pool once
    at init, instead of per-request serial filling.
    """

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig",
    ):
        super().__init__(vllm_config, role, kv_cache_config)

        if role == KVConnectorRole.WORKER:
            # Replace the per-request worker with our pre-fill worker
            self.connector_worker = _PreFilledWorker(vllm_config)

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        assert self.connector_worker is not None
        self.connector_worker.register_kv_caches(kv_caches)


class _PreFilledWorker:
    """Worker that pre-fills the entire KV cache pool at registration time.

    After init, start_fill_kv is a no-op: blocks are already filled.
    """

    def __init__(self, vllm_config: "VllmConfig"):
        self.vllm_config = vllm_config
        self.block_size = vllm_config.cache_config.block_size

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

    def start_fill_kv(self, metadata):
        # No-op: entire pool is already filled at init
        logger.debug("PreFilledDecodeBenchConnector: start_fill_kv called (no-op, pool pre-filled)")
