# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Parallelism Wrappers

Wraps any attention backend with a parallelism strategy. Not a standalone
backend — compose around a real backend (VANILLA/TRTLLM/FA4/CUTEDSL).

"""

from typing import TYPE_CHECKING, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F

if TYPE_CHECKING:
    from ..mapping import VisualGenMapping

from tensorrt_llm._torch.distributed import all_to_all_4d, all_to_all_5d

from ...attention_backend.interface import PredefinedAttentionMask
from .interface import AttentionBackend, AttentionTensorLayout

_flash_attn_combine_import_error = None
try:
    from flash_attn.cute.interface import flash_attn_combine as _flash_attn_combine
except (ImportError, OSError) as e:
    _flash_attn_combine = None
    _flash_attn_combine_import_error = e


class UlyssesAttention(AttentionBackend):
    """
    Ulysses Sequence Parallelism wrapper.

    Wraps any attention backend with sequence parallelism via all-to-all.
    Not a standalone backend -- compose around a real backend (VANILLA/TRTLLM).
    Fully transparent to backend-specific kwargs: everything in ``**kwargs``
    is forwarded to the inner backend unchanged (except ``seq_len`` which is
    overridden with the post-all-to-all value).

    Architecture:
        Input:  [B, S/P, H, D] (sequence sharded across P processes)
        Step 1: All-to-All → [B, S, H/P, D] (gather sequence, shard heads)
        Step 2: Compute attention with wrapped backend (VANILLA or TRTLLM)
        Step 3: All-to-All → [B, S/P, H, D] (restore sequence sharding)
        Output: [B, S/P, H, D] (sequence sharded)

    Two modes (auto-selected via ``inner_backend.support_fused_qkv()``):
    - Unfused: 3 separate all-to-all for Q/K/V + 1 for output (4 collectives)
    - Fused: stacks Q/K/V into [B, S/P, 3, H, D], 1 fused 5D all-to-all
      + 1 for output (2 collectives total)
    """

    def __init__(
        self,
        inner_backend: AttentionBackend,
        process_group: torch.distributed.ProcessGroup,
    ):
        self.inner_backend = inner_backend
        self.process_group = process_group
        self._preferred_layout = AttentionTensorLayout.NHD

        self.head_dim = inner_backend.head_dim
        self.sharded_num_heads = inner_backend.num_heads
        self.sharded_num_kv_heads = getattr(inner_backend, "num_kv_heads", self.sharded_num_heads)

        self.world_size = torch.distributed.get_world_size(group=process_group)

        self.num_heads = self.sharded_num_heads * self.world_size
        self.num_kv_heads = self.sharded_num_kv_heads * self.world_size

        # Try to use UserBuffers all-to-all; fall back to NCCL if unavailable.
        self._ub_a2a = None
        import os
        if os.environ.get("TRTLLM_FORCE_NCCL_ALLREDUCE", "0") == "0":
            try:
                from ..nccl_ub_reg import UBAllToAll, _ub_available
                if _ub_available():
                    # Max elements: conservative upper bound of a single QKV slice
                    max_elems = 4096 * self.num_heads * self.head_dim
                    self._ub_a2a = UBAllToAll(max_elems, dtype=torch.bfloat16)
            except Exception:
                pass  # UB not available; fall through to NCCL

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass with Ulysses sequence parallelism.

        q/k/v: [B, S/P, H, D] each.  All other arguments are forwarded
        transparently to the inner backend via ``**kwargs``.
        """
        # Catches upstream floor-division bugs (e.g. num_heads // ulysses_size when
        # num_heads % ulysses_size != 0) before they corrupt the all-to-all.
        if q.shape[2] % self.world_size != 0:
            raise ValueError(
                f"UlyssesAttention: q num_heads ({q.shape[2]}) must be divisible "
                f"by world_size ({self.world_size})."
            )
        if k.shape[2] % self.world_size != 0:
            raise ValueError(
                f"UlyssesAttention: k num_kv_heads ({k.shape[2]}) must be divisible "
                f"by world_size ({self.world_size})."
            )

        if self.inner_backend.support_fused_qkv():
            return self._forward_fused(q, k, v, **kwargs)
        return self._forward_unfused(q, k, v, **kwargs)

    def _forward_fused(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        batch_size = q.shape[0]
        qkv = torch.stack([q, k, v], dim=2)
        if self._ub_a2a is not None:
            qkv = self._ub_a2a(qkv, scatter_dim=3, gather_dim=1, process_group=self.process_group)
        else:
            qkv = all_to_all_5d(qkv, scatter_dim=3, gather_dim=1, process_group=self.process_group)

        B, seq_len, _, Hp, D = qkv.shape

        # Caller passed pre-A2A (sharded) seq_len; the inner backend
        # reshapes by it, so hand it the post-A2A length instead.
        kwargs["batch_size"] = batch_size
        kwargs["seq_len"] = seq_len
        kwargs["seq_len_kv"] = seq_len

        output = self.inner_backend.forward(q=qkv, k=None, v=None, **kwargs)

        return self._output_a2a(output, batch_size, seq_len)

    def _forward_unfused(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        batch_size = q.shape[0]
        if self._ub_a2a is not None:
            q = self._ub_a2a(q, scatter_dim=2, gather_dim=1, process_group=self.process_group)
            k = self._ub_a2a(k, scatter_dim=2, gather_dim=1, process_group=self.process_group)
            v = self._ub_a2a(v, scatter_dim=2, gather_dim=1, process_group=self.process_group)
        else:
            q = all_to_all_4d(q, scatter_dim=2, gather_dim=1, process_group=self.process_group)
            k = all_to_all_4d(k, scatter_dim=2, gather_dim=1, process_group=self.process_group)
            v = all_to_all_4d(v, scatter_dim=2, gather_dim=1, process_group=self.process_group)

        seq_len_full = q.shape[1]
        kv_seq_len_full = k.shape[1]

        if self.inner_backend.preferred_layout == AttentionTensorLayout.HND:
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

        # Caller passed pre-A2A (sharded) seq_lens; hand the inner
        # backend the post-A2A lengths instead.
        kwargs["batch_size"] = batch_size
        kwargs["seq_len"] = seq_len_full
        kwargs["seq_len_kv"] = kv_seq_len_full

        output = self.inner_backend.forward(q=q, k=k, v=v, **kwargs)

        return self._output_a2a(output, batch_size, seq_len_full)

    def _output_a2a(
        self,
        output: torch.Tensor,
        batch_size: int,
        seq_len_full: int,
    ) -> torch.Tensor:
        """Reverse all-to-all: [B, S, H/P, D] → [B, S/P, H, D]"""
        inner_layout = self.inner_backend.preferred_layout

        if inner_layout == AttentionTensorLayout.HND:
            output = output.transpose(1, 2).contiguous()
        else:
            if output.dim() == 3:
                output = output.view(
                    batch_size, seq_len_full, self.sharded_num_heads, self.head_dim
                )
            output = output.contiguous()

        if self._ub_a2a is not None:
            output = self._ub_a2a(output, scatter_dim=1, gather_dim=2, process_group=self.process_group)
        else:
            output = all_to_all_4d(output, scatter_dim=1, gather_dim=2, process_group=self.process_group)

        return output

    @property
    def preferred_layout(self) -> AttentionTensorLayout:
        """Preferred tensor layout: [B, S, H, D]"""
        return self._preferred_layout

    @classmethod
    def support_fused_qkv(cls) -> bool:
        return True


class Attention2DAttention(AttentionBackend):
    """
    Attention2D Context Parallelism wrapper for video-generation inference.

    Based on:
        "Attention2D: Communication Efficient Distributed Self-Attention Mechanism"
        https://arxiv.org/pdf/2503.15758

    The original paper targets LLM training with causal attention.  This is a
    simplified adaptation for video generation inference with full (non-causal)
    attention.

    Motivation vs. Ulysses and Ring attention
    -----------------------------------------
    *vs. Ulysses*: Ulysses (head-sharding) requires the parallelism degree to divide
    the model's head count (e.g. for WAN: degree ≤ 12 and must divide 12).
    Attention2D removes this constraint entirely — any ``row_size × col_size`` mesh
    is valid regardless of head count.  Attention2D can also be composed with Ulysses
    (head-sharding across a separate process group) to combine both parallelism axes.

    *vs. Ring attention*: Ring attention is the closest algorithmic alternative — both
    distribute sequence across GPUs without a head-count constraint.  Attention2D
    scales better: for a symmetric mesh (``row_size ≈ col_size ≈ √P``), communication
    volume scales as ``O(N / √P)`` where ``N`` is the sequence length and ``P`` is the
    total number of GPUs, compared to ``O(N)`` for ring attention.

    Mesh layout
    -----------
    Ranks are arranged in a 2-D logical mesh of shape ``[row_size, col_size]``
    (total parallelism degree = ``P = row_size * col_size``).  Each rank holds a
    ``[B, S/P, H, D]`` shard of Q, K, and V.

    Example for ``row_size=2, col_size=3`` (6 ranks total)::

                   col group (K/V all-gather)
                     ↓        ↓        ↓
                   col 0    col 1    col 2
        row 0  [  rank 0 | rank 1 | rank 2  ]  ← row group (Q all-gather)
        row 1  [  rank 3 | rank 4 | rank 5  ]  ← row group (Q all-gather)

    Ranks in the same **row** share a ``row_process_group`` and all-gather Q.
    Ranks in the same **column** share a ``col_process_group`` and all-gather K/V.

    Architecture:
        Input:   [B, S/P, H, D]  (sequence sharded across P = row_size × col_size ranks)
        Step 1:  Q all-gather within row group:        [B, S/P, H, D] → [B, S/col_size, H, D]
        Step 2:  K/V fused all-gather within col group [B, S/P, H, D] → [B, S/row_size, H, D]
                   (K and V packed into [2, B, S/P, H, D] before the gather,
                    halving NCCL launch overhead vs. two separate collectives)
        Step 3:  Local attention with inner backend:
                   Q [B, S/col_size, H, D] × K,V [B, S/row_size, H, D]
                   → output [B, S/col_size, H, D] + LSE [B, H, S/col_size]
        Step 4:  Reduce-scatter output within row group, split into:
                   all_to_all_single to exchange partial outputs and LSEs, then
                   LSE-weighted combine via flash_attn_combine
                   → [B, S/P, H, D]  (fully reduced, matching input layout)
        Output:  [B, S/P, H, D]

    Supported inner backends
    ------------------------
    The inner backend must support LSE output (``support_lse() -> True``) — required
    for the reduce-scatter combine step.  Currently the FA4 and CUTEDSL
    backends meet this requirement.

    Note: ``AttentionTensorLayout.NHD`` and ``AttentionTensorLayout.HND`` are both
    handled transparently; transposition is applied before the inner forward and
    reversed afterward.

    Note: ``support_fused_qkv()`` is *not* required — fused QKV would not reduce
    communication costs because Q and K/V are gathered over different process
    groups and cannot be merged into a single collective.

    Constraints
    -----------
    * Only ``PredefinedAttentionMask.FULL`` (or ``None``) is supported.
    * ``flash_attn_combine`` (JIT CUDA kernel) must be importable at
      construction time; the constructor raises ``ImportError`` otherwise.
    * The ``_combine`` step is wrapped in ``@torch.compiler.disable`` because
      the JIT kernel accesses raw data pointers that are incompatible with
      ``FakeTensor`` tracing during ``torch.compile``.
    """

    def __init__(
        self,
        inner_backend: AttentionBackend,
        row_process_group: torch.distributed.ProcessGroup,
        col_process_group: torch.distributed.ProcessGroup,
    ):
        self.inner_backend = inner_backend
        self.row_process_group = row_process_group
        self.col_process_group = col_process_group

        self.row_group_size = torch.distributed.get_world_size(group=row_process_group)
        self.col_group_size = torch.distributed.get_world_size(group=col_process_group)
        # Always NHD: all-gather kernels operate on [B, S/P, H, D]. Any HND conversion
        # needed by the inner backend is handled internally in forward.
        self._preferred_layout = AttentionTensorLayout.NHD

        if _flash_attn_combine is None:
            raise ImportError(
                "flash_attn_combine is not available. Attention2DAttention requires "
                "the Flash Attention JIT kernels to be built. "
                f"Import error: {_flash_attn_combine_import_error}"
            ) from _flash_attn_combine_import_error

        if not inner_backend.support_lse():
            raise RuntimeError(
                f"{type(inner_backend).__name__} does not support LSE output "
                "(support_lse() returned False). Attention2DAttention requires "
                "the inner backend to support LSE."
            )

        for attr in ("head_dim", "num_heads"):
            if not hasattr(inner_backend, attr):
                raise RuntimeError(
                    f"{type(inner_backend).__name__} is missing required attribute '{attr}'. "
                    "Attention2DAttention requires the inner backend to expose 'head_dim' and "
                    "'num_heads' as instance attributes."
                )
        self.head_dim = inner_backend.head_dim
        self.num_heads = inner_backend.num_heads
        self._inner_layout = inner_backend.preferred_layout
        if self._inner_layout not in (AttentionTensorLayout.NHD, AttentionTensorLayout.HND):
            raise NotImplementedError(
                f"{type(inner_backend).__name__} uses unsupported layout: {self._inner_layout}"
            )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass with Attention2D sequence parallelism.

        q/k/v: [B, S/P, H, D] each.
        """
        B, shard_seq, H, D = q.shape
        attention_mask = kwargs.get("attention_mask", None)

        if attention_mask is not None and attention_mask != PredefinedAttentionMask.FULL:
            raise ValueError(
                f"Attention2DAttention only supports FULL attention mask, got {attention_mask}."
            )

        if self.row_group_size > 1:
            # All-gather q within row_process_group using a single flat buffer.
            # [B, S/P, H, D] → [row_group_size, B, S/P, H, D] → [B, S/col_group_size, H, D]
            q_recv = q.new_empty(self.row_group_size, B, shard_seq, H, D)
            torch.distributed.all_gather_into_tensor(
                q_recv.view(-1), q.contiguous().view(-1), group=self.row_process_group
            )
            q = q_recv.permute(1, 0, 2, 3, 4).reshape(B, self.row_group_size * shard_seq, H, D)

        if self.col_group_size > 1:
            # Fuse K and V into a single all-gather to reduce NCCL launch overhead.
            # [2, B, S/P, H, D] → [col_group_size, 2, B, S/P, H, D] → split back to K, V
            kv_send = k.new_empty(2, B, shard_seq, H, D)
            kv_send[0].copy_(k)
            kv_send[1].copy_(v)
            kv_recv = k.new_empty(self.col_group_size, 2, B, shard_seq, H, D)
            torch.distributed.all_gather_into_tensor(
                kv_recv.view(-1), kv_send.view(-1), group=self.col_process_group
            )
            k = (
                kv_recv[:, 0]
                .permute(1, 0, 2, 3, 4)
                .reshape(B, self.col_group_size * shard_seq, H, D)
            )
            v = (
                kv_recv[:, 1]
                .permute(1, 0, 2, 3, 4)
                .reshape(B, self.col_group_size * shard_seq, H, D)
            )

        seq_len = q.shape[1]

        if self._inner_layout == AttentionTensorLayout.HND:
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

        output, lse = self.inner_backend.forward_with_lse(q=q, k=k, v=v, **kwargs)

        if self._inner_layout == AttentionTensorLayout.HND:
            output = output.transpose(1, 2).contiguous()
        else:
            if output.dim() == 3:
                output = output.view(B, seq_len, self.num_heads, self.head_dim)
            output = output.contiguous()

        if self.row_group_size > 1:
            # Reduce-scatter output+lse along sequence within row_process_group,
            # implemented as all_to_all_single + local LSE-based reduce.
            N = self.row_group_size
            B, seq_row, H, D = output.shape  # seq_row = S/col_group_size = N * (S/P)
            shard_seq = seq_row // N

            # output: [B, seq_row, H, D] → [N, B, shard_seq, H, D] (grouped by dest rank)
            o_send = output.view(B, N, shard_seq, H, D).permute(1, 0, 2, 3, 4).contiguous()
            o_recv = torch.empty_like(o_send)
            torch.distributed.all_to_all_single(o_recv, o_send, group=self.row_process_group)
            # o_recv: [N, B, shard_seq, H, D] — already stacked by source rank

            # lse: [B, H, seq_row] → [N, B, H, shard_seq] (grouped by dest rank)
            lse_send = lse.view(B, H, N, shard_seq).permute(2, 0, 1, 3).contiguous()
            lse_recv = torch.empty_like(lse_send)
            torch.distributed.all_to_all_single(lse_recv, lse_send, group=self.row_process_group)
            # lse_recv: [N, B, H, shard_seq] — already stacked by source rank

            # flash_attn_combine expects lse as [N, B, S/P, H] with stride(-2)==1;
            # do not call .contiguous() after permute as it would reset the strides.
            lse_recv = lse_recv.permute(0, 1, 3, 2)  # [N, B, shard_seq, H]
            output, _ = self._combine(o_recv, lse_recv, output.dtype)

        return output

    @torch.compiler.disable
    def _combine(
        self,
        o_partial: torch.Tensor,
        lse_partial: torch.Tensor,
        out_dtype: torch.dtype,
    ):
        """Combine partial attention outputs via LSE reduction.

        Isolated under @torch.compiler.disable because _flash_attn_combine is a
        JIT CUDA kernel that accesses raw data pointers, incompatible with
        FakeTensor tracing during torch.compile.
        """
        return _flash_attn_combine(
            o_partial.float().contiguous(), lse_partial, out_dtype=out_dtype, return_lse=False
        )

    @property
    def preferred_layout(self) -> AttentionTensorLayout:
        return self._preferred_layout

    @classmethod
    def support_fused_qkv(cls) -> bool:
        # FlashAttn4 (the only backend currently supporting the required LSE output)
        # does not support fused QKV. Even if it did, fused QKV would not reduce
        # communication costs in Attention2D since Q and K/V are gathered over
        # different process groups and cannot be fused into a single collective.
        # If a future backend supports both LSE and fused QKV with a faster kernel,
        # add fused QKV support.
        return False

    @classmethod
    def support_lse(cls) -> bool:
        return False


class RingAttention(AttentionBackend):
    """Ring sequence parallelism around an LSE-capable attention backend."""

    def __init__(
        self,
        inner_backend: AttentionBackend,
        process_group: dist.ProcessGroup,
    ):
        # Invariant: only instantiated when ring_size > 1 (see attention.py),
        # so distributed must be initialized and the group must be non-trivial.
        if not type(inner_backend).support_lse():
            raise ValueError(
                f"RingAttention requires an LSE-capable inner backend (FA4); "
                f"got {type(inner_backend).__name__}"
            )

        # Required attributes for buffer allocation in _ensure_buffers.
        for attr in ("head_dim", "num_heads"):
            if not hasattr(inner_backend, attr):
                raise RuntimeError(
                    f"{type(inner_backend).__name__} is missing required attribute "
                    f"'{attr}'. RingAttention needs the inner backend to expose "
                    "'head_dim' and 'num_heads' as instance attributes."
                )

        # Ring's _ensure_buffers / _update_out_and_lse assume NHD ([B, S, H, D]).
        # No transpose is applied around the inner forward, so an HND backend would
        # silently produce wrong results.
        if inner_backend.preferred_layout != AttentionTensorLayout.NHD:
            raise NotImplementedError(
                f"RingAttention requires an NHD inner backend; "
                f"{type(inner_backend).__name__} prefers {inner_backend.preferred_layout}."
            )

        self.inner = inner_backend
        self.pg = process_group
        self.world_size = dist.get_world_size(group=process_group)
        self.num_heads = inner_backend.num_heads
        self.num_kv_heads = getattr(inner_backend, "num_kv_heads", self.num_heads)
        self.head_dim = inner_backend.head_dim
        self._preferred_layout = AttentionTensorLayout.NHD

        # P2P ring topology cached at construction time (avoids per-step lookups).
        ring_rank = dist.get_rank(group=process_group)
        self._send_rank = dist.get_global_rank(process_group, (ring_rank + 1) % self.world_size)
        self._recv_rank = dist.get_global_rank(process_group, (ring_rank - 1) % self.world_size)
        self._send_first = ring_rank % 2 == 0
        self._p2p_reqs: list = []

        self._buf_key = None
        self._kv_bufs = None
        self._out_buf = None
        self._lse_buf = None

        # UB P2P buffers for Ring (registered once, reused across steps)
        self._ub_send_buf = None
        self._ub_recv_buf = None
        self._ub_p2p_ready = False
        import os
        ring_transport = os.environ.get("TRTLLM_RING_TRANSPORT", "auto")
        if ring_transport in ("ub", "auto") and os.environ.get("TRTLLM_FORCE_NCCL_ALLREDUCE", "0") == "0":
            try:
                from ..nccl_ub_reg import _ub_available
                from tensorrt_llm.bindings.internal.userbuffers import (  # type: ignore
                    ub_allocate, userbuffers_send, userbuffers_recv,
                )
                if _ub_available():
                    self._ub_p2p_ready = True
                    self._ub_allocate = ub_allocate
                    self._ub_send_fn = userbuffers_send
                    self._ub_recv_fn = userbuffers_recv
            except Exception:
                pass

        # NIXL P2P for Ring — used when TRTLLM_RING_TRANSPORT=nixl, or auto with no UB
        self._nixl_ready = False
        self._nixl_agent = None
        self._nixl_prev_agent_name: str | None = None
        self._nixl_prev_base_ptr: int | None = None
        self._nixl_prev_device: int | None = None
        self._nixl_kv_bufs_registered = False
        self._nixl_status = None
        if ring_transport == "nixl" or (ring_transport == "auto" and not self._ub_p2p_ready):
            self._init_nixl_ring()

    def _ring_send_recv(self, send: torch.Tensor, recv: torch.Tensor, cur: int = 0) -> None:
        """Post a non-blocking neighbor exchange.

        cur is the current ping-pong slot index (0 or 1), used by the NIXL path
        to compute the remote buffer offset. Even ranks send-then-recv in the
        NCCL path to avoid deadlock against odd ranks doing recv-then-send.
        """
        if self._nixl_ready:
            self._ring_send_recv_nixl(send, recv, cur)
            return
        if self._ub_p2p_ready:
            self._ring_send_recv_ub(send, recv)
            return
        if self._send_first:
            ops = [
                dist.P2POp(dist.isend, send, self._send_rank, group=self.pg),
                dist.P2POp(dist.irecv, recv, self._recv_rank, group=self.pg),
            ]
        else:
            ops = [
                dist.P2POp(dist.irecv, recv, self._recv_rank, group=self.pg),
                dist.P2POp(dist.isend, send, self._send_rank, group=self.pg),
            ]
        self._p2p_reqs = dist.batch_isend_irecv(ops)

    def _ring_send_recv_ub(self, send: torch.Tensor, recv: torch.Tensor) -> None:
        """UB-based P2P exchange: one-sided push + spin-wait recv."""
        nbytes = send.nbytes
        # Allocate UB buffers on first use or if size changes.
        if self._ub_send_buf is None or self._ub_send_buf.size < nbytes:
            self._ub_send_buf = self._ub_allocate(nbytes)
            self._ub_recv_buf = self._ub_allocate(nbytes)

        import ctypes
        # Copy send tensor into UB send buffer.
        send_c = send.contiguous()
        send_view = torch.frombuffer(
            (ctypes.c_byte * nbytes).from_address(self._ub_send_buf.addr),
            dtype=send.dtype,
        ).reshape(send.shape)
        send_view.copy_(send_c)

        stream_ptr = torch.cuda.current_stream().cuda_stream
        # Push data to peer's recv buffer; peer reads from its own UB recv buf.
        self._ub_send_fn(
            self._ub_send_buf.handle, 0,
            self._ub_recv_buf.handle, 0,
            nbytes, self._send_rank, stream_ptr,
        )
        # Wait for incoming data from the previous rank.
        self._ub_recv_fn(
            self._ub_send_buf.handle, 0,
            self._ub_recv_buf.handle, 0,
            nbytes, self._recv_rank, stream_ptr,
        )
        torch.cuda.synchronize()

        # Copy result out of UB recv buffer into recv tensor.
        recv_view = torch.frombuffer(
            (ctypes.c_byte * nbytes).from_address(self._ub_recv_buf.addr),
            dtype=recv.dtype,
        ).reshape(recv.shape)
        recv.copy_(recv_view)

    def _init_nixl_ring(self) -> None:
        """Initialize NIXL agent and exchange descriptors with ring neighbors."""
        try:
            from tensorrt_llm._torch.disaggregation.nixl._agent_py import NixlTransferAgent
        except ImportError:
            return
        try:
            global_rank = dist.get_rank()
            self._nixl_agent = NixlTransferAgent(
                name=f"ring_rank_{global_rank}", use_prog_thread=True
            )
            # Serialize local agent descriptor and exchange with all CP-group members.
            local_desc = self._nixl_agent.get_local_agent_desc()
            desc_tensor = torch.tensor(list(local_desc), dtype=torch.uint8).cuda()
            desc_len = torch.tensor([len(local_desc)], dtype=torch.int64).cuda()
            all_lens = [torch.zeros(1, dtype=torch.int64).cuda() for _ in range(self.world_size)]
            dist.all_gather(all_lens, desc_len, group=self.pg)
            max_len = int(max(int(l.item()) for l in all_lens))
            padded = torch.zeros(max_len, dtype=torch.uint8).cuda()
            padded[: len(local_desc)] = desc_tensor
            all_descs = [torch.zeros(max_len, dtype=torch.uint8).cuda() for _ in range(self.world_size)]
            dist.all_gather(all_descs, padded, group=self.pg)
            ring_rank = dist.get_rank(group=self.pg)
            prev_ring_rank = (ring_rank - 1) % self.world_size
            prev_global_rank = dist.get_global_rank(self.pg, prev_ring_rank)
            prev_len = int(all_lens[prev_ring_rank].item())
            prev_desc = bytes(all_descs[prev_ring_rank][:prev_len].cpu().tolist())
            self._nixl_prev_agent_name = f"ring_rank_{prev_global_rank}"
            self._nixl_agent.load_remote_agent(self._nixl_prev_agent_name, prev_desc)
            self._nixl_ready = True
        except Exception:
            self._nixl_agent = None

    def _nixl_register_kv_bufs(self) -> None:
        """Register kv_bufs VRAM region with NIXL and exchange base pointers."""
        from tensorrt_llm._torch.disaggregation.base.agent import RegMemoryDescs
        kv = self._kv_bufs
        device_id = torch.cuda.current_device()
        # One region covers both ping-pong slots (kv_bufs[0] and kv_bufs[1]).
        reg = RegMemoryDescs(type="VRAM", descs=[(kv.data_ptr(), kv.nbytes, device_id, "")])
        self._nixl_agent.register_memory(reg)
        # All-gather base pointers so each rank knows the previous rank's buffer address.
        info = torch.tensor([kv.data_ptr(), kv.nbytes, device_id], dtype=torch.int64).cuda()
        all_info = [torch.zeros(3, dtype=torch.int64).cuda() for _ in range(self.world_size)]
        dist.all_gather(all_info, info, group=self.pg)
        ring_rank = dist.get_rank(group=self.pg)
        prev_ring_rank = (ring_rank - 1) % self.world_size
        prev = all_info[prev_ring_rank]
        self._nixl_prev_base_ptr = int(prev[0].item())
        self._nixl_prev_device = int(prev[2].item())
        self._nixl_kv_bufs_registered = True

    def _ring_send_recv_nixl(self, send: torch.Tensor, recv: torch.Tensor, cur: int) -> None:
        """NIXL P2P exchange: READ previous rank's kv_bufs[cur] into local recv buffer.

        Uses a barrier to guarantee the remote send buffer is populated before the
        READ is submitted. The barrier cost is acceptable for a first experiment;
        stream-event synchronization can replace it once correctness is confirmed.
        """
        from tensorrt_llm._torch.disaggregation.base.agent import MemoryDescs, TransferRequest

        # Ensure local CUDA work (copy_ into send buffer) is globally visible.
        torch.cuda.current_stream().synchronize()
        dist.barrier(group=self.pg)

        slot_bytes = send.nbytes
        remote_src_ptr = self._nixl_prev_base_ptr + cur * slot_bytes
        local_dst_ptr = recv.data_ptr()
        device_id = torch.cuda.current_device()

        src = MemoryDescs("VRAM", [(remote_src_ptr, slot_bytes, self._nixl_prev_device)])
        dst = MemoryDescs("VRAM", [(local_dst_ptr, slot_bytes, device_id)])
        req = TransferRequest(
            op="READ",
            src_descs=src,
            dst_descs=dst,
            remote_name=self._nixl_prev_agent_name,
        )
        self._nixl_status = self._nixl_agent.submit_transfer_requests(req)

    def _ring_wait(self) -> None:
        if self._nixl_ready:
            if self._nixl_status is not None:
                self._nixl_status.wait()
                self._nixl_status = None
            # Fence so subsequent CUDA kernels see the newly-received KV data.
            torch.cuda.synchronize()
            return
        if self._ub_p2p_ready:
            return  # UB kernels synchronize internally
        for r in self._p2p_reqs:
            r.wait()
        self._p2p_reqs.clear()

    def _ensure_buffers(self, q: torch.Tensor, k: torch.Tensor) -> None:
        B, S, H, D = q.shape
        H_kv = k.shape[2]
        key = (B, S, H, H_kv, D, q.device, q.dtype, k.dtype)
        if key == self._buf_key:
            return
        self._kv_bufs = k.new_empty(2, 2, B, S, H_kv, D)
        # Accumulate ring blocks in fp32 to avoid repeated bf16<->fp32 rounding
        # across online-softmax merges
        self._out_buf = q.new_empty(B, S, H, D, dtype=torch.float32)
        self._lse_buf = q.new_empty(B, S, H, dtype=torch.float32)
        self._buf_key = key
        # Register new kv_bufs with NIXL when the shape changes (lazy, deferred
        # from __init__ because buffer size isn't known until the first forward).
        if self._nixl_ready:
            self._nixl_kv_bufs_registered = False
            self._nixl_register_kv_bufs()

    def _update_out_and_lse(
        self,
        out: torch.Tensor,
        lse: torch.Tensor,
        block_out: torch.Tensor,
        block_lse: torch.Tensor,
    ) -> None:
        """Online-softmax merge of (out, lse) with (block_out, block_lse). In-place on out/lse."""
        c = torch.sigmoid(block_lse.unsqueeze(-1) - lse.unsqueeze(-1))
        out.sub_(c * (out - block_out))
        lse.sub_(F.logsigmoid(lse - block_lse))

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        attention_mask: PredefinedAttentionMask = PredefinedAttentionMask.FULL,
        **kwargs,
    ) -> torch.Tensor:
        # Bypass ring for cross-attention (Q/KV seq lengths differ).
        if k.shape[1] != q.shape[1]:
            return self.inner.forward(q=q, k=k, v=v, attention_mask=attention_mask, **kwargs)
        if attention_mask != PredefinedAttentionMask.FULL:
            raise NotImplementedError(
                f"RingAttention only supports FULL attention mask, got {attention_mask}."
            )

        inner_kw = {kk: vv for kk, vv in kwargs.items() if kk != "attention_mask"}

        self._ensure_buffers(q, k)
        kv_bufs = self._kv_bufs
        out = self._out_buf
        lse = self._lse_buf

        kv_bufs[0, 0].copy_(k)
        kv_bufs[0, 1].copy_(v)
        for step in range(self.world_size):
            cur, nxt = step % 2, 1 - step % 2
            if step < self.world_size - 1:
                self._ring_send_recv(kv_bufs[cur], kv_bufs[nxt], cur)
            block_out, block_lse_bh = self.inner.forward_with_lse(
                q=q,
                k=kv_bufs[cur, 0],
                v=kv_bufs[cur, 1],
                attention_mask=PredefinedAttentionMask.FULL,
                **inner_kw,
            )
            # Inner backend returns LSE as [B, H, S]; merge uses [B, S, H] with out [B, S, H, D].
            block_lse = block_lse_bh.transpose(1, 2).contiguous()
            if step == 0:
                out.copy_(block_out)
                lse.copy_(block_lse)
            else:
                self._update_out_and_lse(out, lse, block_out, block_lse)
            if step < self.world_size - 1:
                self._ring_wait()
        if out.dtype != q.dtype:
            return out.to(dtype=q.dtype)
        return out

    @property
    def preferred_layout(self) -> AttentionTensorLayout:
        return self._preferred_layout

    @classmethod
    def support_fused_qkv(cls) -> bool:
        return False

    @classmethod
    def support_lse(cls) -> bool:
        return False


def wrap_parallel_attention(
    attn: AttentionBackend,
    *,
    visual_gen_mapping: Optional["VisualGenMapping"] = None,
    enable_sequence_parallel: bool = True,
) -> AttentionBackend:
    """Wrap a compute backend with the configured parallelism strategy.

    Nesting order (inner → outer):
    - Attention2D + Ulysses: Attention2DAttention → UlyssesAttention
    - Ring + Ulysses: RingAttention → UlyssesAttention

    When ``enable_sequence_parallel`` is False, no wrappers are applied (callers
    use this for cross-attention paths that cannot use Ulysses/Ring/Attention2D).
    """
    if not enable_sequence_parallel or visual_gen_mapping is None:
        return attn

    vgm = visual_gen_mapping
    ring_size = vgm.ring_size
    ulysses_size = vgm.ulysses_size
    attn2d_size = vgm.attn2d_row_size * vgm.attn2d_col_size

    if attn2d_size > 1:
        attn = Attention2DAttention(
            inner_backend=attn,
            row_process_group=vgm.attn2d_row_group,
            col_process_group=vgm.attn2d_col_group,
        )
    elif ring_size > 1:
        attn = RingAttention(attn, process_group=vgm.ring_group)

    if ulysses_size > 1:
        attn = UlyssesAttention(attn, process_group=vgm.ulysses_group)
    return attn
