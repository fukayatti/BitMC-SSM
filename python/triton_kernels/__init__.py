"""
Triton GPU Kernel Suite for BitMC-SSM
Ultra-fast GPU fused operations for 1-bit / 2-bit quantization, FWHT, Delta-SSM recurrent scans,
Flash Cross-Entropy, RMSNorm, and Ternary GEMM.
"""

from .hadamard_act4 import (
    fused_hadamard_act4,
    pytorch_fast_hadamard_transform,
    pytorch_quantize_act_4bit,
)
from .delta_ssm_scan import (
    fused_delta_ssm_scan,
    pytorch_delta_ssm_scan,
)
from .ternary_quant import (
    fused_ternary_quant,
    pytorch_quantize_weight_ternary,
)
from .cross_entropy import (
    fused_cross_entropy,
    pytorch_cross_entropy,
)
from .rmsnorm_silu import (
    fused_silu_gating,
    pytorch_rmsnorm,
    pytorch_silu_gating,
    FusedRMSNorm,
    FusedRMSNormFunction,
)
from .ternary_gemm import (
    fused_ternary_gemm,
    pytorch_ternary_gemm,
)

try:
    import triton
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

__all__ = [
    "HAS_TRITON",
    "fused_hadamard_act4",
    "fused_delta_ssm_scan",
    "fused_ternary_quant",
    "fused_cross_entropy",
    "fused_silu_gating",
    "fused_ternary_gemm",
    "FusedRMSNorm",
    "FusedRMSNormFunction",
    "pytorch_fast_hadamard_transform",
    "pytorch_quantize_act_4bit",
    "pytorch_delta_ssm_scan",
    "pytorch_quantize_weight_ternary",
    "pytorch_cross_entropy",
    "pytorch_rmsnorm",
    "pytorch_silu_gating",
    "pytorch_ternary_gemm",
]
