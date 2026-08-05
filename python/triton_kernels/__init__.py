"""
Triton GPU Kernel Suite for BitMC-SSM
Ultra-fast GPU fused operations for 1-bit / 2-bit quantization, FWHT, and Delta-SSM recurrent scans.
"""

from .hadamard_act4 import fused_hadamard_act4, pytorch_fast_hadamard_transform, pytorch_quantize_act_4bit
from .delta_ssm_scan import fused_delta_ssm_scan, pytorch_delta_ssm_scan
from .ternary_quant import fused_ternary_quant, pytorch_quantize_weight_ternary

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
    "pytorch_fast_hadamard_transform",
    "pytorch_quantize_act_4bit",
    "pytorch_delta_ssm_scan",
    "pytorch_quantize_weight_ternary",
]
