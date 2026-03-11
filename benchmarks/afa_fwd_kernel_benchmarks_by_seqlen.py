#!/usr/bin/env python3
"""
Benchmark script for AFA forward kernel across different sequence lengths.

Usage:
    python afa_fwd_kernel_benchmarks_by_seqlen.py <config_string> <batch_size> <n_heads> <seq_len1> [seq_len2] [seq_len3] ...

Example:
    python afa_fwd_kernel_benchmarks_by_seqlen.py "(FP16, 128, 64, 32, 4): async+eager+swizzled+load_2_2_0_tiles" 16 16 512 1024 2048 4096
"""

import sys
import os
import torch
import math
import argparse
from typing import List

# Add parent directory to path to import afa_py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from afa_py.afa_config import AFAForwardKernelConfig, is_valid_config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark AFA forward kernel across different sequence lengths",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "config_string",
        type=str,
        help="Configuration string, e.g., '(FP16, 128, 64, 32, 4): async+eager+swizzled+load_2_2_0_tiles'"
    )
    parser.add_argument(
        "batch_size",
        type=int,
        help="Batch size"
    )
    parser.add_argument(
        "n_heads",
        type=int,
        help="Number of attention heads"
    )
    parser.add_argument(
        "seq_lens",
        type=int,
        nargs="+",
        help="List of sequence lengths to benchmark"
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Number of warmup iterations (default: 5)"
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=30,
        help="Number of benchmark iterations (default: 30)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    return parser.parse_args()


def benchmark_kernel(
    cfg: AFAForwardKernelConfig,
    batch_size: int,
    seq_len: int,
    n_heads: int,
    warmup: int = 5,
    repeats: int = 30,
    seed: int = 42
) -> float:
    """
    Benchmark the kernel for a specific configuration and sequence length.
    
    Returns:
        Mean runtime in milliseconds
    """
    # Set random seed
    torch.manual_seed(seed)
    device = torch.device("cuda")
    
    # Get dtype from config
    dtype = cfg.dtype.to_torch_dtype()
    d_head = cfg.d_head
    
    # Prepare data (Layout: [Batch, Seq, Head, Dim])
    q = torch.randn(batch_size, seq_len, n_heads, d_head, device=device, dtype=dtype)
    k = torch.randn(batch_size, seq_len, n_heads, d_head, device=device, dtype=dtype)
    v = torch.randn(batch_size, seq_len, n_heads, d_head, device=device, dtype=dtype)
    o = torch.empty_like(q)
    
    # Warmup
    for _ in range(warmup):
        _, _ = fa.forward(cfg, q, k, v, o, True)
    
    # Synchronize before benchmarking
    torch.cuda.synchronize()
    
    # Benchmark
    runtimes = []
    for _ in range(repeats):
        _, runtime = fa.forward(cfg, q, k, v, o, True)
        runtimes.append(runtime)
    
    # Calculate mean runtime
    mean_runtime = sum(runtimes) / len(runtimes)
    return mean_runtime


def main():
    """Main function."""
    args = parse_args()
    
    # Try to load the compiled kernel module
    try:
        global fa
        import afa_flash_attention_kernels as fa
    except ImportError:
        print("Error: Could not import afa_flash_attention_kernels module.")
        print("Please build the extension first: python setup.py build_ext --inplace")
        sys.exit(1)
    
    # Parse configuration string
    try:
        cfg = AFAForwardKernelConfig.from_string_config(args.config_string)
    except Exception as e:
        print(f"Error: Failed to parse config string: {e}")
        sys.exit(1)
    
    # Validate configuration
    if not is_valid_config(cfg):
        print(f"Warning: Configuration may not be valid: {cfg}")
    
    # Print header
    print("=" * 80)
    print("AFA Forward Kernel Benchmark by Sequence Length")
    print("=" * 80)
    print(f"Config: {cfg}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Number of Heads: {args.n_heads}")
    print(f"Head Dimension: {cfg.d_head}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"SM Version: {torch.cuda.get_device_capability()[0]}{torch.cuda.get_device_capability()[1]}")
    print(f"Warmup iterations: {args.warmup}")
    print(f"Benchmark iterations: {args.repeats}")
    print("=" * 80)
    print(f"{'Seq Len':<10} {'Runtime (ms)':<15} {'Throughput (TFLOPS)':<20}")
    print("-" * 80)
    
    # Benchmark each sequence length
    results = []
    for seq_len in args.seq_lens:
        try:
            runtime_ms = benchmark_kernel(
                cfg=cfg,
                batch_size=args.batch_size,
                seq_len=seq_len,
                n_heads=args.n_heads,
                warmup=args.warmup,
                repeats=args.repeats,
                seed=args.seed
            )
            
            # Calculate FLOPS (simplified: 2 * batch * n_heads * seq_len^2 * d_head)
            # This is a rough estimate for attention computation
            flops = 2 * args.batch_size * args.n_heads * seq_len * seq_len * cfg.d_head
            tflops = flops / (runtime_ms * 1e-3) / 1e12
            
            results.append((seq_len, runtime_ms, tflops))
            print(f"{seq_len:<10} {runtime_ms:<15.4f} {tflops:<20.2f}")
            
        except Exception as e:
            print(f"{seq_len:<10} ERROR: {e}")
            results.append((seq_len, None, None))
    
    print("=" * 80)
    print("\nSummary:")
    print(f"{'Seq Len':<10} {'Runtime (ms)':<15} {'Throughput (TFLOPS)':<20}")
    print("-" * 80)
    for seq_len, runtime_ms, tflops in results:
        if runtime_ms is not None:
            print(f"{seq_len:<10} {runtime_ms:<15.4f} {tflops:<20.2f}")
        else:
            print(f"{seq_len:<10} {'FAILED':<15}")


if __name__ == "__main__":
    main()
