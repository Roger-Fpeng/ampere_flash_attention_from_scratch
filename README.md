# ampere_flash_attention_from_scratch
This is an implementation of flash attention from scratch on Nvidia's Ampere architecture ( RTX 3080 for verification).

``` shell
# step1: build kernel
python setup.py build_ext --inplace

# step2: test demo
python demo.py
```

## Performance Test
Conducted performance testing of the kernel implementation versus the official implementation ([script for official implementation testing](https://github.com/Roger-Fpeng/flash-attention/commit/93c173b13d9392551ca8ac7e378a77ba4f4dbd08#diff-ea9860ac378d44e5f307b5be14b559e326da5a5894124ebfde64faf916238210)). All experiments were run on an NVIDIA RTX 3080 GPU with:
- PyTorch version: 2.7.0+cu128
- Capability: (8, 6)



The header denotes [batch_size, seq_len, num_heads, head_dim].
Row 0 reports the official implementation TFLOPs, while rows 1–5 show performance as a percentage relative to the official implementation. 
Incidentally, the theoretical FP16 compute throughput of the RTX 3080 is [29.77 TFLOPs](https://www.waredb.com/processor/nvidia-geforce-rtx-3080). For simplicity, we count the total FLOPs as:

2
×
batch_size
×
num_heads
×
seq_len
2
×
head_dim
.
2×batch_size×num_heads×seq_len
2
×head_dim.

Under this approximation, the measured TFLOPs can be slightly greater than the theoretical value.

| Kernel Revision                                                       |           [16, 1024, 32, 128] |       [16, 2048, 32, 128] |       [16, 4096, 32, 128] |  [16, 8192, 32, 128]  |
| :---------------------------------------------------------------------- | -------------: | ---------: | -------------: | ---------: |
| 0. Official Impl. (ms,TFLOPs)|    [5.32, 25.84]|      [18.87, 29.14] |  [72.79, 30.21] |       [292.02, 30.12]|
|                                                                         |                |            |                |            |
| 1. Base Impl.|         [10.35, 13.28] |      [36.15, 15.21] |         [144.21, 15.25] |      [570.67, 15.41] |
| 2. 1 + async_copy|         [9.61, 14.30] |      [34.94, 15.74] |         [139.67, 15.74] |      [560.26, 15.70] |
| 3. 2 + Eagerly Loading K & V Blocks|    [10.69, 12.86] |      [37.42, 14.69] |         [136.99, 16.05]|     [543.61, 16.18] |
|4. 3 + mem swizzling|  [5.22, 26.32] | [18.91, 29.08] |[76.94, 28.58] |[310.35, 28.34]|
| 5. 4 + Interleaving LD/ST with Computation|          [5.11, 26.91] |      [19.22, 28.60] |         [77.53, 28.36] |     [322.10, 27.31] |
| 6. 5 + Double Buffering SM2RF Loads|          [5.10, 26.96] |      [19.02, 28.90] |          [82.06, 26.80]|     [323.92, 27.15] |
|7. 6 + optimized softmax |[5.00, 27.50]|[19.39, 28.36]|[81.35, 27.03]|[322.10, 27.31]|
|7. 7 with Br=128, Bc=64 and 8 warps|[5.53, 24.83]|[19.45, 28.26]|[78.22, 28.11]|[313.72, 28.04]|


# Implementation Explanation (Blog Post)
1. [Flash attention mechanism](https://zhuanlan.zhihu.com/p/2011287950818840919)
2. [GEMM and Tiling](https://zhuanlan.zhihu.com/p/2011926671276651005)
3. Data movement (TODO)
4. Online Softmax (TODO)