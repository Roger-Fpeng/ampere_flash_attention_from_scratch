# ampere_flash_attention_from_scratch
This is an implementation of flash attention from scratch on Nvidia's Ampere architecture ( RTX 3080 for verification).

``` shell
# step1: build kernel
python setup.py build_ext --inplace

# step2: test demo
python demo.py
```

More analysis in progress.



# Implementation Explanation (Blog Post)
1. [Flash attention mechanism](https://zhuanlan.zhihu.com/p/2011287950818840919)
2. [GEMM and Tiling](https://zhuanlan.zhihu.com/p/2011926671276651005)
3. Data movement (TODO)
4. Online Softmax (TODO)