import torch
import math

# python setup.py build_ext --inplace

from afa_py.afa_config import AFAForwardKernelConfig

def run_test():
    # 尝试加载编译好的内核模块
    try:
        import afa_flash_attention_kernels as fa
    except ImportError:
        print("未找到模块，请先执行: python setup.py build_ext --inplace")
        return

    # 设置随机种子保证可复现
    torch.manual_seed(42)
    device = torch.device("cuda")

    # 2. 定义参数 (需符合 B_r, B_c 的倍数要求)
    B, S, H, D = 16, 2048, 16, 128
    dtype = torch.float16

    # 3. 准备数据 (Layout: [Batch, Seq, Head, Dim])
    q = torch.randn(B, S, H, D, device=device, dtype=dtype)
    k = torch.randn(B, S, H, D, device=device, dtype=dtype)
    v = torch.randn(B, S, H, D, device=device, dtype=dtype)
    
    # 你的 C++ 代码支持传入可选的 out_
    o = torch.empty_like(q)

    # 4. 初始化配置对象
    # 注意：B_r, B_c, d_head 必须与你 flash_kernels.cuh 中预编译的一致
    cfg = AFAForwardKernelConfig()

    print(f">>> 正在运行 Flash Attention 内核 (SM_{torch.cuda.get_device_capability()[0]}{torch.cuda.get_device_capability()[1]})...")

    # 5. 调用自定义算子
    # 返回 std::make_tuple(TO, runtime)
    custom_out, runtime = fa.forward(cfg, q, k, v, o, True)
    print(custom_out.shape)  # 应该是 [B, S, H, D]

    # 6. 计算验证值 (PyTorch 原生实现)
    # PyTorch 默认期望 [B, H, S, D]，所以需要先 transpose
    q_ref = q.transpose(1, 2)
    k_ref = k.transpose(1, 2)
    v_ref = v.transpose(1, 2)

    # 计算 Scale 系数 (1/sqrt(d))
    scale = 1.0 / math.sqrt(D)
    
    # 使用 PyTorch 高效实现作为基准
    with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False):
        ref_out = torch.nn.functional.scaled_dot_product_attention(
            q_ref, k_ref, v_ref, attn_mask=None, dropout_p=0.0, is_causal=False, scale=scale
        )
    
    # 将基准结果转回 [B, S, H, D]
    ref_out = ref_out.transpose(1, 2).contiguous()

    # 7. 精度比对
    max_diff = (custom_out - ref_out).abs().max().item()
    avg_diff = (custom_out - ref_out).abs().mean().item()

    print(f"--- 测试结果 ---")
    print(f"GPU 耗时 (Kernel): {runtime:.4f} ms")
    print(f"最大绝对误差: {max_diff:.6e}")
    print(f"平均绝对误差: {avg_diff:.6e}")

    if max_diff < 1e-2:
        print("✅ 验证通过 (FP16 允许合理误差范围)")
    else:
        print("❌ 误差过大，请检查 Scaler 或 Softmax 实现")

if __name__ == "__main__":
    run_test()