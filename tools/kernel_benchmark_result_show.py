import re
import numpy as np
import matplotlib.pyplot as plt

table_text = """
| Kernel Revision                                                       | [16, 1024, 32, 128] | [16, 2048, 32, 128] | [16, 4096, 32, 128] | [16, 8192, 32, 128] |
| 0. Official Impl. (ms,TFLOPs)|    [5.32, 25.84]|      [18.87, 29.14] |  [72.79, 30.21] |       [292.02, 30.12]|
| 1. Base Impl.|         [10.35, 13.28] |      [36.15, 15.21] |         [144.21, 15.25] |      [570.67, 15.41] |
| 2. 1 + async_copy|         [9.61, 14.30]|      [34.94, 15.74] |         [139.67, 15.74] |      [560.26, 15.70] |
| 3. 2 + Eagerly Loading K & V Blocks|    [10.69, 12.86] |      [37.42, 14.69] |         [136.99, 16.05]|     [543.61, 16.18] |
|4. 3 + mem swizzling|  [5.22, 26.32] | [18.91, 29.08] |[76.94, 28.58] |[310.35, 28.34]|
| 5. 4 + Interleaving LD/ST with Computation|[5.11, 26.91]|      [19.22, 28.60] |         [77.53, 28.36]|[322.10, 27.31] |
| 6. 5 + Double Buffering SM2RF Loads|[5.10, 26.96]|      [19.02, 28.90] |          [82.06, 26.80]|     [323.92, 27.15] |
|7. 6 + optimized softmax |[5.00, 27.50]|[19.39, 28.36]|[81.35, 27.03]|[322.10, 27.31]|
|7. 7 with Br=128, Bc=64 and 8 warps|[5.53, 24.83]|[19.45, 28.26]|[78.22, 28.11]|[313.72, 28.04]|
"""

lines = table_text.split("\n")

kernels = []
data = []

pattern = r"\[(\d+\.\d+),\s*(\d+\.\d+)\]"

for line in lines:
    if "[" not in line or "Official" in line or "Impl" in line or "Base" in line or "+" in line or "Br=" in line:
        parts = line.split("|")
        if len(parts) < 2:
            continue
        
        name = parts[1].strip()
        matches = re.findall(pattern, line)
        
        if len(matches) == 4:
            kernels.append(name)
            row=[]
            for m in matches:
                ms=float(m[0])
                tflops=float(m[1])
                row.append([ms,tflops])
            data.append(row)

data=np.array(data)

configs = [
"[16,1024,32,128]",
"[16,2048,32,128]",
"[16,4096,32,128]",
"[16,8192,32,128]"
]

official_tflops=data[0,:,1]

print("\n=== 自动生成 Markdown 表格 ===\n")

header="| Kernel | " + " | ".join(configs) + " |"
print(header)
print("|---"* (len(configs)+1) + "|")

for i,k in enumerate(kernels):
    row=[]
    for j in range(len(configs)):
        ms,tflops=data[i,j]
        ratio=tflops/official_tflops[j]*100
        row.append(f"[{ms:.2f}, {tflops:.2f}], {ratio:.2f}%")
    
    print("|",k,"|"," | ".join(row),"|")

# -------------------------
# 画 TFLOPs 柱状图
# -------------------------

tflops=data[:,:,1]

n_kernel=len(kernels)
n_config=len(configs)

x=np.arange(n_config)
width=0.08

plt.figure(figsize=(12,6))

for i in range(n_kernel):
    plt.bar(x+i*width,tflops[i],width,label=kernels[i])

plt.xticks(x+width*n_kernel/2,configs)

plt.ylabel("TFLOPs")
plt.xlabel("Config")
plt.title("Kernel TFLOPs Benchmark")
plt.legend(fontsize=8)

plt.tight_layout()
plt.savefig("kernel_benchmarks.png")