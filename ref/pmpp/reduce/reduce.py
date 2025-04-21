import torch
import time
from torch.utils.cpp_extension import load

torch.set_grad_enabled(False)

# 1. 编译并加载你的 extension
#    name: 随你起，sources: 你的 .cu（或 .cpp/.cu 混合）文件名
lib = load(
    name='reduce_lib',
    sources=['reducev0.cu'],  
    extra_cflags=['-std=c++17'],
    extra_cuda_cflags=[
        '-O3',
        '-U__CUDA_NO_HALF_OPERATORS__',
        '-U__CUDA_NO_HALF_CONVERSIONS__',
        '-U__CUDA_NO_HALF2_OPERATORS__',
        '-U__CUDA_NO_BFLOAT16_CONVERSIONS__',
        '--expt-relaxed-constexpr',
        '--expt-extended-lambda',
        '--use_fast_math',
    ],
    verbose=True
)

# 2. 简单的 Benchmark 封装
def run_benchmark(fn, x: torch.Tensor, tag: str, warmup=10, iters=200):
    # GPU 预热
    for _ in range(warmup):
        y = fn(x)
    torch.cuda.synchronize()
    # 正式计时
    start = time.time()
    for _ in range(iters):
        y = fn(x)
    torch.cuda.synchronize()
    elapsed = (time.time() - start) * 1e3 / iters  # ms/iter
    print(f"{tag:>20} -> out={y.item():.4g}, avg_time={elapsed:.4f} ms")
    return y

if __name__ == "__main__":
    # 3. 构造测试输入
    S, K = 2048, 2048
    # 正确测试用例（BlockSize=256 → N=512）
    x_f32 = torch.randn(512, device='cuda', dtype=torch.float32)  # N=512
    run_benchmark(lib.simple_sum_256, x_f32, "BlockSize=256")