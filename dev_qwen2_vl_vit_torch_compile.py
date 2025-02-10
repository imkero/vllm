import os

if "VLLM_USE_V1" not in os.environ:
    os.environ["VLLM_USE_V1"] = "1"
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

MODEL_PATH = "Qwen/Qwen2-VL-7B-Instruct"
ENFORCE_EAGER = os.environ.get("ENFORCE_EAGER", "0") == "1"
VIT_COMPILE = os.environ.get("VIT_COMPILE", "1") == "1"

import torch
import time
from contextlib import contextmanager
from vllm import LLM, envs

@contextmanager
def timed(label):
    start_time = time.perf_counter()
    print(f"[{label}] start")
    try:
        yield
        end_time = time.perf_counter()
        print(f"[{label}] time cost: {end_time - start_time:.3f}s")
    except:
        end_time = time.perf_counter()
        print(f"[{label}] failed after: {end_time - start_time:.3f}s")
        raise

llm = LLM(
    model=MODEL_PATH,
    limit_mm_per_prompt={"image": 1},
    max_model_len=1024,
    max_num_seqs=10,
    max_num_batched_tokens=1024,
    gpu_memory_utilization=0.9,
    mm_processor_kwargs={
        "max_pixels": 1024 * 1024,
    },
    enforce_eager=ENFORCE_EAGER,
)

if envs.VLLM_USE_V1:
    apply_model = llm.llm_engine.engine_core.engine_core.model_executor.apply_model
else:
    apply_model = llm.apply_model

def compile_vit(model):
    model.compile_vit()
    torch.cuda.synchronize()

def apply_vit(pixel_values, grid_thw):
    def _fn(model):
        with torch.inference_mode():
            return model.visual(pixel_values, grid_thw)
    return apply_model(_fn)[0]

if VIT_COMPILE:
    with timed("compile vit"):
        apply_model(compile_vit)

def benchmark_vit(grid_thw):
    num_features = 0
    for t, h, w in grid_thw:
        num_features += t * h * w

    # prepare input
    with torch.inference_mode():
        pixel_values = torch.rand((num_features, 1176), dtype=torch.bfloat16, pin_memory=True)
        grid_thw = torch.tensor(grid_thw, dtype=torch.int64)

        pixel_values = pixel_values.cuda()
        torch.cuda.synchronize()

    t0 = time.perf_counter()

    ret = apply_vit(pixel_values, grid_thw)
    # well-written ViT allows async CUDA operation, so a sync is needed for benchmark
    torch.cuda.synchronize() 

    t1 = time.perf_counter()
    
    del pixel_values
    del ret

    return t1 - t0

print()
print("[torch.compiled]" if VIT_COMPILE else "[eager]")
print()

# warmup
benchmark_vit([
    [1, 126, 126],
])

def benchmark_main(test_cases):
    for grid_thw in test_cases:
        t, h, w = grid_thw

        time_cost = benchmark_vit([grid_thw])
        
        print("\t".join([
            "grid_t",
            "height",
            "width",
            "time_cost_ms",
        ]))
        # print result in csv format
        print("\t".join(map(str, [
            t, # grid_t (1 for image; frame_count // 2 for video)
            h * 14,
            w * 14, # frame size (in pixels)
            round(1000 * time_cost), # time cost (in milliseconds)
        ])))

print("[scaling pixels]")
benchmark_main([
    [1, i, i] # grid_thw
    for i in range(4, 128, 4)
])

print("[scaling nframes]")
benchmark_main([
    [i, 36, 36] # grid_thw
    for i in range(1, 73)
])

del llm

from torch.distributed import destroy_process_group
destroy_process_group()
