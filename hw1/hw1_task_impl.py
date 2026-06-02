import torch


def lowest_ai_fn(x: torch.Tensor) -> torch.Tensor:
    """Lowest arithmetic intensity baseline (0 FLOP/Byte)."""
    return x.clone()


def make_compute_fn(num_ops: int, compiled: bool = True):
    """Return an eager or compiled function whose work scales with num_ops."""
    def fn(x: torch.Tensor) -> torch.Tensor:
        acc = x.clone()
        for _ in range(num_ops):
            acc = acc * x + x
        return acc
    if compiled:
        fn = torch.compile(fn)
    return fn


def benchmark_fn(fn, *args, warmup=25, rep=100) -> float:
    """Benchmark a GPU function using CUDA events.
    Returns median execution time in milliseconds.
    """
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    times = []
    for _ in range(rep):
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        start.record()
        fn(*args)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return float(torch.tensor(times).median())


def compute_elementwise_metrics(num_elements, num_ops, bytes_per_element, ms, variant):
    total_flops = 2 * num_ops * num_elements

    if variant == "compiled":
        total_bytes = 2 * num_elements * bytes_per_element
    else:
        total_bytes = (6 * num_ops + 2) * num_elements * bytes_per_element

    ai = total_flops / total_bytes
    achieved_flops = total_flops / (ms * 1e-3)
    return total_flops, ai, achieved_flops


# ============================================================================
# Part 3: Short Writeup
# ============================================================================
#
# Q1. Why does performance rise as arithmetic intensity increases even though
# the measured runtime changes only a little?
# A1: Runtime stays nearly constant because the kernel remains memory-bound.
# As num_ops grows, more FLOPs are done in the same time, so measured
# FLOP/s rises even though the wall time barely changes.
#
# Q2. Why did matmul 1024x1024 achieve lower FLOP/s than 128 ops compiled?
# A2: A 1024x1024 matmul is too small to saturate all SMs on an L40S.
# Launch overhead and poor occupancy reduce effective throughput.
#
# Q3. Why does runtime increase more noticeably between 64 and 128 ops?
# A3: The kernel transitions from memory-bound to compute-bound.
# Above 64 ops the ALUs become the bottleneck so runtime grows.
#
# Q4. Why do eager ops-K points look so different from compiled ones?
# A4: Eager mode materializes intermediates to HBM between every op,
# inflating bytes moved and keeping AI very low regardless of num_ops.
# Compiled mode fuses the loop so intermediates stay in registers.

# ============================================================================
# Part 3: Updated Writeup with Measured Data
# ============================================================================
#
# Q1. Why does performance rise as arithmetic intensity increases even though
# the measured runtime changes only a little?
# A1: Runtime stays nearly constant (0.876ms to 0.882ms across all compiled
# runs from 1 ops to 128 ops). As num_ops grows from 1 to 128, AI rises
# from 0.25 to 32 FLOP/B and TFLOP/s rises from 0.15 to 19.47 — all while
# wall time barely moves. The kernel is memory-bound so the GPU finishes
# compute while waiting for the next memory transfer.
#
# Q2. Why did matmul 1024x1024 achieve lower FLOP/s than 128 ops compiled?
# A2: matmul 1024x1024 achieved 23.03 TFLOP/s on our L40S run, which is
# actually slightly above 128 ops compiled (19.47 TFLOP/s). However small
# matmuls underperform because the 1024x1024 problem is too small to keep
# all SMs busy — poor occupancy and kernel launch overhead dominate.
# Larger matmuls (2048: 40.50, 4096: 36.17 TFLOP/s) show this clearly.
#
# Q3. Why does runtime increase more noticeably between 64 and 128 ops?
# A3: Measured compiled runtime stays flat at ~0.88ms all the way to 128
# ops — the L40S ridge point is 106 FLOP/B but compiled AI only reaches
# 32 FLOP/B at 128 ops. The kernel is still memory-bound at these sizes
# so compute is not yet the bottleneck. Larger num_ops would eventually
# cross the ridge and show runtime growth.
#
# Q4. Why do eager ops-K points look so different from compiled ones?
# A4: Eager runtime grows linearly with num_ops (3.3ms at 1 op to 325ms
# at 128 ops) while AI stays stuck at ~0.08 FLOP/B the whole time.
# Each multiply and add launches a separate kernel, writing intermediates
# to HBM every iteration — bytes explode while FLOPs stay the same.
# Compiled fuses everything into one kernel: runtime flat at ~0.88ms
# while AI grows from 0.25 to 32 FLOP/B, moving rightward on the roofline.
