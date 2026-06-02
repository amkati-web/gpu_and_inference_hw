import torch
from utils import (
    build_model,
    get_input_ids,
    slow_loop,
    time_generation,
    MODEL_NAME,
    PROFILE_STEPS,
    RESULTS_DIR,
)


def optimized_loop(model, input_ids, n_steps):
    generated_ids = input_ids.clone()
    generated_tokens = []
    past_key_values = None

    with torch.inference_mode():
        for step in range(n_steps):
            if past_key_values is None:
                cur_input = generated_ids
            else:
                cur_input = next_token_id.unsqueeze(0)

            outputs = model(
                input_ids=cur_input,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1)
            generated_tokens.append(next_token_id.item())

    return generated_tokens


def profile(loop_fn, model, input_ids, trace_name: str):
    from torch.profiler import profile as torch_profile, ProfilerActivity

    trace_path = RESULTS_DIR / trace_name

    with torch_profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=False,
    ) as prof:
        loop_fn(model, input_ids, PROFILE_STEPS)

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
    prof.export_chrome_trace(str(trace_path))
    print(f"Chrome trace saved to {trace_path}")


def generate_optimized(optimized_trace_name: str) -> float:
    model = build_model(torch.float16)
    input_ids = get_input_ids()

    profile(optimized_loop, model, input_ids, optimized_trace_name)
    elapsed = time_generation(optimized_loop, model, input_ids, "Optimized")

    return elapsed


def main():
    print("=" * 60)
    print("HW2: LLM Inference Optimization")
    print(f"Model: {MODEL_NAME}")
    print("=" * 60)
    print("\n--- Part 1: Slow baseline ---")
    model = build_model(torch.float32)
    input_ids = get_input_ids()
    profile(slow_loop, model, input_ids, "v0_slow_trace.json")
    slow_elapsed = time_generation(slow_loop, model, input_ids, "Slow")
    del model
    torch.cuda.empty_cache()
    print("\n--- Part 2: Optimized ---")
    optimized_elapsed = generate_optimized(optimized_trace_name="v1_optimized_trace.json")
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if optimized_elapsed is None or optimized_elapsed <= 0:
        print("generate_optimized() did not return a positive elapsed time; "
              "cannot compute speedup.")
    else:
        speedup = slow_elapsed / optimized_elapsed
        print(f"  Slow:      {slow_elapsed:6.2f}s")
        print(f"  Optimized: {optimized_elapsed:6.2f}s")
        print(f"  Speedup:   {speedup:6.2f}x  (vs V0 slow baseline)")


if __name__ == "__main__":
    main()


# ============================================================================
# Writeup
# ============================================================================
#
# Changes made and speedup per fix:
#
# 1. KV cache (use_cache=True + past_key_values):
#    Biggest win. The slow loop re-runs attention over the full growing
#    sequence every step (1024 → 1152 tokens). With KV cache each step
#    only processes one new token. CUDA time for matmul dropped from
#    110ms to 5ms across the profiled steps.
#    Impact: estimated 4-5x of the total speedup.
#
# 2. float16 instead of float32:
#    Halves memory bandwidth required to load weights each step.
#    Flash attention (f16) replaced the f32 efficient attention kernel.
#    CUDA attention time dropped from 8.978ms to 0.348ms.
#    Impact: estimated 1.5x additional speedup.
#
# 3. torch.inference_mode():
#    Disables autograd tracking entirely. Removes gradient tape overhead
#    on every tensor operation through the loop.
#    Impact: ~5-10% additional speedup.
#
# Final result: 1.62s → 0.28s = 5.79x speedup (459 tok/s vs 79 tok/s)
#
# Biggest impact: KV cache. Without it attention is O(seq_len) per step —
# the sequence grows from 1024 to 1152 tokens over 128 steps so cost
# compounds every iteration. With KV cache every step is O(1), which is
# the fundamental algorithmic fix for autoregressive generation.
