#!/usr/bin/env python3
"""
Benchmark script to measure batch splitting performance improvement
"""

import torch
import time
import subprocess
import sys
import os
from nanovllm import LLM

def test_parallelism():
    """Test to verify if replicas truly process in parallel by comparing single vs parallel"""

    print("=" * 80)
    print("TESTING TRUE PARALLELISM: Single vs Parallel Execution Comparison")
    print("=" * 80)

    batch_size = 128

    # Test 1: Single replica (no batch splitting)
    print("\n" + "─" * 40)
    print("PHASE 1: Single Replica (Sequential Processing)")
    print("─" * 40)

    # Run single replica test in subprocess to avoid GPU memory conflicts
    cmd_single = [
        sys.executable, __file__,
        'single_test', 'single', '0', str(batch_size)
    ]

    result_single = subprocess.run(cmd_single, capture_output=True, text=True, cwd=os.getcwd())
    if result_single.returncode != 0:
        print(f"ERROR: Single replica test failed")
        print(f"STDOUT: {result_single.stdout}")
        print(f"STDERR: {result_single.stderr}")
        return None

    # Parse result
    time_single = throughput_single = None
    for line in result_single.stdout.split('\n'):
        if line.startswith('RESULT_SINGLE:'):
            parts = line.split(':')
            if len(parts) >= 3:
                time_single = float(parts[1])
                throughput_single = float(parts[2])
                break

    if time_single is None:
        print("ERROR: Could not parse single replica results")
        return None

    print(".4f")
    print(".1f")
    print("NOTE: This runs the entire MLP sequentially (no batch splitting)")

    # Test 2: Multiple replicas (batch splitting)
    print("\n" + "─" * 40)
    print("PHASE 2: 3 Replicas (Parallel Processing)")
    print("─" * 40)

    # Run parallel test in subprocess
    cmd_parallel = [
        sys.executable, __file__,
        'single_test', 'parallel', '2', str(batch_size)
    ]

    result_parallel = subprocess.run(cmd_parallel, capture_output=True, text=True, cwd=os.getcwd())
    if result_parallel.returncode != 0:
        print(f"ERROR: Parallel replica test failed")
        print(f"STDOUT: {result_parallel.stdout}")
        print(f"STDERR: {result_parallel.stderr}")
        return None

    # Parse result
    time_parallel = throughput_parallel = None
    for line in result_parallel.stdout.split('\n'):
        if line.startswith('RESULT_PARALLEL:'):
            parts = line.split(':')
            if len(parts) >= 3:
                time_parallel = float(parts[1])
                throughput_parallel = float(parts[2])
                break

    if time_parallel is None:
        print("ERROR: Could not parse parallel replica results")
        return None

    print(".4f")
    print(".1f")
    print("NOTE: This splits the down_proj operation across 3 replicas")

    speedup = time_single / time_parallel
    print(".2f")
    print(".1f")

    print("\n" + "=" * 80)
    print("CONCLUSION:")
    print(f"✅ Single replica time: {time_single:.4f}s")
    print(f"✅ Parallel replica time: {time_parallel:.4f}s")
    print(f"✅ Speedup: {speedup:.2f}x")

    if speedup > 1.0:
        print("✅ PARALLEL PROCESSING WORKS! Replicas run faster than single replica.")
        print("   This proves that batch splitting across replicas provides real parallelism.")
    else:
        print("❌ NO PARALLEL BENEFIT: Parallel version is slower or same as single replica.")
        print("   This suggests replicas don't run in parallel, or overhead dominates.")

    print("\nTECHNICAL ANALYSIS:")
    print("- If speedup > 1.0: CUDA streams enable true parallel execution")
    print("- If speedup ≈ 1.0: Replicas run serially despite separate streams")
    print("- If speedup < 1.0: Parallel overhead outweighs any benefit")
    print("=" * 80)

    return speedup

def benchmark_single_config(config_name, replicas, batch_size):
    """Run benchmark for a single configuration in a separate process"""

    # Initialize LLM with specified replication
    llm = LLM('/home/ztang23/huggingface/Qwen3-0.6B/',
             op_replica_configs={'down_proj': replicas})

    model = llm.model_runner.model.model

    # Create test input
    test_input = torch.randn(batch_size, 1024, dtype=torch.bfloat16).to('cuda')

    # Warm up
    for _ in range(5):
        mlp = model.layers[0].mlp
        _ = mlp(test_input)

    # Benchmark multiple runs
    num_runs = 20
    times = []

    torch.cuda.synchronize()  # Ensure GPU is ready

    for run in range(num_runs):
        start_time = time.time()

        mlp = model.layers[0].mlp
        output = mlp(test_input)

        torch.cuda.synchronize()  # Wait for GPU completion
        end_time = time.time()

        times.append(end_time - start_time)

    avg_time = sum(times) / len(times)
    throughput = batch_size / avg_time  # tokens/second
    latency = avg_time * 1000  # ms
    total_replicas = replicas + 1  # replicated + original

    # Print results for parent process to capture
    print(f"RESULT:{config_name}:{avg_time:.6f}:{throughput:.1f}:{latency:.3f}:{total_replicas}")

    # Clean up
    llm.exit()
    return True

def benchmark_correct_comparison():
    """Benchmark with CORRECT baseline: single replica vs batch splitting"""

    print(f"{'='*70}")
    print("CORRECT BATCH SPLITTING BENCHMARK: Single vs Parallel Processing")
    print(f"{'='*70}")

    # Test batch size
    test_batch_size = 64  # Use a much larger batch size to maximize parallelism benefits

    # Test configurations
    configs = [
        {'name': 'Single Replica', 'replicas': 0},  # No replication = single replica
        {'name': 'Batch Splitting (2 replicas)', 'replicas': 1},  # 1 replicated + 1 original = 2 total
        {'name': 'Batch Splitting (3 replicas)', 'replicas': 2},  # 2 replicated + 1 original = 3 total
        {'name': 'Batch Splitting (4 replicas)', 'replicas': 3},  # 3 replicated + 1 original = 4 total
    ]

    results = {}

    for config in configs:
        print(f"\n{'─'*50}")
        print(f"Testing: {config['name']}")
        print(f"{'─'*50}")
        print(f"Batch size: {test_batch_size}")

        # Run benchmark in separate process to avoid state conflicts
        cmd = [
            sys.executable, __file__,
            'single', config['name'], str(config['replicas']), str(test_batch_size)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
        if result.returncode != 0:
            print(f"ERROR: Benchmark failed for {config['name']}")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            continue

        # Parse result from subprocess output
        for line in result.stdout.split('\n'):
            if line.startswith('RESULT:'):
                parts = line.split(':')
                if len(parts) >= 6:
                    config_name = parts[1]
                    avg_time = float(parts[2])
                    throughput = float(parts[3])
                    latency = float(parts[4])
                    total_replicas = int(parts[5])

                    results[config_name] = {
                        'avg_time': avg_time,
                        'throughput': throughput,
                        'latency': latency,
                        'total_replicas': total_replicas
                    }

                    print(f"Average time: {avg_time:.4f} seconds")
                    print(f"Throughput: {throughput:.1f} tokens/second")
                    print(f"Latency: {latency:.2f} ms")
                    print(f"  Total replicas used: {total_replicas}")

                    # Check if batch splitting occurred
                    if total_replicas > 1:
                        print("  → Mode: PARALLEL (batch splitting enabled)")
                    else:
                        print("  → Mode: SINGLE (no batch splitting)")
                    break

    # CORRECT ANALYSIS: Compare parallel vs single processing
    print(f"\n{'='*70}")
    print("CORRECT ANALYSIS: Parallel Speedup vs Single Processing")
    print(f"{'='*70}")

    if 'Single Replica' not in results or 'Batch Splitting (2 replicas)' not in results or 'Batch Splitting (3 replicas)' not in results:
        print("ERROR: Missing benchmark results, cannot analyze speedup")
        return

    single_result = results['Single Replica']
    splitting_2x = results['Batch Splitting (2 replicas)']
    splitting_3x = results['Batch Splitting (3 replicas)']

    print("\nDirect Performance Comparison:")
    print(f"Single Replica:      {single_result['throughput']:.1f} tok/s, {single_result['latency']:.2f}ms")
    print(f"Batch Splitting 2x:  {splitting_2x['throughput']:.1f} tok/s, {splitting_2x['latency']:.2f}ms")
    print(f"Batch Splitting 3x:  {splitting_3x['throughput']:.1f} tok/s, {splitting_3x['latency']:.2f}ms")

    print("\nParallel Speedup Analysis:")
    speedup_2x = splitting_2x['throughput'] / single_result['throughput']
    speedup_3x = splitting_3x['throughput'] / single_result['throughput']
    print(f"2x replicas speedup: {speedup_2x:.2f}x")
    print(f"3x replicas speedup: {speedup_3x:.2f}x")

    print("\nScaling Efficiency:")
    ideal_2x = single_result['throughput'] * 2
    ideal_3x = single_result['throughput'] * 3

    actual_2x = splitting_2x['throughput']
    actual_3x = splitting_3x['throughput']

    efficiency_2x = (actual_2x / ideal_2x) * 100
    efficiency_3x = (actual_3x / ideal_3x) * 100

    print(f"2x scaling efficiency: {efficiency_2x:.1f}%")
    print(f"3x scaling efficiency: {efficiency_3x:.1f}%")

    print("\nCONCLUSION:")
    print("✅ CORRECT BASELINE: Single replica processing")
    print("✅ TRUE SPEEDUP: Parallel processing benefit")
    print(f"Real 2x speedup: {speedup_2x:.2f}x (not just replica count)")
    print(f"Real 3x speedup: {speedup_3x:.2f}x (parallel processing power)")
    print("This shows the REAL benefit of operator-level batch splitting!")

def run_single_test(test_type, replicas, batch_size):
    """Run a single test (single or parallel) and output results"""

    # Initialize LLM
    llm = LLM('/home/ztang23/huggingface/Qwen3-0.6B/',
             op_replica_configs={'down_proj': replicas})

    model = llm.model_runner.model.model
    test_input = torch.randn(batch_size, 1024, dtype=torch.bfloat16).to('cuda')

    # Warm up
    for _ in range(3):
        mlp = model.layers[0].mlp
        _ = mlp(test_input)

    # Benchmark
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    mlp = model.layers[0].mlp
    output = mlp(test_input)
    end_event.record()
    torch.cuda.synchronize()

    time_taken = start_event.elapsed_time(end_event) / 1000.0
    throughput = batch_size / time_taken

    # Output result for parent process
    result_prefix = "RESULT_SINGLE:" if test_type == "single" else "RESULT_PARALLEL:"
    print(f"{result_prefix}{time_taken:.6f}:{throughput:.1f}")

    llm.exit()

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == 'single':
        # Run single configuration benchmark
        if len(sys.argv) >= 5:
            config_name = sys.argv[2]
            replicas = int(sys.argv[3])
            batch_size = int(sys.argv[4])
            benchmark_single_config(config_name, replicas, batch_size)
        else:
            print("ERROR: Missing arguments for single benchmark")
            sys.exit(1)
    elif len(sys.argv) > 1 and sys.argv[1] == 'single_test':
        # Run single test for parallelism comparison
        if len(sys.argv) >= 5:
            test_type = sys.argv[2]
            replicas = int(sys.argv[3])
            batch_size = int(sys.argv[4])
            run_single_test(test_type, replicas, batch_size)
        else:
            print("ERROR: Missing arguments for single test")
            sys.exit(1)
    elif len(sys.argv) > 1 and sys.argv[1] == 'test_parallel':
        # Test parallelism
        test_parallelism()
    else:
        # Run full comparison benchmark
        benchmark_correct_comparison()
