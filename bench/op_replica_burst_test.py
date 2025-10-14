#!/usr/bin/env python3
"""
Stress test for operator-level replication in nano-vLLM.

This script simulates bursty inference requests and measures the impact of
attention operator replicas on tail latency and throughput.
"""

import argparse
import asyncio
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List, Optional
import statistics

import torch
from tqdm import tqdm

from nanovllm import LLM, SamplingParams


@dataclass
class TestResult:
    """Results from a burst test run."""
    total_requests: int
    total_tokens: int
    duration_sec: float
    throughput_tokens_per_sec: float
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    latencies_ms: List[float]


class BurstTestSimulator:
    """Simulates bursty inference workloads to test replica effectiveness."""

    def __init__(self, model_path: str, max_concurrent_requests: int = 32):
        self.model_path = model_path
        self.max_concurrent_requests = max_concurrent_requests
        self.request_queue = asyncio.Queue()
        self.results = []

    async def generate_request_burst(
        self,
        num_requests: int,
        prompt_length: int = 100,
        max_tokens: int = 50,
        burst_interval_sec: float = 0.1
    ) -> List[str]:
        """Generate a burst of requests."""
        prompts = []
        for i in range(num_requests):
            # Create a prompt of approximately prompt_length tokens
            prompt = f"Question {i}: " + " ".join([f"token{j}" for j in range(prompt_length // 10)])
            prompts.append(prompt)

        return prompts

    def run_burst_test(
        self,
        num_bursts: int = 5,
        requests_per_burst: int = 10,
        prompt_length: int = 100,
        max_tokens: int = 50,
        burst_interval_sec: float = 1.0,
        use_replicas: bool = False,
        replica_config: Optional[dict] = None,
    ) -> TestResult:
        """
        Run a burst test with the specified configuration.

        Args:
            num_bursts: Number of bursts to generate
            requests_per_burst: Number of requests per burst
            prompt_length: Approximate length of each prompt in tokens
            max_tokens: Maximum tokens to generate per request
            burst_interval_sec: Time between bursts
            use_replicas: Whether to enable attention replicas
            replica_config: Replica configuration if use_replicas is True

        Returns:
            TestResult with performance metrics
        """
        print(f"Running burst test: {num_bursts} bursts × {requests_per_burst} requests each")
        print(f"Use replicas: {use_replicas}")

        # Configure LLM
        llm_kwargs = {
            "model": self.model_path,
            "enforce_eager": True,
            "tensor_parallel_size": 1,
        }

        if use_replicas and replica_config:
            llm_kwargs.update(replica_config)

        llm = LLM(**llm_kwargs)
        sampling_params = SamplingParams(
            temperature=0.1,
            max_tokens=max_tokens,
            ignore_eos=True
        )

        all_latencies = []
        total_requests = 0
        total_tokens = 0

        start_time = time.time()

        try:
            for burst_idx in range(num_bursts):
                print(f"Burst {burst_idx + 1}/{num_bursts}")

                # Generate burst of requests
                prompts = asyncio.run(self.generate_request_burst(
                    requests_per_burst, prompt_length, max_tokens
                ))

                # Submit all requests in the burst
                burst_start = time.time()
                outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
                burst_duration = time.time() - burst_start

                # Calculate per-request latencies (approximate)
                avg_burst_latency = burst_duration / len(prompts) * 1000  # ms
                all_latencies.extend([avg_burst_latency] * len(prompts))

                total_requests += len(prompts)
                total_tokens += sum(len(output['token_ids']) for output in outputs)

                # Wait before next burst
                if burst_idx < num_bursts - 1:
                    time.sleep(burst_interval_sec)

        finally:
            # Ensure LLM is properly shut down
            pass

        total_duration = time.time() - start_time

        # Calculate metrics
        throughput = total_tokens / total_duration if total_duration > 0 else 0
        avg_latency = statistics.mean(all_latencies) if all_latencies else 0
        p50_latency = statistics.median(all_latencies) if all_latencies else 0
        p95_latency = statistics.quantiles(all_latencies, n=20)[18] if len(all_latencies) >= 20 else max(all_latencies) if all_latencies else 0
        p99_latency = statistics.quantiles(all_latencies, n=100)[98] if len(all_latencies) >= 100 else max(all_latencies) if all_latencies else 0

        return TestResult(
            total_requests=total_requests,
            total_tokens=total_tokens,
            duration_sec=total_duration,
            throughput_tokens_per_sec=throughput,
            avg_latency_ms=avg_latency,
            p50_latency_ms=p50_latency,
            p95_latency_ms=p95_latency,
            p99_latency_ms=p99_latency,
            latencies_ms=all_latencies
        )


def run_comparison_test(model_path: str, output_file: Optional[str] = None):
    """Run comparison test with and without replicas."""

    simulator = BurstTestSimulator(model_path)

    # Test configurations
    configs = [
        {
            "name": "baseline",
            "use_replicas": False,
            "replica_config": None,
        },
        {
            "name": "replicas_2_per_gpu",
            "use_replicas": True,
            "replica_config": {
                "op_replica_configs": {"attention": 2},
                "replica_devices": [0],  # Single GPU for testing
            },
        },
        {
            "name": "replicas_4_per_gpu",
            "use_replicas": True,
            "replica_config": {
                "op_replica_configs": {"attention": 4},
                "replica_devices": [0],
            },
        },
    ]

    results = []

    for config in configs:
        print(f"\n{'='*50}")
        print(f"Testing configuration: {config['name']}")
        print(f"{'='*50}")

        try:
            result = simulator.run_burst_test(
                num_bursts=3,
                requests_per_burst=5,
                prompt_length=50,
                max_tokens=20,
                burst_interval_sec=0.5,
                use_replicas=config["use_replicas"],
                replica_config=config["replica_config"],
            )

            print(f"Results for {config['name']}:")
            print(f"  Total requests: {result.total_requests}")
            print(f"  Total tokens: {result.total_tokens}")
            print(".2f")
            print(".2f")
            print(".2f")
            print(".2f")
            print(".2f")

            results.append((config["name"], result))

        except Exception as e:
            print(f"Error testing {config['name']}: {e}")
            import traceback
            traceback.print_exc()

    # Save results if requested
    if output_file:
        import json
        output_data = {}
        for name, result in results:
            output_data[name] = {
                "total_requests": result.total_requests,
                "total_tokens": result.total_tokens,
                "duration_sec": result.duration_sec,
                "throughput_tokens_per_sec": result.throughput_tokens_per_sec,
                "avg_latency_ms": result.avg_latency_ms,
                "p50_latency_ms": result.p50_latency_ms,
                "p95_latency_ms": result.p95_latency_ms,
                "p99_latency_ms": result.p99_latency_ms,
            }

        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to {output_file}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Burst test for attention operator replicas")
    parser.add_argument("--model", required=True, help="Path to model directory")
    parser.add_argument("--output", help="Output file for results (JSON)")
    parser.add_argument("--quick", action="store_true", help="Run quick test with fewer iterations")

    args = parser.parse_args()

    if args.quick:
        # Quick test mode
        simulator = BurstTestSimulator(args.model)
        result = simulator.run_burst_test(
            num_bursts=1,
            requests_per_burst=2,
            prompt_length=10,
            max_tokens=5,
            burst_interval_sec=0.1,
            use_replicas=True,
            replica_config={
                "op_replica_configs": {"attention": 1},
                "replica_devices": [0],
            },
        )
        print("Quick test completed successfully!")
        return

    run_comparison_test(args.model, args.output)


if __name__ == "__main__":
    main()


