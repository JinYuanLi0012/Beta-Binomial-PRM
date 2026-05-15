#!/usr/bin/env python3
"""Start a vLLM OpenAI-compatible API server for answer judging."""

import argparse
import os
import subprocess
import sys


os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


def main():
    parser = argparse.ArgumentParser(description="Start a vLLM API server.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-32B-Instruct")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8888)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.95)
    parser.add_argument("--served-model-name", type=str, default="Qwen2.5-32B-Instruct")
    parser.add_argument("--max-model-len", type=int, default=None)
    args = parser.parse_args()

    vllm_python = os.environ.get("VLLM_PYTHON", "").strip() or sys.executable
    cmd = [
        vllm_python,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        args.model,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--served-model-name",
        args.served_model_name,
        "--trust-remote-code",
    ]
    if args.max_model_len:
        cmd.extend(["--max-model-len", str(args.max_model_len)])

    print("Starting vLLM API server...")
    print(f"Model: {args.model}")
    print(f"Endpoint: http://{args.host}:{args.port}/v1")
    print(f"Served model name: {args.served_model_name}")
    print(f"Command: {' '.join(cmd)}")

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nServer stopped.")
    except subprocess.CalledProcessError as exc:
        print(f"Failed to start vLLM server: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
