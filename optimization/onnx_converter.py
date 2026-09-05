"""Model Optimization: ONNX Conversion, Quantization, and Benchmarking."""

import logging
import os
import time
from typing import Dict, List, Tuple
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def convert_and_benchmark(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    output_dir: str = "./onnx_models",
    num_runs: int = 50,
) -> Dict[str, any]:
    """
    Exports transformer model to ONNX format, quantizes to INT8,
    and performs comparative latency and throughput benchmarking.
    """
    os.makedirs(output_dir, exist_ok=True)
    onnx_path = os.path.join(output_dir, "model.onnx")
    quant_path = os.path.join(output_dir, "model_quantized.onnx")

    logger.info("Initializing model conversion for: %s", model_name)

    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    torch_model = AutoModel.from_pretrained(model_name)
    torch_model.eval()

    # 1. Export to ONNX
    dummy_text = "Optimizing AI assistant latency and memory throughput with ONNX Runtime."
    inputs = tokenizer(
        dummy_text,
        return_tensors="pt",
        padding="max_length",
        max_length=64,
        truncation=True,
    )

    input_names = ["input_ids", "attention_mask"]
    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "last_hidden_state": {0: "batch_size", 1: "sequence_length"},
    }

    dummy_inputs = (inputs["input_ids"], inputs["attention_mask"])
    if "token_type_ids" in inputs:
        dummy_inputs = (
            inputs["input_ids"],
            inputs["attention_mask"],
            inputs["token_type_ids"],
        )
        input_names.append("token_type_ids")
        dynamic_axes["token_type_ids"] = {0: "batch_size", 1: "sequence_length"}

    logger.info("Exporting PyTorch model to ONNX format...")
    torch.onnx.export(
        torch_model,
        dummy_inputs,
        onnx_path,
        input_names=input_names,
        output_names=["last_hidden_state"],
        dynamic_axes=dynamic_axes,
        opset_version=14,
        do_constant_folding=True,
    )
    logger.info("Exported ONNX model to: %s", onnx_path)

    # 2. INT8 Quantization using ONNX Runtime
    logger.info("Applying dynamic INT8 quantization...")
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        quantize_dynamic(
            model_input=onnx_path,
            model_output=quant_path,
            weight_type=QuantType.QInt8,
        )
        logger.info("Quantized model saved to: %s", quant_path)
    except Exception as q_err:
        logger.warning("Quantization error: %s; continuing with FP32 ONNX", q_err)
        quant_path = onnx_path

    # 3. Benchmark PyTorch vs ONNX Runtime
    import onnxruntime as ort

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    sess_fp32 = ort.InferenceSession(onnx_path, sess_options)
    sess_quant = (
        ort.InferenceSession(quant_path, sess_options)
        if os.path.exists(quant_path)
        else sess_fp32
    )

    test_sentences = [
        "Retrieval Augmented Generation with ChromaDB vector search.",
        "Production AI systems require rate limiting and exponential retries.",
        "Serving open-source models using vLLM continuous batching.",
    ]
    encoded = tokenizer(
        test_sentences, return_tensors="np", padding=True, truncation=True, max_length=64
    )
    ort_inputs = {
        "input_ids": encoded["input_ids"].astype(np.int64),
        "attention_mask": encoded["attention_mask"].astype(np.int64),
    }
    if "token_type_ids" in encoded and "token_type_ids" in input_names:
        ort_inputs["token_type_ids"] = encoded["token_type_ids"].astype(np.int64)

    pt_inputs = {k: torch.tensor(v) for k, v in ort_inputs.items()}

    # Warmup
    for _ in range(5):
        with torch.no_grad():
            torch_model(**pt_inputs)
        sess_fp32.run(None, ort_inputs)
        sess_quant.run(None, ort_inputs)

    # Benchmark PyTorch
    pt_latencies = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        with torch.no_grad():
            torch_model(**pt_inputs)
        pt_latencies.append((time.perf_counter() - t0) * 1000)

    # Benchmark ONNX Runtime FP32
    ort_latencies = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        sess_fp32.run(None, ort_inputs)
        ort_latencies.append((time.perf_counter() - t0) * 1000)

    # Benchmark ONNX Runtime INT8 Quantized
    quant_latencies = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        sess_quant.run(None, ort_inputs)
        quant_latencies.append((time.perf_counter() - t0) * 1000)

    pt_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    quant_size_mb = (
        os.path.getsize(quant_path) / (1024 * 1024)
        if os.path.exists(quant_path)
        else pt_size_mb
    )

    results = {
        "model_name": model_name,
        "batch_size": len(test_sentences),
        "benchmark_runs": num_runs,
        "pytorch_fp32": {
            "mean_ms": round(float(np.mean(pt_latencies)), 2),
            "p50_ms": round(float(np.percentile(pt_latencies, 50)), 2),
            "p95_ms": round(float(np.percentile(pt_latencies, 95)), 2),
            "p99_ms": round(float(np.percentile(pt_latencies, 99)), 2),
        },
        "onnxruntime_fp32": {
            "mean_ms": round(float(np.mean(ort_latencies)), 2),
            "p50_ms": round(float(np.percentile(ort_latencies, 50)), 2),
            "p95_ms": round(float(np.percentile(ort_latencies, 95)), 2),
            "speedup_vs_pytorch": round(
                float(np.mean(pt_latencies) / np.mean(ort_latencies)), 2
            ),
        },
        "onnxruntime_int8": {
            "mean_ms": round(float(np.mean(quant_latencies)), 2),
            "p50_ms": round(float(np.percentile(quant_latencies, 50)), 2),
            "p95_ms": round(float(np.percentile(quant_latencies, 95)), 2),
            "speedup_vs_pytorch": round(
                float(np.mean(pt_latencies) / np.mean(quant_latencies)), 2
            ),
            "size_reduction_ratio": round(pt_size_mb / max(0.1, quant_size_mb), 2),
            "original_onnx_mb": round(pt_size_mb, 2),
            "quantized_onnx_mb": round(quant_size_mb, 2),
        },
    }

    logger.info("Benchmark complete: %s", results)
    return results


if __name__ == "__main__":
    convert_and_benchmark()
