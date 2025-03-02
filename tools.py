# tools.py

"""tools
"""


import os
import torch_directml
from onnxruntime import SessionOptions, GraphOptimizationLevel
from optimum.onnxruntime import ORTModel
from transformers import AutoTokenizer


def test_on_dml(
    model_name: str,
    optimized_dir: str,
):
    """"""

    os.makedirs(optimized_dir, exist_ok=True)

    session_options = SessionOptions()
    session_options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
    session_options.optimized_model_filepath = optimized_dir + "/model_optimized.onnx"

    device = torch_directml.device()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = ORTModel.from_pretrained(
        model_name,
        provider="DmlExecutionProvider",
        session_options=session_options,
    ).to(device)

    inputs = tokenizer(
        "The weather is always wonderful",
        return_tensors="pt",
    ).to(model.device)
    tokens = model.generate(
        **inputs,
        max_new_tokens=64,
        temperature=0.75,
        top_p=0.95,
        do_sample=True,
    )
    print(tokenizer.decode(tokens[0], skip_special_tokens=True))
