#!/usr/bin/env python3

"""test_merge_tokenizer.py
"""

import os
import sys
import onnxruntime_extensions as extensions
from onnx import load_model, save_model
from onnx.checker import check_model
from onnx.compose import merge_graphs
from onnx.external_data_helper import load_external_data_for_model
from onnx.helper import make_model
from transformers import AutoTokenizer

MODEL_PATH = "./exported-float16"


def merge_tokenizer(model_path: str) -> str:
    """merge a tokenizer to a model.
    """

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = load_model(
        model_path + "/model.onnx",
        load_external_data=False,
    )
    graph = model.graph

    pre_model, post_model = extensions.gen_processing_models(
        tokenizer,
        pre_kwargs={},
        post_kwargs={},
        opset=model.opset_import[0].version
    )

    pre_io_map = [
        ("input_ids", "input_ids"),
        ("attention_mask", "attention_mask"),
    ]
    graph = merge_graphs(pre_model.graph, graph, pre_io_map)

    post_io_map = [
        ("logits", "ids"),
    ]
    graph = merge_graphs(graph, post_model.graph, post_io_map)

    opset_imports = model.opset_import
    opset_imports.extend(pre_model.opset_import)
    # opset_imports.extend(post_model.opset_import)
    merged_model = make_model(
        graph,
        opset_imports=opset_imports,
        functions=model.functions,
        ir_version=model.ir_version,
    )
    load_external_data_for_model(merged_model, model_path)

    merged_model_name = "merged_model.onnx"
    if os.path.exists(model_path + "/" + merged_model_name + "_data"):
        os.remove(model_path + "/" + merged_model_name + "_data")
    save_model(
        merged_model, model_path + "/" + merged_model_name,
        save_as_external_data=True,
        location=merged_model_name + "_data",
    )

    return merged_model_name


def main() -> int:
    """main
    """

    merge_tokenizer(MODEL_PATH)

    # # The merged model could not be checked.
    # check_model(
    #     MODEL + "/merged_model.onnx",
    #     full_check=True,
    # )

    return 0


if __name__ == "__main__":
    sys.exit(main())
