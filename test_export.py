#!/usr/bin/env python3

"""test_export.py

This script exports a StableLM model into ONNX with optional float16 conversion.
"""

import os
import sys
import onnx
from onnxconverter_common import float16
from onnxruntime import SessionOptions, GraphOptimizationLevel
from optimum.exporters.onnx import main_export
from optimum.onnxruntime import ORTModelForCausalLM

EXPORTED = "./exported"
WITH_FLOAT16 = True

if WITH_FLOAT16:
    EXPORTED += "-float16"


def main() -> int:
    """main
    """

    if os.path.exists("./exported-intermediate/model.onnx_data"):
        os.remove("./exported-intermediate/model.onnx_data")
    main_export(
        # "stabilityai/stablelm-3b-4e1t",
        "ibm-granite/granite-3b-code-instruct-128k",
        output="./exported-intermediate",
        trust_remote_code=True,
        # dtype="fp16",
        # device="dml",
    )

    if WITH_FLOAT16:
        # Conversion to float16 by Optimum had problems with DirectML.
        model = onnx.load_model("./exported-intermediate/model.onnx")
        model = float16.convert_float_to_float16(model, disable_shape_infer=True)

        os.remove("./exported-intermediate/model.onnx_data")
        onnx.save_model(
            model, "./exported-intermediate/model.onnx",
            save_as_external_data=True,
            location="model.onnx_data",
        )

    session_options = SessionOptions()
    session_options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_BASIC
    model = ORTModelForCausalLM.from_pretrained(
        "./exported-intermediate",
        provider="DmlExecutionProvider",
        session_options=session_options,
    )

    if os.path.exists(EXPORTED + "/model.onnx_data"):
        os.remove(EXPORTED + "/model.onnx_data")
    model.save_pretrained(EXPORTED)

    return 0


if __name__ == "__main__":
    sys.exit(main())
