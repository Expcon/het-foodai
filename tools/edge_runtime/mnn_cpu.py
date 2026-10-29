"""Run a prepared float32 RGB NCHW tensor using a matching explicit contract."""
import argparse
import json
from pathlib import Path


def validate_contract(contract):
    if contract.get("task") not in {"classify", "detect"}:
        raise ValueError("task must explicitly be classify or detect")
    if contract.get("input_layout") != "NCHW" or contract.get("input_dtype") != "float32":
        raise ValueError("this example only accepts float32 NCHW")
    shape = contract.get("input_shape", [])
    if len(shape) != 4 or shape[:2] != [1, 3] or any(type(n) is not int or n <= 0 for n in shape):
        raise ValueError("input_shape must be positive [1, 3, height, width]")
    expected = "classification-probabilities" if contract["task"] == "classify" else "yolo11-detect-raw"
    if contract.get("output_kind") != expected:
        raise ValueError("unsupported output parser: supply a model-specific contract")
    if type(contract.get("num_classes")) is not int or contract["num_classes"] <= 0:
        raise ValueError("num_classes must be positive")
    output_shape = contract.get("output_shape", [])
    nc = contract["num_classes"]
    valid_output = output_shape == [1, nc] if contract["task"] == "classify" else len(output_shape) == 3 and output_shape[:2] == [1, 4 + nc]
    if not valid_output or any(type(n) is not int or n <= 0 for n in output_shape):
        raise ValueError("output_shape must match the task-specific parser")
    return contract


def validate_input(array, contract):
    import numpy as np

    validate_contract(contract)
    if list(array.shape) != contract["input_shape"] or array.dtype != np.float32:
        raise ValueError("prepared tensor shape/dtype does not match contract")
    if not np.isfinite(array).all() or array.min() < 0 or array.max() > 1:
        raise ValueError("this specimen expects finite RGB values in [0, 1]")
    return np.ascontiguousarray(array)


def validate_output(array, contract):
    import numpy as np

    validate_contract(contract)
    if list(array.shape) != contract["output_shape"] or array.dtype != np.float32 or not np.isfinite(array).all():
        raise ValueError("output shape/values do not match the model-specific contract")
    nc = contract["num_classes"]
    if contract["task"] == "classify":
        if array.shape != (1, nc) or array.min() < -1e-5 or array.max() > 1.00001:
            raise ValueError("expected [1, num_classes] probabilities")
        if not np.allclose(array.sum(axis=1), 1, atol=0.01):
            raise ValueError("classification probabilities do not sum to one")
    elif array.ndim != 3 or array.shape[:2] != (1, 4 + nc):
        raise ValueError("expected YOLO11 raw [1, 4 + num_classes, anchors], no NMS")


def run_mnn(model, array, contract, threads=1):
    """Use CPU high precision; never import MNN.tools (its logger accesses cloud)."""
    import MNN
    import numpy as np

    array = validate_input(array, contract)
    if threads < 1:
        raise ValueError("threads must be positive")
    manager = MNN.nn.create_runtime_manager(({"backend": "CPU", "precision": "high", "numThread": threads},))
    module = MNN.nn.load_module_from_file(str(model), [], [], runtime_manager=manager, rearrange=True)
    info = module.get_info()
    if len(info["inputNames"]) != 1 or len(info["outputNames"]) != 1:
        raise ValueError("this example requires exactly one model input and output")
    var = MNN.expr.const(array, array.shape, MNN.expr.NCHW)
    outputs = module.onForward([var])
    result = np.asarray(outputs[0].read()).copy()
    validate_output(result, contract)
    return result


def main():
    from tools.edge_export.offline_guard import install

    install()
    import numpy as np

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tensor", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="raw output .npy (no invented labels or boxes)")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    contract = validate_contract(json.loads(args.contract.read_text(encoding="utf-8")))
    result = run_mnn(args.model, np.load(args.tensor, allow_pickle=False), contract, args.threads)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, result, allow_pickle=False)
    print(json.dumps({"task": contract["task"], "backend": "CPU", "precision": "high", "output_shape": list(result.shape), "output_dtype": str(result.dtype)}))


if __name__ == "__main__":
    main()
