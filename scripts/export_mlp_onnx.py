#!/usr/bin/env python3
"""Convert an MLP saved by train_mlp.py (NumPy .npz) into an ONNX graph.

The graph takes the raw feature vector `phi` (1, F) as float32, applies the
standardisation (mu, sigma) and the ReLU MLP, and outputs (1, 2). Running it
in ONNX Runtime next to the organisers' GRU keeps the per-row work in one
engine (see docs/NEUTRINO_SOLUTION.md, step 3).

    python scripts/export_mlp_onnx.py solution/mlp.npz solution/mlp.onnx
"""
from __future__ import annotations

import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def main(src: str, dst: str) -> None:
    d = np.load(src)
    keep = d["keep"] if "keep" in d else None
    n_layers = int(d["n_layers"])
    f_in = int(d["W0"].shape[0])
    inits = [
        numpy_helper.from_array(d["mu"].astype(np.float32), "mu"),
        numpy_helper.from_array((1.0 / d["sigma"].astype(np.float64)).astype(np.float32), "inv_sigma"),
    ]
    nodes = [
        helper.make_node("Sub", ["phi", "mu"], ["z0"]),
        helper.make_node("Mul", ["z0", "inv_sigma"], ["h0"]),
    ]
    cur = "h0"
    for i in range(n_layers):
        inits.append(numpy_helper.from_array(np.ascontiguousarray(d[f"W{i}"], dtype=np.float32), f"W{i}"))
        inits.append(numpy_helper.from_array(d[f"b{i}"].astype(np.float32), f"b{i}"))
        nodes.append(helper.make_node("MatMul", [cur, f"W{i}"], [f"m{i}"]))
        out = "prediction" if i == n_layers - 1 else f"a{i}"
        nodes.append(helper.make_node("Add", [f"m{i}", f"b{i}"], [out if i == n_layers - 1 else f"s{i}"]))
        if i != n_layers - 1:
            nodes.append(helper.make_node("Relu", [f"s{i}"], [out]))
        cur = out
    graph = helper.make_graph(
        nodes, "neutrino_mlp",
        [helper.make_tensor_value_info("phi", TensorProto.FLOAT, [1, f_in])],
        [helper.make_tensor_value_info("prediction", TensorProto.FLOAT, [1, 2])],
        initializer=inits,
    )
    model = helper.make_model(graph, producer_name="neutrino-wunder", opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, dst)
    print(f"wrote {dst}: {f_in} inputs, {n_layers} linear layers, keep={'all' if keep is None else len(keep)}")
    if keep is not None and len(keep) != f_in:
        raise SystemExit("keep/W0 mismatch")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
