#!/usr/bin/env python3
"""Inspect a Torch/Triton/TileLang reference file before baseline generation.

The output is intentionally descriptive rather than authoritative. It gives the
agent enough structure to generate a conservative HIP baseline and to know when
manual repair is required.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path
from typing import Any


REFERENCE_NAME_PRIORITY = [
    "reference",
    "torch_ref",
    "golden",
    "ref",
    "forward",
    "model_forward",
    "run",
]


def _read(path: str) -> str:
    return Path(path).read_text(encoding="utf-8", errors="ignore")


def _json_loads(text: str) -> dict[str, int]:
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"--dims must be valid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise SystemExit("--dims must be a JSON object")
    out: dict[str, int] = {}
    for key, value in raw.items():
        try:
            out[str(key)] = int(value)
        except (TypeError, ValueError) as exc:
            raise SystemExit(f"--dims value for {key!r} must be int-like") from exc
    return out


def _decorator_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _decorator_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    if isinstance(node, ast.Call):
        return _decorator_name(node.func)
    return ""


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _call_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return ""


def _target_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Subscript):
        return _target_name(node.value)
    if isinstance(node, ast.Attribute):
        return _target_name(node.value)
    return ""


def _node_text(source: str, node: ast.AST) -> str:
    try:
        return ast.get_source_segment(source, node) or ""
    except Exception:
        return ""


def _literal_number(node: ast.AST) -> float | int | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = _literal_number(node.operand)
        if isinstance(inner, (int, float)):
            return -inner
    return None


class FunctionInspector(ast.NodeVisitor):
    def __init__(self, source: str, fn: ast.FunctionDef | ast.AsyncFunctionDef):
        self.source = source
        self.fn = fn
        self.calls: list[str] = []
        self.copy_targets: list[str] = []
        self.return_exprs: list[str] = []
        self.has_return_value = False
        self.binops: list[str] = []
        self.constants: list[float | int] = []
        self.subscripts: list[str] = []

    def visit_Call(self, node: ast.Call) -> Any:
        name = _call_name(node.func)
        if name:
            self.calls.append(name)
        if name.endswith(".copy_") and isinstance(node.func, ast.Attribute):
            target = _target_name(node.func.value)
            if target:
                self.copy_targets.append(target)
        self.generic_visit(node)

    def visit_Return(self, node: ast.Return) -> Any:
        if node.value is not None:
            self.has_return_value = True
            text = _node_text(self.source, node.value).strip()
            if text:
                self.return_exprs.append(text)
        self.generic_visit(node)

    def visit_BinOp(self, node: ast.BinOp) -> Any:
        self.binops.append(type(node.op).__name__)
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> Any:
        if isinstance(node.value, (int, float)):
            self.constants.append(node.value)

    def visit_Subscript(self, node: ast.Subscript) -> Any:
        text = _node_text(self.source, node).strip()
        if text:
            self.subscripts.append(text)
        self.generic_visit(node)


def _function_info(source: str, fn: ast.FunctionDef | ast.AsyncFunctionDef) -> dict[str, Any]:
    inspector = FunctionInspector(source, fn)
    inspector.visit(fn)
    decorators = [_decorator_name(d) for d in fn.decorator_list]
    args = [a.arg for a in fn.args.args]
    return {
        "name": fn.name,
        "lineno": fn.lineno,
        "args": args,
        "decorators": [d for d in decorators if d],
        "is_jit_kernel": any(
            key in d.lower()
            for d in decorators
            for key in ("triton.jit", "jit", "tilelang", "tl.jit")
        ),
        "calls": sorted(set(inspector.calls)),
        "copy_targets": sorted(set(inspector.copy_targets)),
        "has_return_value": inspector.has_return_value,
        "return_exprs": inspector.return_exprs[:3],
        "binops": sorted(set(inspector.binops)),
        "constants": sorted(set(inspector.constants), key=lambda x: float(x))[:16],
        "subscripts": inspector.subscripts[:16],
    }


def _imports(tree: ast.AST) -> list[str]:
    out: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                out.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                out.append(node.module)
    return sorted(set(out))


def _choose_reference(functions: list[dict[str, Any]]) -> dict[str, Any] | None:
    plain = [f for f in functions if not f.get("is_jit_kernel")]
    by_name = {f["name"]: f for f in plain}
    for name in REFERENCE_NAME_PRIORITY:
        if name in by_name:
            return by_name[name]
    with_outputs = [f for f in plain if f.get("copy_targets") or f.get("has_return_value")]
    if with_outputs:
        return sorted(with_outputs, key=lambda f: f["lineno"])[0]
    return plain[0] if plain else None


def _classify(imports: list[str], functions: list[dict[str, Any]], dims: dict[str, int]) -> dict[str, Any]:
    text_calls = " ".join(" ".join(f.get("calls", [])) for f in functions).lower()
    text_decorators = " ".join(" ".join(f.get("decorators", [])) for f in functions).lower()
    dim_keys = {k.upper() for k in dims}

    reasons: list[str] = []
    frameworks: list[str] = []
    imports_text = " ".join(imports).lower()
    if "torch" in imports_text:
        frameworks.append("torch")
    if "triton" in imports_text or "triton" in text_decorators:
        frameworks.append("triton")
    if "tilelang" in imports_text or "tilelang" in text_decorators:
        frameworks.append("tilelang")

    op = "unknown"
    if {"M", "N", "K"}.issubset(dim_keys) or any(x in text_calls for x in ("matmul", "mm", "bmm", "tl.dot", "dot")):
        op = "matmul"
        reasons.append("matmul/dot call or M,N,K dims detected")
    elif any(x in text_calls for x in ("softmax", "sum", "mean", "amax", "max", "norm", "layer_norm")):
        op = "reduction"
        reasons.append("reduction-like call detected")
    elif "N" in dim_keys:
        op = "elementwise"
        reasons.append("single flat N dimension detected")

    if not frameworks:
        frameworks.append("python")
    return {"op_kind": op, "frameworks": frameworks, "reasons": reasons}


def inspect_ref(ref: str, dims: dict[str, int]) -> dict[str, Any]:
    source = _read(ref)
    try:
        tree = ast.parse(source, filename=ref)
    except SyntaxError as exc:
        raise SystemExit(f"cannot parse {ref}: {exc}") from exc

    functions = [
        _function_info(source, node)
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    imports = _imports(tree)
    chosen = _choose_reference(functions)
    classification = _classify(imports, functions, dims)

    tensor_args: list[str] = []
    dim_args: list[str] = []
    output_args: list[str] = []
    return_style = "unknown"
    if chosen:
        dim_names = set(dims)
        output_args = list(chosen.get("copy_targets", []))
        tensor_args = [a for a in chosen.get("args", []) if a not in dim_names]
        dim_args = [a for a in chosen.get("args", []) if a in dim_names]
        if chosen.get("has_return_value"):
            return_style = "returns_tensor"
        elif output_args:
            return_style = "inplace"

    unsupported: list[str] = []
    if chosen is None:
        unsupported.append("no plain Python reference candidate found")
    elif chosen.get("is_jit_kernel"):
        unsupported.append("chosen function appears to be a JIT kernel, not a Python oracle")
    if classification["op_kind"] == "unknown":
        unsupported.append("operation family is unknown; generated kernel may be placeholder")
    if not tensor_args and chosen:
        unsupported.append("no tensor-like function arguments detected")

    return {
        "ref_file": os.path.abspath(ref),
        "dims": dims,
        "imports": imports,
        "functions": functions,
        "chosen_reference": chosen,
        "signature_hint": {
            "tensor_args": tensor_args,
            "dim_args": dim_args,
            "output_args": output_args,
            "return_style": return_style,
        },
        "classification": classification,
        "unsupported_assumptions": unsupported,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a Torch/Triton/TileLang reference file")
    parser.add_argument("--ref", required=True)
    parser.add_argument("--dims", default="", help="JSON shape/dim object")
    parser.add_argument("--dims-file", default="", help="Read shape/dim JSON from a file")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    if bool(args.dims) == bool(args.dims_file):
        raise SystemExit("pass exactly one of --dims or --dims-file")
    dims_text = Path(args.dims_file).read_text(encoding="utf-8") if args.dims_file else args.dims
    report = inspect_ref(args.ref, _json_loads(dims_text))
    payload = json.dumps(report, indent=2, ensure_ascii=False)
    print(payload)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
