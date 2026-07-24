from __future__ import annotations

import ast
import math
import operator
from typing import Callable, Dict, Type

from omegaconf import OmegaConf


_MAX_EXPRESSION_LENGTH = 256
_MAX_AST_NODES = 64
_MAX_ABSOLUTE_VALUE = 1_000_000_000_000
_BINARY_OPERATORS: Dict[Type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
}
_UNARY_OPERATORS: Dict[Type[ast.unaryop], Callable[[float], float]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _validate_result(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Arithmetic resolver results must be numeric.")
    if not math.isfinite(float(value)):
        raise ValueError("Arithmetic resolver results must be finite.")
    if abs(value) > _MAX_ABSOLUTE_VALUE:
        raise ValueError(
            f"Arithmetic resolver result exceeds {_MAX_ABSOLUTE_VALUE}."
        )
    return value


def safe_arithmetic_eval(expression: str):
    """Evaluate a small numeric expression without executing Python code."""
    if not isinstance(expression, str):
        raise TypeError("Arithmetic resolver input must be a string.")
    if len(expression) > _MAX_EXPRESSION_LENGTH:
        raise ValueError(
            f"Arithmetic resolver expression exceeds {_MAX_EXPRESSION_LENGTH} characters."
        )
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Invalid arithmetic expression: {expression!r}.") from exc
    if sum(1 for _ in ast.walk(tree)) > _MAX_AST_NODES:
        raise ValueError(
            f"Arithmetic resolver expression exceeds {_MAX_AST_NODES} syntax nodes."
        )

    def evaluate(node):
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant):
            return _validate_result(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
            left = evaluate(node.left)
            right = evaluate(node.right)
            try:
                return _validate_result(_BINARY_OPERATORS[type(node.op)](left, right))
            except ZeroDivisionError as exc:
                raise ValueError("Arithmetic resolver division by zero.") from exc
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
            return _validate_result(_UNARY_OPERATORS[type(node.op)](evaluate(node.operand)))
        raise ValueError(
            f"Unsupported syntax in arithmetic resolver: {type(node).__name__}."
        )

    return evaluate(tree)


def register_safe_omegaconf_resolvers() -> None:
    """Register compatibility resolver names using non-executable implementations."""
    OmegaConf.register_new_resolver(
        "eval",
        safe_arithmetic_eval,
        replace=True,
        use_cache=False,
    )
