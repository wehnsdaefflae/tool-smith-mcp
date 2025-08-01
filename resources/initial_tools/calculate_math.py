"""Basic mathematical calculation tool."""

import ast
import operator
from typing import Union


def calculate_math(expression: str) -> float:
    """Evaluate a mathematical expression safely.
    
    Args:
        expression: Mathematical expression to evaluate (e.g., "2 + 3 * 4")
        
    Returns:
        Result of the calculation
    """
    # Safe evaluation of mathematical expressions
    allowed_operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
    }
    
    def eval_node(node: ast.AST) -> Union[int, float]:
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.BinOp):
            left = eval_node(node.left)
            right = eval_node(node.right)
            op = allowed_operators[type(node.op)]
            return op(left, right)
        elif isinstance(node, ast.UnaryOp):
            operand = eval_node(node.operand)
            op = allowed_operators[type(node.op)]
            return op(operand)
        else:
            raise ValueError(f"Unsupported operation: {type(node)}")
    
    tree = ast.parse(expression, mode='eval')
    return float(eval_node(tree.body))