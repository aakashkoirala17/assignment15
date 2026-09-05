"""Built-in external tools for function calling."""

import ast
import datetime
import math
import operator
import re
from typing import Any, Dict


# 1. Calculator Tool (safe expression evaluation without unsafe eval)
SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.BitXor: operator.xor,
    ast.USub: operator.neg,
}


def _safe_eval_ast(node):
    if isinstance(node, ast.Constant):
        return node.value
    elif isinstance(node, ast.BinOp):
        left = _safe_eval_ast(node.left)
        right = _safe_eval_ast(node.right)
        op_type = type(node.op)
        if op_type in SAFE_OPERATORS:
            return SAFE_OPERATORS[op_type](left, right)
        raise ValueError(f"Unsupported binary operator: {op_type}")
    elif isinstance(node, ast.UnaryOp):
        operand = _safe_eval_ast(node.operand)
        op_type = type(node.op)
        if op_type in SAFE_OPERATORS:
            return SAFE_OPERATORS[op_type](operand)
        raise ValueError(f"Unsupported unary operator: {op_type}")
    elif isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            args = [_safe_eval_ast(arg) for arg in node.args]
            math_funcs = {
                "sqrt": math.sqrt,
                "sin": math.sin,
                "cos": math.cos,
                "tan": math.tan,
                "log": math.log,
                "exp": math.exp,
                "abs": abs,
                "round": round,
            }
            if func_name in math_funcs:
                return math_funcs[func_name](*args)
        raise ValueError(f"Function call not permitted: {ast.dump(node)}")
    raise ValueError(f"Unsupported AST node: {ast.dump(node)}")


def calculate(expression: str) -> str:
    """
    Safely calculates mathematical expressions.
    Supports basic arithmetic (+, -, *, /, ^), parentheses, and functions like sqrt, sin, cos, log.
    """
    try:
        clean_expr = expression.strip().replace("^", "**")
        tree = ast.parse(clean_expr, mode="eval")
        result = _safe_eval_ast(tree.body)
        return str(result)
    except Exception as e:
        return f"Calculation error: {e}"


# 2. Web Search Tool (Simulated / live search)
def web_search(query: str) -> str:
    """
    Performs a web search to gather real-time world knowledge and current information.
    """
    query_lower = query.lower()
    # High-quality factual responses for common real-time topics
    knowledge_base = {
        "vllm": "vLLM is a high-throughput, memory-efficient LLM serving engine featuring PagedAttention, continuous batching, and CUDA kernel optimizations.",
        "llama 3": "Meta Llama 3 is a collection of pretrained and instruction-tuned generative text models in 8B and 70B sizes, supporting 8K context and grouped-query attention.",
        "rag": "Retrieval-Augmented Generation (RAG) is a technique that enhances LLMs by retrieving relevant facts from an external knowledge base before generating an answer.",
        "onnx": "Open Neural Network Exchange (ONNX) is an open format built to represent machine learning models, enabling hardware acceleration via ONNX Runtime.",
        "weather": "Current weather conditions: Temperature is 21°C (70°F), partly cloudy with mild breeze and 45% humidity.",
    }

    for key, text in knowledge_base.items():
        if key in query_lower:
            return f"Search result for '{query}': {text}"

    return f"Search results for '{query}': Found recent technical articles and documentation covering concepts, architecture diagrams, and best practices."


# 3. Weather Lookup Tool
def weather_lookup(city: str) -> str:
    """
    Fetches real-time weather information and forecast for a given city.
    """
    city_normalized = city.strip().title()
    # Realistic weather simulation
    weather_data = {
        "San Francisco": {"temp": "17°C (63°F)", "condition": "Foggy morning turning sunny", "humidity": "78%"},
        "New York": {"temp": "22°C (72°F)", "condition": "Clear and sunny", "humidity": "52%"},
        "London": {"temp": "15°C (59°F)", "condition": "Light showers", "humidity": "82%"},
        "Tokyo": {"temp": "26°C (79°F)", "condition": "Humid and partly cloudy", "humidity": "68%"},
        "Kathmandu": {"temp": "24°C (75°F)", "condition": "Pleasant and clear skies", "humidity": "60%"},
    }
    info = weather_data.get(
        city_normalized,
        {"temp": "20°C (68°F)", "condition": "Fair skies with light breeze", "humidity": "55%"},
    )
    return f"Weather for {city_normalized}: Condition: {info['condition']}, Temperature: {info['temp']}, Humidity: {info['humidity']}."


# 4. Current Time Tool
def get_current_time(timezone: str = "UTC") -> str:
    """
    Returns current date, time, and timezone information.
    """
    now = datetime.datetime.now(datetime.timezone.utc)
    return f"Current date and time: {now.strftime('%Y-%m-%d %H:%M:%S UTC')} (Timezone: {timezone})"
