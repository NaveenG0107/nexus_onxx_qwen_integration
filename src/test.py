import os
import sys

# Ensure root directory is on Python path
sys.path.insert(0, os.path.abspath("."))

from src.generation.service import get_model_manager

print("Initializing Dual Model Manager...")
manager = get_model_manager()

print("\n" + "=" * 50)
print("1. TESTING WITH CUSTOM ONNX MODEL (use_custom_onnx=True)")
print("=" * 50)

# Custom test 1: fed query
q1 = "Where is Mitrahsoft second branch is located?"
ans1, model1 = manager.generate(prompt=q1, max_tokens=30, use_custom_onnx=True)
print(f"Query: '{q1}'\nModel: {model1}\nOutput: {ans1}\n")

# Custom test 2: variation of fed query
q2 = "what is mitrah soft"
ans2, model2 = manager.generate(prompt=q2, max_tokens=30, use_custom_onnx=True)
print(f"Query: '{q2}'\nModel: {model2}\nOutput: {ans2}\n")

# Custom test 3: un-fed query (should return Not found)
q3 = "Where is hcl is located?"
ans3, model3 = manager.generate(prompt=q3, max_tokens=30, use_custom_onnx=True)
print(f"Query: '{q3}'\nModel: {model3}\nOutput: {ans3}\n")


print("=" * 50)
print("2. TESTING WITH BASE QWEN ONNX MODEL (use_custom_onnx=False)")
print("=" * 50)

q4 = "Explain artificial intelligence in one brief sentence."
ans4, model4 = manager.generate(prompt=q4, max_tokens=60, use_custom_onnx=False)
print(f"Query: '{q4}'\nModel: {model4}\nOutput: {ans4}\n")
