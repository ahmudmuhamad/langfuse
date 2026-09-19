# 02 - Instrumentation

Methods of instrumenting your code with Langfuse:
- **`01_decorator_observe.py`**: The `@observe()` decorator pattern (the most idiomatic Python approach).
- **`02_llm_integration.py`**: Automated tracing wrappers (e.g., `from langfuse.openai import openai`).
- **`03_low_level_sdk.py`**: Manual SDK tracing (`langfuse.trace()`, `trace.span()`).
