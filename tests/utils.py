import unittest


def require_fasttext(test_case):
    try:
        import fasttext  # noqa: F401
    except ImportError:
        test_case = unittest.skip("test requires fasttext")(test_case)
    return test_case

def require_hf_hub(test_case):
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        test_case = unittest.skip("test requires huggingface_hub")(test_case)
    return test_case

def require_openai(test_case):
    try:
        from openai import OpenAI  # noqa: F401
    except ImportError:
        test_case = unittest.skip("test requires openai")(test_case)
    return test_case

def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'