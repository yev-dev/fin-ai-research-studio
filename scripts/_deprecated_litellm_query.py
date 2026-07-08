
import asyncio
import os

from dotenv import load_dotenv
from litellm import acompletion, completion


def _normalize_ollama_endpoint(endpoint: str) -> str:
    endpoint = endpoint.rstrip("/")
    if endpoint.endswith("/v1"):
        return endpoint[: -len("/v1")]
    return endpoint


def _build_completion_args(api_host: str, question: str, system_prompt: str) -> dict:
    if api_host == "github":
        return {
            "model": os.getenv("GITHUB_MODEL", "openai/gpt-4o"),
            "api_base": os.getenv("GITHUB_ENDPOINT", "https://models.github.ai/inference"),
            "api_key": os.environ["GITHUB_TOKEN"],
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
        }

    if api_host == "ollama":
        return {
            "model": f"ollama/{os.getenv('OLLAMA_MODEL', 'llama3.1')}",
            "api_base": _normalize_ollama_endpoint(os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434")),
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
        }

    raise ValueError("API_HOST must be either 'github' or 'ollama'.")


def main() -> None:
    load_dotenv(override=True)

    api_host = os.getenv("API_HOST", "github")
    question = "What is the most important financial metric for a company?"
    system_prompt = "You're a financial analysis agent. Answer questions correctly. My pension depends on it."
    request_args = _build_completion_args(api_host=api_host, question=question, system_prompt=system_prompt)

    response = completion(**request_args)
    print(response.choices[0].message.content)


async def async_main() -> None:
    """Asynchronous variant of the same LiteLLM request flow."""
    load_dotenv(override=True)

    api_host = os.getenv("API_HOST", "github")
    question = "What is the most important financial metric for a company?"
    system_prompt = "You're a financial analysis agent. Answer questions correctly. My pension depends on it."
    request_args = _build_completion_args(api_host=api_host, question=question, system_prompt=system_prompt)

    response = await acompletion(**request_args)
    print(response.choices[0].message.content)


if __name__ == "__main__":
    if os.getenv("USE_ASYNC", "false").lower() in ("1", "true", "yes"):
        asyncio.run(async_main())
    else:
        main()