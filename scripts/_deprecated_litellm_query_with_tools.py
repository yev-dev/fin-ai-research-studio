
import asyncio
import json
import os
from dotenv import load_dotenv
from litellm import acompletion, completion
import yfinance as yf

# This script demonstrates how to query a language model (like GitHub Copilot or Ollama) using LiteLLM with Tools.

def get_latest_yahoo_price(ticker: str) -> dict:
    """Fetch the latest stock price for a given ticker from Yahoo Finance."""
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period="1d")
        if data.empty:
            return {"error": f"No data found for ticker {ticker}"}
        latest_price = data["Close"].iloc[-1]
        return {"ticker": ticker, "price": float(latest_price)}
    except Exception as e:
        return {"error": str(e)}


tools = [
    {
        "type": "function",
        "function": {
            "name": "get_latest_yahoo_price",
            "description": "Fetch the latest stock price for a given ticker from Yahoo Finance",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol (e.g., AAPL, GOOGL)"
                    }
                },
                "required": ["ticker"]
            }
        }
    }
]

def _normalize_ollama_endpoint(endpoint: str) -> str:
    endpoint = endpoint.rstrip("/")
    if endpoint.endswith("/v1"):
        return endpoint[: -len("/v1")]
    return endpoint


def _build_completion_args(api_host: str, question: str, system_prompt: str, tools: list = None) -> dict:
    if api_host == "github":
        return {
            "model": os.getenv("GITHUB_MODEL", "openai/gpt-4o"),
            "api_base": os.getenv("GITHUB_ENDPOINT", "https://models.github.ai/inference"),
            "api_key": os.environ["GITHUB_TOKEN"],
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
            "tools": tools,
        }

    if api_host == "ollama":
        return {
            "model": f"ollama/{os.getenv('OLLAMA_MODEL', 'llama3.1')}",
            "api_base": _normalize_ollama_endpoint(os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434")),
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
            "tools": tools,
        }

    raise ValueError("API_HOST must be either 'github' or 'ollama'.")


def main() -> None:
    load_dotenv(override=True)

    api_host = os.getenv("API_HOST", "github")

    question = "What is the latest Yahoo Finance price for AAPL? Use the get_latest_yahoo_price function tool."
    system_prompt = (
        "You are a financial analysis assistant with access to a function tool named "
        "get_latest_yahoo_price. When a user asks for a latest/current stock price, you must "
        "call this function with the correct ticker symbol. Do not guess or fabricate live prices. "
        "If a ticker is missing, ask for clarification. After receiving tool output, summarize the result briefly."
    )
    request_args = _build_completion_args(api_host=api_host, question=question, system_prompt=system_prompt, tools=tools)


    response = completion(**request_args)
    print(response.choices[0].message.content)
    print(response.choices[0].message.tool_calls)


async def async_main() -> None:
    """Asynchronous variant of the same LiteLLM request flow."""
    load_dotenv(override=True)

    api_host = os.getenv("API_HOST", "github")
    question = "What is the latest Yahoo Finance price for AAPL? Use the get_latest_yahoo_price function tool."
    system_prompt = (
        "You are a financial analysis assistant with access to a function tool named "
        "get_latest_yahoo_price. When a user asks for a latest/current stock price, you must "
        "call this function with the correct ticker symbol. Do not guess or fabricate live prices. "
        "If a ticker is missing, ask for clarification. After receiving tool output, summarize the result briefly."
    )
    request_args = _build_completion_args(api_host=api_host, question=question, system_prompt=system_prompt, tools=tools)

    response = await acompletion(**request_args)
    print(response.choices[0].message.content)
    print(response.choices[0].message.tool_calls)


if __name__ == "__main__":
    if os.getenv("USE_ASYNC", "false").lower() in ("1", "true", "yes"):
        asyncio.run(async_main())
    else:
        main()