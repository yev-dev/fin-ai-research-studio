
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


def _extract_tool_calls(message: object) -> list[dict]:
    tool_calls = getattr(message, "tool_calls", None)
    if not tool_calls:
        return []

    extracted: list[dict] = []
    for tool_call in tool_calls:
        if isinstance(tool_call, dict):
            function_payload = tool_call.get("function", {})
            name = function_payload.get("name")
            arguments_text = function_payload.get("arguments", "{}")
            call_id = tool_call.get("id")
        else:
            function_payload = getattr(tool_call, "function", None)
            name = getattr(function_payload, "name", None)
            arguments_text = getattr(function_payload, "arguments", "{}")
            call_id = getattr(tool_call, "id", None)

        try:
            arguments = json.loads(arguments_text or "{}")
        except json.JSONDecodeError:
            arguments = {}

        extracted.append(
            {
                "id": call_id or f"call_{len(extracted)}",
                "name": name,
                "arguments": arguments,
                "arguments_text": arguments_text or "{}",
            }
        )

    return extracted


def _execute_tool(name: str | None, arguments: dict) -> dict:
    if name == "get_latest_yahoo_price":
        ticker = str(arguments.get("ticker", "")).strip()
        if not ticker:
            return {"error": "Missing required argument: ticker"}
        return get_latest_yahoo_price(ticker)

    return {"error": f"Unsupported tool: {name}"}


def _build_follow_up_messages(
    question: str,
    system_prompt: str,
    assistant_content: str | None,
    tool_calls: list[dict],
) -> list[dict]:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
        {
            "role": "assistant",
            "content": assistant_content or "",
            "tool_calls": [
                {
                    "id": tool_call["id"],
                    "type": "function",
                    "function": {
                        "name": tool_call["name"],
                        "arguments": tool_call["arguments_text"],
                    },
                }
                for tool_call in tool_calls
            ],
        },
    ]

    for tool_call in tool_calls:
        tool_result = _execute_tool(tool_call["name"], tool_call["arguments"])
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "name": tool_call["name"] or "unknown_tool",
                "content": json.dumps(tool_result),
            }
        )

    return messages


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
    first_message = response.choices[0].message
    print(first_message.content)
    print(first_message.tool_calls)

    tool_calls = _extract_tool_calls(first_message)
    if not tool_calls:
        return

    follow_up_messages = _build_follow_up_messages(
        question=question,
        system_prompt=system_prompt,
        assistant_content=first_message.content,
        tool_calls=tool_calls,
    )

    follow_up_args = _build_completion_args(
        api_host=api_host,
        question=question,
        system_prompt=system_prompt,
        tools=tools,
    )
    follow_up_args["messages"] = follow_up_messages
    final_response = completion(**follow_up_args)
    print(final_response.choices[0].message.content)


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
    first_message = response.choices[0].message
    print(first_message.content)
    print(first_message.tool_calls)

    tool_calls = _extract_tool_calls(first_message)
    if not tool_calls:
        return

    follow_up_messages = _build_follow_up_messages(
        question=question,
        system_prompt=system_prompt,
        assistant_content=first_message.content,
        tool_calls=tool_calls,
    )

    follow_up_args = _build_completion_args(
        api_host=api_host,
        question=question,
        system_prompt=system_prompt,
        tools=tools,
    )
    follow_up_args["messages"] = follow_up_messages
    final_response = await acompletion(**follow_up_args)
    print(final_response.choices[0].message.content)


if __name__ == "__main__":
    if os.getenv("USE_ASYNC", "false").lower() in ("1", "true", "yes"):
        asyncio.run(async_main())
    else:
        main()