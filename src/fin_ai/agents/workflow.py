"""
Agent workflow classes — FinRobot, SingleAssistant, SingleAssistantRAG,
MultiAssistant, and MultiAssistantWithLeader.

All tool registration flows through FinRobot.register_proxy(), which
looks up callables from ``fin_ai.agents.tools`` by name and registers
them with autogen.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional
from collections import defaultdict
from functools import partial
from abc import ABC, abstractmethod

import autogen
from autogen.cache import Cache
from autogen import (
    ConversableAgent,
    AssistantAgent,
    UserProxyAgent,
    GroupChat,
    GroupChatManager,
    register_function,
)

from .agent_library import library
from .agentic_rag import get_rag_function
from .utils import (
    instruction_trigger,
    instruction_message,
    order_trigger,
    order_message,
)
from .prompts import leader_system_message, role_system_message

# ---------------------------------------------------------------------------
# Built-in tool name → callable mapping (from fin_ai.agents.tools)
# ---------------------------------------------------------------------------

def _build_tool_map() -> dict[str, Callable]:
    """Map tool names to callables from fin_ai.core.tools and engine_bridge."""
    import fin_ai.core.tools as _core_tools
    from .engine_bridge import ENGINE_BRIDGE_TOOLS

    tool_map: dict[str, Callable] = {}

    # Core YFinance tools
    for name in dir(_core_tools):
        if callable(getattr(_core_tools, name)) and not name.startswith("_"):
            tool_map[name] = getattr(_core_tools, name)

    # Engine bridge tools (local RAG, model listing, etc.)
    tool_map.update(ENGINE_BRIDGE_TOOLS)

    return tool_map

_TOOL_MAP: dict[str, Callable] = _build_tool_map()


def register_tools(
    tool_names: list[str],
    caller: ConversableAgent,
    executor: ConversableAgent,
) -> None:
    """Register named tools from ``fin_ai.agents.tools`` with autogen.

    Parameters
    ----------
    tool_names : list[str]
        Names of callables in ``fin_ai.agents.tools`` to register.
    caller : ConversableAgent
        The agent that will invoke the tools.
    executor : ConversableAgent
        The agent that will execute the tools (typically the UserProxy).
    """
    for name in tool_names:
        func = _TOOL_MAP.get(name)
        if func is None:
            raise ValueError(
                f"Tool '{name}' not found. "
                f"Available: {sorted(_TOOL_MAP)}"
            )
        register_function(
            func,
            caller=caller,
            executor=executor,
            name=name,
            description=func.__doc__ or "",
        )


# ---------------------------------------------------------------------------
# AIAgent — configurable agent wrapper
# ---------------------------------------------------------------------------


class AIAgent(AssistantAgent):
    """A configurable agent that wraps autogen's ``AssistantAgent``.

    Supports lookup from the agent library by name, config preprocessing,
    and automatic tool registration from ``fin_ai.agents.tools``.
    """

    def __init__(
        self,
        agent_config: str | Dict[str, Any],
        system_message: str | None = None,
        tools: list[str] | None = None,
        proxy: UserProxyAgent | None = None,
        **kwargs,
    ):
        orig_name = ""
        if isinstance(agent_config, str):
            orig_name = agent_config
            name = orig_name.replace("_Shadow", "")
            assert name in library, f"AIAgent '{name}' not found in agent library."
            agent_config = library[name]

        agent_config = self._preprocess_config(agent_config)

        assert agent_config, "agent_config is required."
        assert agent_config.get("name", ""), "name needs to be in config."

        name = orig_name if orig_name else agent_config["name"]
        default_system_message = agent_config.get("profile", None)
        default_tools: list[str] = agent_config.get("tools", [])

        system_message = system_message or default_system_message
        _tool_names = tools if tools is not None else default_tools

        name = name.replace(" ", "_").strip()

        completion_guard = (
            "\n\nCompletion rules: provide the final answer directly once you have enough "
            "information. Do not ask the user to continue, do not ask for permission, and "
            "do not end with a question. After the final answer, end the message with TERMINATE."
        )
        system_message = (system_message or "") + completion_guard

        super().__init__(
            name,
            system_message,
            description=agent_config.get("description", ""),
            **kwargs,
        )

        # Store tool names separately — 'tools' is a read-only property on AIAgent
        self._tool_names = _tool_names

        if proxy is not None:
            self.register_proxy(proxy)

    def _preprocess_config(self, config: dict) -> dict:
        """Merge role/leader prompts into the profile."""
        config = dict(config)  # shallow copy
        role_prompt, leader_prompt, responsibilities = "", "", ""

        if "responsibilities" in config:
            title = config.get("title", config.get("name", ""))
            if "name" not in config:
                config["name"] = config["title"]
            responsibilities = config["responsibilities"]
            responsibilities = (
                "\n".join([f" - {r}" for r in responsibilities])
                if isinstance(responsibilities, list)
                else responsibilities
            )
            role_prompt = role_system_message.format(
                title=title,
                responsibilities=responsibilities,
            )

        name = config.get("name", "")
        description = (
            f"Name: {name}\nResponsibility:\n{responsibilities}"
            if responsibilities
            else f"Name: {name}"
        )
        config["description"] = description.strip()

        if "group_desc" in config:
            group_desc = config["group_desc"]
            leader_prompt = leader_system_message.format(group_desc=group_desc)

        config["profile"] = (
            (role_prompt + "\n\n").strip()
            + (leader_prompt + "\n\n").strip()
            + config.get("profile", "")
        ).strip()

        return config

    def register_proxy(self, proxy):
        """Register this agent's tools with the given executor proxy."""
        register_tools(self._tool_names, self, proxy)


# ---------------------------------------------------------------------------
# Single Assistant Base
# ---------------------------------------------------------------------------


class SingleAssistantBase(ABC):
    """Abstract base for single-assistant workflows."""

    def __init__(
        self,
        agent_config: str | Dict[str, Any],
        llm_config: Dict[str, Any] | None = None,
    ):
        self.assistant = AIAgent(
            agent_config=agent_config,
            llm_config=llm_config or {},
            proxy=None,
        )

    @abstractmethod
    def chat(self, message: str, **kwargs):
        ...

    @abstractmethod
    def reset(self):
        ...


# ---------------------------------------------------------------------------
# SingleAssistant
# ---------------------------------------------------------------------------


class SingleAssistant(SingleAssistantBase):
    """Single agent + user proxy workflow."""

    def __init__(
        self,
        agent_config: str | Dict[str, Any],
        llm_config: Dict[str, Any] | None = None,
        is_termination_msg=lambda x: x.get("content", "")
        and x.get("content", "").endswith("TERMINATE"),
        human_input_mode: str = "NEVER",
        max_consecutive_auto_reply: int = 10,
        code_execution_config: dict | None = None,
        **kwargs,
    ):
        super().__init__(agent_config, llm_config=llm_config)
        _code_exec = code_execution_config or {
            "work_dir": "coding",
            "use_docker": False,
        }
        self.user_proxy = UserProxyAgent(
            name="User_Proxy",
            is_termination_msg=is_termination_msg,
            human_input_mode=human_input_mode,
            max_consecutive_auto_reply=max_consecutive_auto_reply,
            code_execution_config=_code_exec,
            **kwargs,
        )
        self.assistant.register_proxy(self.user_proxy)

    def chat(self, message: str, use_cache: bool = False, **kwargs):
        with Cache.disk() as cache:
            self.user_proxy.initiate_chat(
                self.assistant,
                message=message,
                cache=cache if use_cache else None,
                **kwargs,
            )
        print("Current chat finished. Resetting agents ...")
        self.reset()

    def reset(self):
        self.user_proxy.reset()
        self.assistant.reset()


# ---------------------------------------------------------------------------
# SingleAssistantRAG
# ---------------------------------------------------------------------------


class SingleAssistantRAG(SingleAssistant):
    """Single assistant with RAG retrieval capability.

    Parameters
    ----------
    retrieve_config : dict
        Configuration for the autogen ``RetrieveUserProxyAgent``.
    rag_description : str
        Description of the RAG tool shown to the agent.
    """

    def __init__(
        self,
        agent_config: str | Dict[str, Any],
        llm_config: Dict[str, Any] | None = None,
        is_termination_msg=lambda x: x.get("content", "")
        and x.get("content", "").endswith("TERMINATE"),
        human_input_mode: str = "NEVER",
        max_consecutive_auto_reply: int = 10,
        code_execution_config: dict | None = None,
        retrieve_config: dict | None = None,
        rag_description: str = "",
        **kwargs,
    ):
        super().__init__(
            agent_config,
            llm_config=llm_config,
            is_termination_msg=is_termination_msg,
            human_input_mode=human_input_mode,
            max_consecutive_auto_reply=max_consecutive_auto_reply,
            code_execution_config=code_execution_config,
            **kwargs,
        )
        assert retrieve_config, "retrieve_config cannot be empty for RAG Agent."
        rag_func, rag_assistant = get_rag_function(retrieve_config, rag_description)
        self.rag_assistant = rag_assistant
        register_function(
            rag_func,
            caller=self.assistant,
            executor=self.user_proxy,
            description=rag_description if rag_description else rag_func.__doc__,
        )

    def reset(self):
        super().reset()
        self.rag_assistant.reset()


# ---------------------------------------------------------------------------
# SingleAssistantShadow
# ---------------------------------------------------------------------------


class SingleAssistantShadow(SingleAssistant):
    """Single assistant with a shadow agent for self-reflection via nested chats."""

    def __init__(
        self,
        agent_config: str | Dict[str, Any],
        llm_config: Dict[str, Any] | None = None,
        is_termination_msg=lambda x: x.get("content", "")
        and x.get("content", "").endswith("TERMINATE"),
        human_input_mode: str = "NEVER",
        max_consecutive_auto_reply: int = 10,
        code_execution_config: dict | None = None,
        **kwargs,
    ):
        super().__init__(
            agent_config,
            llm_config=llm_config,
            is_termination_msg=is_termination_msg,
            human_input_mode=human_input_mode,
            max_consecutive_auto_reply=max_consecutive_auto_reply,
            code_execution_config=code_execution_config,
            **kwargs,
        )
        if isinstance(agent_config, dict):
            agent_config_shadow = agent_config.copy()
            agent_config_shadow["name"] = agent_config["name"] + "_Shadow"
            agent_config_shadow["_tool_names"] = []
        else:
            agent_config_shadow = agent_config + "_Shadow"

        self.assistant_shadow = AIAgent(
            agent_config_shadow,
            tools=[],
            llm_config=llm_config or {},
            proxy=None,
        )
        self.assistant.register_nested_chats(
            [
                {
                    "sender": self.assistant,
                    "recipient": self.assistant_shadow,
                    "message": instruction_message,
                    "summary_method": "last_msg",
                    "max_turns": 2,
                    "silent": True,
                }
            ],
            trigger=instruction_trigger,
        )


# ---------------------------------------------------------------------------
# Multi Assistant Base
# ---------------------------------------------------------------------------


class MultiAssistantBase(ABC):
    """Abstract base for multi-agent workflows."""

    def __init__(
        self,
        group_config: str | dict,
        agent_configs: List[Dict[str, Any] | str | ConversableAgent] | None = None,
        llm_config: Dict[str, Any] | None = None,
        user_proxy: UserProxyAgent | None = None,
        is_termination_msg=lambda x: x.get("content", "")
        and x.get("content", "").endswith("TERMINATE"),
        human_input_mode: str = "NEVER",
        max_consecutive_auto_reply: int = 10,
        code_execution_config: dict | None = None,
        **kwargs,
    ):
        self.group_config = group_config
        self.llm_config = llm_config or {}
        _code_exec = code_execution_config or {
            "work_dir": "coding",
            "use_docker": False,
        }
        if user_proxy is None:
            self.user_proxy = UserProxyAgent(
                name="User_Proxy",
                is_termination_msg=is_termination_msg,
                human_input_mode=human_input_mode,
                max_consecutive_auto_reply=max_consecutive_auto_reply,
                code_execution_config=_code_exec,
                **kwargs,
            )
        else:
            self.user_proxy = user_proxy
        self.agent_configs = agent_configs or group_config.get("agents", [])
        assert self.agent_configs, "agent_configs is required."
        self.agents: list = []
        self._init_agents()
        self.representative = self._get_representative()

    def _init_single_agent(self, agent_config):
        if isinstance(agent_config, ConversableAgent):
            return agent_config
        return AIAgent(
            agent_config,
            llm_config=self.llm_config,
            proxy=self.user_proxy,
        )

    def _init_agents(self):
        agent_dict = defaultdict(list)
        for c in self.agent_configs:
            agent = self._init_single_agent(c)
            agent_dict[agent.name].append(agent)

        for name, agent_list in agent_dict.items():
            if len(agent_list) == 1:
                self.agents.append(agent_list[0])
                continue
            for idx, agent in enumerate(agent_list):
                agent._name = f"{name}_{idx + 1}"
                self.agents.append(agent)

    @abstractmethod
    def _get_representative(self) -> ConversableAgent:
        ...

    def chat(self, message: str, use_cache: bool = False, **kwargs):
        with Cache.disk() as cache:
            self.user_proxy.initiate_chat(
                self.representative,
                message=message,
                cache=cache if use_cache else None,
                **kwargs,
            )
        print("Current chat finished. Resetting agents ...")
        self.reset()

    def reset(self):
        self.user_proxy.reset()
        self.representative.reset()
        for agent in self.agents:
            agent.reset()


# ---------------------------------------------------------------------------
# MultiAssistant — Group Chat Workflow
# ---------------------------------------------------------------------------


class MultiAssistant(MultiAssistantBase):
    """Group Chat workflow with multiple agents."""

    def _get_representative(self):
        def custom_speaker_selection_func(
            last_speaker: autogen.Agent, groupchat: autogen.GroupChat
        ):
            messages = groupchat.messages
            if len(messages) <= 1:
                return groupchat.agents[0]
            if last_speaker is self.user_proxy:
                return groupchat.agent_by_name(messages[-2]["name"])
            elif "tool_calls" in messages[-1] or messages[-1]["content"].endswith(
                "TERMINATE"
            ):
                return self.user_proxy
            else:
                return groupchat.next_agent(last_speaker, groupchat.agents[:-1])

        self.group_chat = GroupChat(
            self.agents + [self.user_proxy],
            messages=[],
            speaker_selection_method=custom_speaker_selection_func,
            send_introductions=True,
        )
        manager_name = (self.group_config.get("name", "") + "_chat_manager").strip(
            "_"
        )
        manager = GroupChatManager(
            self.group_chat, name=manager_name, llm_config=self.llm_config
        )
        return manager


# ---------------------------------------------------------------------------
# MultiAssistantWithLeader — Leader-based Workflow
# ---------------------------------------------------------------------------


class MultiAssistantWithLeader(MultiAssistantBase):
    """Leader-based workflow with nested chats.

    Group config structure::

        {
            "leader": {
                "title": "CIO",
                "responsibilities": ["Coordinate analysis", ...]
            },
            "agents": [
                {"title": "Analyst 1", "responsibilities": [...]},
                ...
            ]
        }
    """

    def _get_representative(self):
        assert (
            "leader" in self.group_config and "agents" in self.group_config
        ), "Leader and Agents must be explicitly defined in config."

        assert self.agent_configs, (
            "At least one agent must be defined in the group config."
        )

        need_suffix = (
            len(
                set(
                    c["title"] for c in self.agent_configs if isinstance(c, dict)
                )
            )
            == 1
        )

        group_desc = ""
        for i, c in enumerate(self.agent_configs):
            if isinstance(c, ConversableAgent):
                group_desc += c.description + "\n\n"
            else:
                name = (c.get("title", c.get("name", ""))).replace(" ", "_").strip()
                name += f"_{i + 1}" if need_suffix else ""
                responsibilities = "\n".join(
                    [f" - {r}" for r in c.get("responsibilities", [])]
                )
                group_desc += f"Name: {name}\nResponsibility:\n{responsibilities}\n\n"

        self.leader_config = dict(self.group_config["leader"])
        self.leader_config["group_desc"] = group_desc.strip()

        leader = self._init_single_agent(self.leader_config)

        for agent in self.agents:
            self.user_proxy.register_nested_chats(
                [
                    {
                        "sender": self.user_proxy,
                        "recipient": agent,
                        "message": partial(order_message, agent.name),
                        "summary_method": "reflection_with_llm",
                        "max_turns": 10,
                        "max_consecutive_auto_reply": 3,
                    }
                ],
                trigger=partial(
                    order_trigger, name=leader.name, pattern=f"[{agent.name}]"
                ),
            )
        return leader
