from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool

from .models import ModelResponse, RuntimeMessage, ToolCallRequest
from .retry import RetryPolicy
from .retry import TransientModelError
from .tools import SideEffect, ToolSpec

def structured_tool_spec(
    tool: BaseTool,
    *,
    side_effect: SideEffect,
    allowed_agents: set[str] | frozenset[str] | None = None,
    retry_policy: RetryPolicy | None = None,
    timeout_seconds: float = 60.0,
    supports_idempotency: bool = False,
) -> ToolSpec:
    '把 LangChain 的 BaseTool（通常是 StructuredTool）转换为 Runtime 的 ToolSpec。\n\nside_effect: SideEffect（必填） 工具的副作用等级。必须显式指定，确保注册者思考过"这工具重试安不安全？"'

    if tool.args_schema is None:
        raise ValueError(f"StructuredTool {tool.name} 没有参数 Schema")

    async def invoke_tool(**arguments: Any) -> Any:
        return await tool.ainvoke(arguments)

    return ToolSpec(
        name=tool.name,
        description=tool.description,
        callable=invoke_tool,
        arguments_model=tool.args_schema,  
        side_effect=side_effect,
        allowed_agents=frozenset(allowed_agents or ()),
        timeout_seconds=timeout_seconds,
        retry_policy=retry_policy or RetryPolicy(),
        supports_idempotency=supports_idempotency,
    )

class LangChainModelAdapter:
    '将任意 LangChain 聊天模型包装为 Runtime 的标准调用接口。'

    def __init__(self, model: Any):
        """
        参数：
          model: Any
            任意 LangChain 聊天模型实例（如 ChatOpenAI、ChatAnthropic 等）。
            只要这个对象支持 .bind_tools() 和 .ainvoke() 方法即可。

            类型标注为 Any 是为了不强制依赖某个具体的 LangChain 类型，
            方便接入不同的模型实现。
        """
        self.model = model

    async def generate(
        self, messages: list[RuntimeMessage], tools: list[dict]
    ) -> ModelResponse:
        '调用模型生成回复（Runtime 的标准入口）。'

        openai_tools = [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool["parameters"],
                },
            }
            for tool in tools
        ]

        runnable = self.model.bind_tools(openai_tools) if openai_tools else self.model

        try:

            response: AIMessage = await runnable.ainvoke(
                [self._to_langchain(message) for message in messages]
            )
        except Exception as exc:

            status_code = getattr(exc, "status_code", None)

            transient_names = {
                "RateLimitError",        
                "APITimeoutError",       
                "APIConnectionError",    
                "InternalServerError",   
            }

            if type(exc).__name__ in transient_names or status_code in {408, 429, 500, 502, 503, 504}:
                
                raise TransientModelError(str(exc)) from exc
            
            raise

        calls = [
            ToolCallRequest(
                id=call.get("id") or "",
                name=call["name"],
                arguments=call.get("args", {}),
            )
            for call in (response.tool_calls or [])
        ]

        content = response.content if isinstance(response.content, str) else str(response.content)

        if not calls and content:
            try:
                payload = json.loads(content)
            except (TypeError, ValueError):
                payload = None
            known_tools = {tool["name"] for tool in tools}
            if (
                isinstance(payload, dict)
                and payload.get("name") in known_tools
                and isinstance(payload.get("arguments"), dict)
            ):
                calls = [
                    ToolCallRequest(
                        name=payload["name"],
                        arguments=payload["arguments"],
                    )
                ]
                content = ""

        return ModelResponse(content=content, tool_calls=calls)

    @staticmethod
    def _to_langchain(message: RuntimeMessage):
        '把一条 Runtime 消息转换为对应的 LangChain 消息类型。'
        if message.role == "system":
            return SystemMessage(content=message.content)
        if message.role == "user":
            return HumanMessage(content=message.content)
        if message.role == "tool":
            
            return ToolMessage(
                content=message.content,
                tool_call_id=message.tool_call_id or "unknown",
            )
        
        return AIMessage(
            content=message.content,
            tool_calls=[
                {"id": call.id, "name": call.name, "args": call.arguments}
                for call in message.tool_calls
            ],
        )
