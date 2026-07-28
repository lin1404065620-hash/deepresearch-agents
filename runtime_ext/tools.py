from __future__ import annotations

import asyncio
import inspect
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Awaitable, Callable, get_type_hints

from pydantic import BaseModel, ValidationError, create_model

from .models import ToolExecutionResult
from .retry import OutcomeUnknownError, RetryPolicy, TransientToolError
from .security import SecurityViolation

class SideEffect(StrEnum):
    '描述一个工具对系统的影响程度，决定重试策略。'

    NONE = "none"

    READ = "read"

    IDEMPOTENT_WRITE = "idempotent_write"

    NON_IDEMPOTENT_WRITE = "non_idempotent_write"

class DuplicateToolError(ValueError):
    """
    尝试注册同名工具时抛出的错误。

    为什么不允许同名？
      模型通过工具名称来调用工具（如 "web_search"），
      如果有两个同名工具，ToolRegistry 不知道该执行哪个。
      在注册阶段直接报错，比运行时才发现问题要好。
    """
    pass

@dataclass(frozen=True)
class ToolSpec:
    '把一个 Python 函数包装成 Agent 可用的"工具"。\n\n简单来说，ToolSpec 就是"工具的使用说明书 + 安全守则 + 调用入口"三合一。'

    name: str

    description: str

    callable: Callable[..., Any]

    arguments_model: type[BaseModel]

    side_effect: SideEffect

    allowed_agents: frozenset[str] = frozenset()

    timeout_seconds: float = 60.0

    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)

    supports_idempotency: bool = False

    @property
    def json_schema(self) -> dict[str, Any]:
        '生成该工具参数列表的 JSON Schema（模型 function calling 格式）。'
        return self.arguments_model.model_json_schema()

    @classmethod
    def from_callable(
        cls,
        function: Callable[..., Any],
        *,
        name: str | None = None,
        description: str | None = None,
        side_effect: SideEffect,
        allowed_agents: set[str] | frozenset[str] | None = None,
        timeout_seconds: float = 60.0,
        retry_policy: RetryPolicy | None = None,
        supports_idempotency: bool = False,
    ) -> "ToolSpec":
        '从任意 Python 函数自动构造 ToolSpec（最常用的创建方式）。\n\nside_effect: SideEffect（必填，没有默认值） 工具的副作用等级。必须显式指定，防止开发者忘了考虑安全性。 这是设计意图：让每个工具的注册者都必须思考"这工具重试安不安全？"'
        
        signature = inspect.signature(function)

        hints = get_type_hints(function)

        fields: dict[str, tuple[Any, Any]] = {}
        for parameter_name, parameter in signature.parameters.items():
            
            annotation = hints.get(parameter_name, Any)
            
            default = ... if parameter.default is inspect.Parameter.empty else parameter.default
            fields[parameter_name] = (annotation, default)

        model_name = f"{name or function.__name__}_arguments"
        arguments_model = create_model(model_name, **fields)

        return cls(
            name=name or function.__name__,
            description=description or inspect.getdoc(function) or "",
            callable=function,
            arguments_model=arguments_model,
            side_effect=side_effect,
            allowed_agents=frozenset(allowed_agents or ()),
            timeout_seconds=timeout_seconds,
            retry_policy=retry_policy or RetryPolicy(),
            supports_idempotency=supports_idempotency,
        )

class ToolRegistry:
    '工具注册中心：管理所有工具，并负责安全执行。\n\n三大职责： 1. 注册管理 — register()、get()、schema_for() 2. 权限控制 — schemas_for_agent() 按 Agent 过滤、execute() 检查权限 3. 安全执行 — execute() 处理参数校验、超时、重试、幂等、异常分类'

    def __init__(self, *, sleep: Callable[[float], Awaitable[None]] = asyncio.sleep):
        """
        参数：
          sleep: 重试等待函数。默认 asyncio.sleep。
            测试时可以传入一个假函数跳过等待：
            ToolRegistry(sleep=lambda _: None)  # 重试不用等
        """
        self._tools: dict[str, ToolSpec] = {}
        self._idempotency_results: dict[tuple[str, str], ToolExecutionResult] = {}
        self._sleep = sleep

    def register(self, spec: ToolSpec) -> None:
        """
        注册一个工具。

        参数：
          spec: ToolSpec — 要注册的工具（通常由 ToolSpec.from_callable() 创建）

        抛出：
          DuplicateToolError — 如果同名工具已经注册过
        """
        if spec.name in self._tools:
            raise DuplicateToolError(f"工具已注册: {spec.name}")
        self._tools[spec.name] = spec

    def get(self, name: str) -> ToolSpec:
        '根据名称获取工具的完整定义。'
        try:
            return self._tools[name]
        except KeyError as exc:
            raise KeyError(f"未知工具: {name}") from exc

    def schema_for(self, name: str) -> dict[str, Any]:
        '获取指定工具的 JSON Schema。'
        return self.get(name).json_schema

    def schemas_for_agent(self, agent_id: str) -> list[dict[str, Any]]:
        '获取某个 Agent 有权使用的所有工具的 JSON Schema 列表。'
        return [
            {"name": spec.name, "description": spec.description, "parameters": spec.json_schema}
            for spec in self._tools.values()
            if not spec.allowed_agents or agent_id in spec.allowed_agents
        ]

    def restricted(self, allowed_names: frozenset[str] | set[str]) -> "ToolRegistry":
        '创建一个"受限版"的 ToolRegistry，只包含指定的工具。'
        restricted = ToolRegistry(sleep=self._sleep)
        for name in allowed_names:
            restricted.register(self.get(name))
        return restricted

    async def execute(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        agent_id: str,
        tool_call_id: str,
        idempotency_key: str | None = None,
    ) -> ToolExecutionResult:
        '执行一个工具调用（ToolRegistry 最核心的方法）。'
        
        started = time.perf_counter()

        try:
            spec = self.get(name)
        except KeyError as exc:
            return self._failure(tool_call_id, name, "tool_not_found", str(exc), started)

        if spec.allowed_agents and agent_id not in spec.allowed_agents:
            return self._failure(tool_call_id, name, "permission_denied", "Agent 无权调用该工具", started)

        if spec.supports_idempotency and not idempotency_key:
            return self._failure(tool_call_id, name, "idempotency_key_required", "幂等写工具必须提供幂等键", started)

        cache_key = (name, idempotency_key or "")
        if spec.supports_idempotency and cache_key in self._idempotency_results:
            return self._idempotency_results[cache_key]

        try:
            validated = spec.arguments_model.model_validate(arguments).model_dump()
        except ValidationError as exc:
            return self._failure(tool_call_id, name, "invalid_arguments", str(exc), started)

        attempts = 0
        while True:
            attempts += 1
            try:
                
                async with asyncio.timeout(spec.timeout_seconds):
                    output = await self._invoke(spec.callable, validated)

                result = ToolExecutionResult(
                    tool_call_id=tool_call_id,
                    tool_name=name,
                    status="succeeded",
                    output=output,
                    attempts=attempts,
                    duration_ms=int((time.perf_counter() - started) * 1000),
                )

                if spec.supports_idempotency:
                    self._idempotency_results[cache_key] = result
                return result

            except OutcomeUnknownError as exc:
                
                return ToolExecutionResult(
                    tool_call_id=tool_call_id, tool_name=name, status="outcome_unknown",
                    error_code="outcome_unknown", error_message=str(exc), attempts=attempts,
                    duration_ms=int((time.perf_counter() - started) * 1000),
                )

            except SecurityViolation as exc:
                
                return self._failure(tool_call_id, name, "security_violation", str(exc), started, attempts)

            except TimeoutError as exc:
                
                if spec.side_effect == SideEffect.NON_IDEMPOTENT_WRITE:
                    
                    return ToolExecutionResult(
                        tool_call_id=tool_call_id, tool_name=name, status="outcome_unknown",
                        error_code="outcome_unknown", error_message="非幂等工具超时，结果未知",
                        attempts=attempts, duration_ms=int((time.perf_counter() - started) * 1000),
                    )
                
                if attempts >= spec.retry_policy.max_attempts:
                    return self._failure(tool_call_id, name, "timeout", str(exc), started, attempts)

            except TransientToolError as exc:
                
                if spec.side_effect == SideEffect.NON_IDEMPOTENT_WRITE or attempts >= spec.retry_policy.max_attempts:
                    return self._failure(tool_call_id, name, "transient_error", str(exc), started, attempts)

            except Exception as exc:

                return self._failure(tool_call_id, name, "tool_error", str(exc), started, attempts)

            await self._sleep(spec.retry_policy.delay_for(attempts))

    @staticmethod
    async def _invoke(function: Callable[..., Any], arguments: dict[str, Any]) -> Any:
        '调用目标函数，自动判断是 async 还是同步函数。'
        if inspect.iscoroutinefunction(function):
            return await function(**arguments)
        return await asyncio.to_thread(function, **arguments)

    @staticmethod
    def _failure(
        tool_call_id: str,
        tool_name: str,
        code: str,
        message: str,
        started: float,
        attempts: int = 1,
    ) -> ToolExecutionResult:
        '快速构造一个"执行失败"的结果（辅助方法，减少重复代码）。'
        return ToolExecutionResult(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            status="failed",
            error_code=code,
            error_message=message,
            attempts=attempts,
            duration_ms=int((time.perf_counter() - started) * 1000),
        )
