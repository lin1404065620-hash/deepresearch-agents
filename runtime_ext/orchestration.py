from __future__ import annotations

from dataclasses import dataclass
import json
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Any

from .context_manager import ContextManager, SummaryGenerator
from .events import EventBus
from .models import ExecutionContext, RuntimeEvent, new_id
from .runtime import AgentRuntime, ModelAdapter
from .store import MemoryStore
from .tools import ToolRegistry

@dataclass(frozen=True)
class AgentDefinition:
    '定义一个子 Agent 的完整配置。'

    agent_id: str

    description: str

    model: ModelAdapter

    allowed_tools: frozenset[str]

    system_prompt: str = ""

    max_steps: int = 10

    max_tokens: int = 4_000

    required_tool: str | None = None

    summary_generator: SummaryGenerator | None = None

class SubAgentExecutor:
    """
    子 Agent 的执行调度器。

    负责：
      1. 验证主 Agent 是否有权限委派该子 Agent
      2. 创建子 Agent 的隔离执行上下文
      3. 选择合适的执行路径（快速通道 / 完整 Agent Loop）
      4. 统一处理子 Agent 的成功和失败返回
    """

    def __init__(
        self,
        definitions: dict[str, AgentDefinition],
        registry: ToolRegistry,
        store: MemoryStore,
        event_bus: EventBus,
    ):
        '参数： definitions: dict[str, AgentDefinition] 所有已注册的子 Agent 定义。key 是 agent_id，value 是 AgentDefinition。 例如 {"network": AgentDefinition(...), "database": AgentDefinition(...)}'
        self.definitions = definitions
        self.registry = registry
        self.store = store
        self.event_bus = event_bus

    async def delegate(
        self,
        parent_context: ExecutionContext,
        agent_id: str,
        task: str,
    ) -> dict[str, Any]:
        '委派一个任务给指定的子 Agent 执行。'

        permission = f"delegate:{agent_id}"
        if permission not in parent_context.permissions:
            raise PermissionError(f"缺少委派权限: {permission}")

        try:
            definition = self.definitions[agent_id]
        except KeyError as exc:
            raise ValueError(f"未知子 Agent: {agent_id}") from exc

        child_context = parent_context.for_child_agent(
            agent_id,
            permissions=frozenset(f"tool:{name}" for name in definition.allowed_tools),
        )

        await self.event_bus.publish(RuntimeEvent(
            session_id=child_context.session_id,
            task_id=child_context.task_id,
            run_id=child_context.run_id,
            agent_id=agent_id,
            type="assistant_call",
            payload={
                "agent_id": agent_id,
                "description": definition.description,
                "task": task,
            },
        ), child_context.user_id)

        if definition.required_tool:
            
            tool_call_id = new_id("call")
            await self.event_bus.publish(RuntimeEvent(
                session_id=child_context.session_id, task_id=child_context.task_id,
                run_id=child_context.run_id, agent_id=agent_id, type="tool_started",
                payload={"tool_call_id": tool_call_id, "tool_name": definition.required_tool},
            ), child_context.user_id)

            result = await self.registry.execute(
                definition.required_tool, {"query": task}, agent_id=agent_id,
                tool_call_id=tool_call_id,
            )

            await self.event_bus.publish(RuntimeEvent(
                session_id=child_context.session_id, task_id=child_context.task_id,
                run_id=child_context.run_id, agent_id=agent_id, type="tool_completed",
                payload=result.model_dump(mode="json"),
            ), child_context.user_id)

            if result.status != "succeeded":
                return {
                    "agent_id": agent_id, "status": "failed", "conclusion": "",
                    "errors": [result.error_message or result.error_code or "工具执行失败"],
                }

            conclusion = (
                result.output if isinstance(result.output, str)
                else json.dumps(result.output, ensure_ascii=False)
            )

            time_markers = ("几点", "当前时间", "现在时间", "北京时间", "current time")
            if agent_id == "network" and any(
                marker in task.casefold() for marker in time_markers
            ):
                observed = datetime.now(ZoneInfo("Asia/Shanghai")).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                conclusion = (
                    f"当前北京时间：{observed}（Asia/Shanghai，运行时观测）\n"
                    f"网络检索结果：{conclusion}"
                )

            return {
                "agent_id": agent_id, "status": "succeeded",
                "conclusion": conclusion, "errors": [],
            }

        runtime = AgentRuntime(
            definition.model,
            self.registry.restricted(definition.allowed_tools),  
            self.store,
            self.event_bus,
            context_manager=ContextManager(
                max_tokens=definition.max_tokens,
                output_budget=min(1_024, definition.max_tokens // 4),
                safety_margin=min(256, definition.max_tokens // 16),
                summary_generator=definition.summary_generator,
            ),
            max_steps=definition.max_steps,
            system_prompt=definition.system_prompt,
        )

        try:
            
            conclusion = await runtime.run(
                child_context, task, finalize_task=False,
            )
            if self.store.exists_failed_tool_call(
                child_context.run_id, child_context.user_id,
            ):
                return {
                    "agent_id": agent_id,
                    "status": "failed",
                    "conclusion": "",
                    "errors": ["子 Agent 的工具调用未能成功完成"],
                }
            return {
                "agent_id": agent_id,
                "status": "succeeded",
                "conclusion": conclusion,
                "errors": [],
            }
        except Exception:
            
            return {
                "agent_id": agent_id,
                "status": "failed",
                "conclusion": "",
                "errors": ["子 Agent 执行失败"],
            }
