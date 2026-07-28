from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Awaitable, Callable
from typing import Protocol

from .context import runtime_scope
from .context_manager import CompactionResult, ContextManager
from .events import EventBus
from .models import (
    ExecutionContext,
    ModelResponse,
    RuntimeEvent,
    RuntimeMessage,
    ToolExecutionResult,
    new_id,
)
from .retry import RetryPolicy, TransientModelError
from .store import MemoryStore
from .tools import ToolRegistry

class ModelAdapter(Protocol):
    '模型适配器必须遵守的接口协议。'
    async def generate(
        self, messages: list[RuntimeMessage], tools: list[dict]
    ) -> ModelResponse:
        """
        参数：
          messages: Runtime 格式的消息列表
          tools: 工具 Schema 列表

        返回：
          ModelResponse — 包含模型文本回复和工具调用请求
        """
        ...

class AgentRuntime:
    'Agent 运行时：管理 Agent Loop 的完整生命周期。'

    def __init__(
        self,
        model: ModelAdapter,
        registry: ToolRegistry,
        store: MemoryStore,
        event_bus: EventBus,
        *,
        context_manager: ContextManager | None = None,
        max_steps: int = 20,
        system_prompt: str | None = None,
        model_retry_policy: RetryPolicy | None = None,
        task_timeout_seconds: float = 1200,
        lease_heartbeat_seconds: float = 20,
        max_history_messages: int = 1_000,
        max_history_scan_per_run: int = 2_000,
        max_history_compaction_per_run: int = 1_000,
        max_compaction_rounds_per_run: int = 5,
        recent_history_messages: int = 100,
        history_compaction_threshold: int = 150,
        max_episode_candidates: int = 500,
        max_recalled_episodes: int = 3,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ):
        '参数： model: ModelAdapter 模型适配器（如 LangChainModelAdapter），Runtime 通过它调用 LLM。'
        
        if max_history_messages <= 0:
            raise ValueError("max_history_messages must be positive")
        if max_history_scan_per_run <= 0:
            raise ValueError("max_history_scan_per_run must be positive")
        if max_history_compaction_per_run <= 0:
            raise ValueError("max_history_compaction_per_run must be positive")
        if max_compaction_rounds_per_run <= 0:
            raise ValueError("max_compaction_rounds_per_run must be positive")
        if recent_history_messages <= 0:
            raise ValueError("recent_history_messages must be positive")
        if history_compaction_threshold <= 0:
            raise ValueError("history_compaction_threshold must be positive")
        if history_compaction_threshold < recent_history_messages:
            raise ValueError(
                "history_compaction_threshold must be greater than or equal to "
                "recent_history_messages"
            )
        if max_episode_candidates <= 0:
            raise ValueError("max_episode_candidates must be positive")
        if max_recalled_episodes <= 0:
            raise ValueError("max_recalled_episodes must be positive")

        self.model = model
        self.registry = registry
        self.store = store
        self.event_bus = event_bus

        self.context_manager = context_manager or ContextManager(
            max_tokens=8_000,
            output_budget=1_024,
            safety_margin=256,
        )

        self.max_steps = max_steps
        self.system_prompt = system_prompt
        self.model_retry_policy = model_retry_policy or RetryPolicy(max_attempts=3)

        self.task_timeout_seconds = task_timeout_seconds
        self.lease_heartbeat_seconds = lease_heartbeat_seconds

        self.max_history_messages = max_history_messages
        self.max_history_scan_per_run = max_history_scan_per_run
        self.max_history_compaction_per_run = max_history_compaction_per_run
        self.max_compaction_rounds_per_run = max_compaction_rounds_per_run
        self.recent_history_messages = min(
            recent_history_messages,
            self.store.MAX_PAGE_SIZE,  
        )
        self.history_compaction_threshold = history_compaction_threshold
        self.max_episode_candidates = min(max_episode_candidates, 500)
        self.max_recalled_episodes = min(
            max_recalled_episodes,
            self.max_episode_candidates,  
        )

        self.sleep = sleep

    async def run(
        self, context: ExecutionContext, query: str, *, finalize_task: bool = True
    ) -> str:
        '启动一次 Agent 运行（外部调用的唯一入口）。'
        
        heartbeat = asyncio.create_task(self._lease_heartbeat(context)) if finalize_task else None
        try:
            
            async with asyncio.timeout(self.task_timeout_seconds):
                return await self._run_impl(context, query, finalize_task=finalize_task)
        except asyncio.CancelledError:
            raise  
        except TimeoutError:
            if finalize_task:
                await self._fail_if_running(context, "timed_out", "任务执行超时", "task_timeout")
            raise
        except Exception as exc:
            if finalize_task:
                await self._fail_if_running(context, "failed", str(exc), type(exc).__name__)
            raise
        finally:
            
            if heartbeat is not None:
                heartbeat.cancel()
                await asyncio.gather(heartbeat, return_exceptions=True)

    async def _run_impl(
        self, context: ExecutionContext, query: str, *, finalize_task: bool = True
    ) -> str:
        'Agent Loop 的完整实现（整个 Runtime 最核心的方法）。'
        
        if self.store.get_task(context.task_id, context.user_id)["status"] == "cancelled":
            self.store.finish_run(context.run_id, context.user_id, "cancelled")
            raise asyncio.CancelledError("任务已取消")

        messages: list[RuntimeMessage] = []
        persisted_message_ids: list[int] = []

        if self.system_prompt:
            messages.append(RuntimeMessage(
                session_id=context.session_id, task_id=context.task_id, run_id=context.run_id,
                agent_id=context.agent_id, role="system", content=self.system_prompt,
            ))

        memory_state = self.store.get_memory_state(context.session_id, context.user_id)
        relevant_episodes = self.store.search_memory_episodes(
            context.session_id,
            context.user_id,
            query,
            limit=self.max_recalled_episodes,
            candidate_limit=self.max_episode_candidates,
        )
        memory_content: str | None = None
        if memory_state is not None or relevant_episodes:
            memory_payload = {
                "state": memory_state["state"] if memory_state is not None else {},
                "relevant_episodes": [
                    {
                        "summary": episode["summary"],
                        "importance": episode["importance"],
                        "message_range": [
                            episode["start_message_id"],
                            episode["end_message_id"],
                        ],
                    }
                    for episode in relevant_episodes
                ],
            }
            
            memory_content = self._bounded_memory_context(memory_payload)
            messages.append(RuntimeMessage(
                session_id=context.session_id,
                task_id=context.task_id,
                run_id=context.run_id,
                agent_id=context.agent_id,
                role="system",
                content=memory_content,
            ))

        fixed_context_messages = list(messages)

        latest_summary = self.store.get_latest_summary(context.session_id, context.user_id)
        after_message_id = (
            latest_summary.get("end_message_id") or 0
            if latest_summary is not None
            else 0
        )

        tool_schemas = self.registry.schemas_for_agent(context.agent_id)
        total_unsummarized = self.store.count_messages_after(
            context.session_id, context.user_id, after_message_id,
        )

        count_compaction_triggered = (
            total_unsummarized > self.history_compaction_threshold
        )
        if count_compaction_triggered:
            recent_count = min(total_unsummarized, self.recent_history_messages)
            backlog_count = max(0, total_unsummarized - recent_count)
        else:
            recent_count = total_unsummarized
            backlog_count = 0

        backlog_work_limit = min(
            backlog_count,
            self.max_history_scan_per_run,
            self.max_history_compaction_per_run,
        )
        backlog_messages = list(fixed_context_messages)
        backlog_ids: list[int] = []
        backlog_cursor = after_message_id

        while len(backlog_ids) < backlog_work_limit:
            page_limit = min(
                self.store.MAX_PAGE_SIZE,
                self.max_history_messages,
                backlog_work_limit - len(backlog_ids),
            )
            history = self.store.list_messages(
                context.session_id, context.user_id,
                after_message_id=backlog_cursor, limit=page_limit,
            )
            if not history:
                break
            for item in history:
                backlog_messages.append(self._stored_message(context, item))
                backlog_ids.append(item["message_id"])
            backlog_cursor = history[-1]["message_id"]
            if len(history) < page_limit:
                break

        if backlog_ids:
            _, _, latest_summary = await self._compact_backlog_with_limits(
                context, backlog_messages, backlog_ids, latest_summary, tool_schemas,
            )

        messages = list(fixed_context_messages)
        persisted_message_ids = []

        if latest_summary is not None:
            messages.append(RuntimeMessage(
                session_id=context.session_id, task_id=context.task_id, run_id=context.run_id,
                agent_id=context.agent_id, role="system",
                content=f"[持久化历史摘要]\n{latest_summary['content']}",
            ))

        current_summary_end = (
            latest_summary.get("end_message_id") or 0
            if latest_summary is not None
            else 0
        )

        recent_history = self.store.list_recent_messages_after(
            context.session_id, context.user_id,
            after_message_id=current_summary_end,
            limit=(
                self.recent_history_messages
                if count_compaction_triggered
                else max(1, recent_count)
            ),
        )

        remaining_count = self.store.count_messages_after(
            context.session_id, context.user_id, current_summary_end,
        )
        remaining_backlog_count = max(0, remaining_count - len(recent_history))
        has_history_gap = remaining_backlog_count > 0

        if has_history_gap:
            gap_start = current_summary_end + 1
            gap_end = recent_history[0]["message_id"] - 1
            messages.append(RuntimeMessage(
                session_id=context.session_id,
                task_id=context.task_id,
                run_id=context.run_id,
                agent_id=context.agent_id,
                role="system",
                content=(
                    "[未摘要积压区]\n"
                    f"message_id={gap_start}..{gap_end}，共 {remaining_backlog_count} 条。"
                    "该区间仍保存在数据库，将由后续请求或独立压缩 Worker "
                    "从最早未覆盖位置连续整理；本轮不得假定该区间内容已包含。"
                ),
            ))

        for item in recent_history:
            messages.append(self._stored_message(context, item))
            persisted_message_ids.append(item["message_id"])

        messages.append(RuntimeMessage(
            session_id=context.session_id, task_id=context.task_id, run_id=context.run_id,
            agent_id=context.agent_id, role="user", content=query,
        ))
        
        persisted_message_ids.append(self.store.append_message(
            context.session_id, context.user_id, role="user", content=query,
            task_id=context.task_id, run_id=context.run_id, agent_id=context.agent_id,
        ))

        await self._event(context, "run_started", {"query": query})

        with runtime_scope(context):
            for step in range(1, self.max_steps + 1):
                
                if self.store.get_task(context.task_id, context.user_id)["status"] == "cancelled":
                    self.store.finish_run(context.run_id, context.user_id, "cancelled")
                    await self._event(context, "run_cancelled", {})
                    raise asyncio.CancelledError("任务已取消")

                if has_history_gap:

                    messages, compaction = await self._compact_ephemeral_until_ready(
                        context, messages, tool_schemas,
                    )
                else:
                    
                    (
                        messages, persisted_message_ids, latest_summary, compaction,
                    ) = await self._compact_until_ready(
                        context, messages, persisted_message_ids, latest_summary, tool_schemas,
                    )

                effective_messages = compaction.messages
                response = await self._generate_with_retry(context, effective_messages)

                assistant = RuntimeMessage(
                    session_id=context.session_id, task_id=context.task_id, run_id=context.run_id,
                    agent_id=context.agent_id, role="assistant", content=response.content,
                    tool_calls=response.tool_calls,
                )
                messages.append(assistant)
                persisted_message_ids.append(self.store.append_message(
                    context.session_id, context.user_id, role="assistant", content=response.content,
                    task_id=context.task_id, run_id=context.run_id, agent_id=context.agent_id,
                ))

                if not response.tool_calls:

                    if not has_history_gap:
                        await self._compact_and_persist(
                            context, messages, persisted_message_ids, latest_summary, tool_schemas,
                        )
                    if finalize_task:

                        if self._requires_artifact(query) and not self._run_has_artifact(context):
                            raise RuntimeError("报告任务没有生成可下载文件")
                        self.store.update_task_status(
                            context.task_id, context.user_id, "completed", result=response.content,
                        )
                        self.store.finish_run(context.run_id, context.user_id, "completed")
                    await self._event(
                        context,
                        "run_completed" if finalize_task else "agent_completed",
                        {"result": response.content, "steps": step},
                    )
                    return response.content

                for call in response.tool_calls:
                    
                    await self._event(context, "tool_started", {
                        "tool_call_id": call.id, "tool_name": call.name,
                    })

                    spec = self.registry.get(call.name)
                    idempotency_key = call.id if spec.supports_idempotency else None

                    cached = self.store.find_succeeded_tool_call(
                        call.name, idempotency_key, context.user_id,
                    ) if idempotency_key else None

                    if cached:
                        result = ToolExecutionResult.model_validate(
                            json.loads(cached["result_json"])
                        )
                    else:
                        
                        self.store.start_tool_call(
                            call.id, context.run_id, context.user_id, call.name,
                            spec.side_effect.value, arguments=call.arguments,
                            idempotency_key=idempotency_key,
                        )
                        try:
                            
                            result = await self.registry.execute(
                                call.name, call.arguments, agent_id=context.agent_id,
                                tool_call_id=call.id, idempotency_key=idempotency_key,
                            )
                        except BaseException:
                            
                            self.store.update_tool_call(
                                call.id, context.user_id,
                                status="outcome_unknown"
                                if spec.side_effect.value == "non_idempotent_write"
                                else "failed",
                            )
                            raise
                        
                        self.store.update_tool_call(
                            call.id, context.user_id, status=result.status,
                            result=result.model_dump(mode="json"),
                        )

                    tool_content = json.dumps(
                        result.output if result.status == "succeeded" else {
                            "error": result.error_code, "message": result.error_message,
                        }, ensure_ascii=False,
                    )
                    tool_message = RuntimeMessage(
                        session_id=context.session_id, task_id=context.task_id,
                        run_id=context.run_id, agent_id=context.agent_id,
                        role="tool", content=tool_content, tool_call_id=call.id,
                    )
                    messages.append(tool_message)
                    persisted_message_ids.append(self.store.append_message(
                        context.session_id, context.user_id, role="tool", content=tool_content,
                        task_id=context.task_id, run_id=context.run_id, agent_id=context.agent_id,
                    ))
                    
                    await self._event(context, "tool_completed", result.model_dump(mode="json"))

        if finalize_task:
            self.store.update_task_status(
                context.task_id, context.user_id, "failed", error="超过最大 Agent Loop 轮数",
            )
            self.store.finish_run(context.run_id, context.user_id, "failed")
        await self._event(
            context,
            "run_failed" if finalize_task else "agent_failed",
            {"error": "max_steps_exceeded"},
        )
        raise RuntimeError("超过最大 Agent Loop 轮数")

    @staticmethod
    def _stored_message(context: ExecutionContext, item: dict) -> RuntimeMessage:
        """
        把 SQLite 中读出的消息行转为 RuntimeMessage 对象。

        参数：
          context: 执行上下文（提供 session_id）
          item: SQLite 查询返回的字典行，至少包含 role 和 content

        返回：
          RuntimeMessage — 保留数据库中记录的原始 task_id/run_id/agent_id，
          不会改写成当前 context 的值（保持历史身份不变）。
        """
        return RuntimeMessage(
            session_id=context.session_id,
            task_id=item.get("task_id"),
            run_id=item.get("run_id"),
            agent_id=item.get("agent_id"),
            role=item["role"],
            content=item["content"],
        )

    async def _compact_backlog_with_limits(
        self,
        context: ExecutionContext,
        messages: list[RuntimeMessage],
        persisted_message_ids: list[int],
        latest_summary: dict | None,
        tool_schemas: list[dict],
    ) -> tuple[list[RuntimeMessage], list[int], dict | None]:
        '在单 Run 额度内推进最早连续积压区的压缩。'
        summarized_total = 0
        for _ in range(self.max_compaction_rounds_per_run):
            remaining_budget = self.max_history_compaction_per_run - summarized_total
            if remaining_budget <= 0 or not persisted_message_ids:
                break
            
            compaction, persisted_message_ids, latest_summary = (
                await self._compact_and_persist(
                    context, messages, persisted_message_ids, latest_summary, tool_schemas,
                    force_compaction=True,
                )
            )
            if compaction.summarized_count <= 0:
                raise RuntimeError("连续历史积压压缩没有推进 summary_end")
            summarized_total += compaction.summarized_count
            messages = compaction.messages
        return messages, persisted_message_ids, latest_summary

    async def _compact_ephemeral_until_ready(
        self,
        context: ExecutionContext,
        messages: list[RuntimeMessage],
        tool_schemas: list[dict],
    ) -> tuple[list[RuntimeMessage], CompactionResult]:
        """
        积压缺口存在时只压缩内存中的模型输入，不更新数据库累计摘要。

        为什么不能持久化？
          最近原文位于未摘要积压区之后，若把它持久化进滚动摘要会跳过中间的
          message_id，破坏"连续推进"的约束。

        参数：
          context: 执行上下文
          messages: 当前内存中的消息列表
          tool_schemas: 工具 Schema

        返回：
          (压缩后的消息列表, CompactionResult)
        """
        max_rounds = max(1, len(messages) + 1)
        previous_summary: str | None = None
        for _ in range(max_rounds):
            compaction = await self.context_manager.acompact_with_metadata(
                messages,
                tool_schemas=tool_schemas,
                context=context,
                previous_summary=previous_summary,
            )
            if compaction.summarized_count:
                messages = compaction.messages
                previous_summary = compaction.summary_content
            if not compaction.requires_additional_compaction:
                return messages, compaction
            if compaction.summarized_count <= 0:
                raise RuntimeError("临时上下文压缩没有推进")
        raise RuntimeError("临时上下文压缩超过安全轮数")

    async def _compact_until_ready(
        self,
        context: ExecutionContext,
        messages: list[RuntimeMessage],
        persisted_message_ids: list[int],
        latest_summary: dict | None,
        tool_schemas: list[dict],
    ) -> tuple[list[RuntimeMessage], list[int], dict | None, CompactionResult]:
        '连续执行有界压缩批次，直到结果可以安全发送给主模型。'
        max_rounds = max(1, len(persisted_message_ids) + 1)
        for _ in range(max_rounds):
            compaction, persisted_message_ids, latest_summary = await self._compact_and_persist(
                context, messages, persisted_message_ids, latest_summary, tool_schemas,
            )
            if compaction.summarized_count:
                messages = compaction.messages
            if not compaction.requires_additional_compaction:
                return messages, persisted_message_ids, latest_summary, compaction
            if compaction.summarized_count <= 0:
                raise RuntimeError("上下文分批压缩没有推进覆盖区间")
        raise RuntimeError("上下文分批压缩超过安全轮数")

    async def _compact_and_persist(
        self,
        context: ExecutionContext,
        messages: list[RuntimeMessage],
        persisted_message_ids: list[int],
        latest_summary: dict | None,
        tool_schemas: list[dict],
        *,
        force_compaction: bool = False,
    ) -> tuple[CompactionResult, list[int], dict | None]:
        '压缩当前模型上下文，并持久化三层长期记忆。'
        
        compaction = await self.context_manager.acompact_with_metadata(
            messages,
            tool_schemas=tool_schemas,
            context=context,
            previous_summary=(
                latest_summary["content"] if latest_summary is not None else None
            ),
            force_compaction=force_compaction,
        )
        if not compaction.summarized_count:
            
            return compaction, persisted_message_ids, latest_summary

        covered_ids = persisted_message_ids[:compaction.summarized_count]
        if not covered_ids:
            return compaction, persisted_message_ids, latest_summary

        if latest_summary is None:
            
            summary_id = self.store.append_summary(
                context.session_id, context.user_id,
                content=compaction.summary_content or "",
                token_estimate=compaction.token_estimate,
                start_message_id=covered_ids[0],
                end_message_id=covered_ids[-1],
            )
            latest_summary = self.store.get_latest_summary(
                context.session_id, context.user_id,
            )
            if latest_summary is None or latest_summary["summary_id"] != summary_id:
                raise RuntimeError("累计摘要持久化失败")
        else:
            
            latest_summary = self.store.update_rolling_summary(
                latest_summary["summary_id"],
                context.session_id, context.user_id,
                content=compaction.summary_content or "",
                token_estimate=compaction.token_estimate,
                end_message_id=covered_ids[-1],
            )

        summary_content = compaction.summary_content or ""
        self.store.record_memory_episode(
            new_id("episode"),
            context.session_id, context.user_id,
            start_message_id=covered_ids[0],
            end_message_id=covered_ids[-1],
            summary=summary_content,
            keywords=self._memory_keywords(summary_content),
            importance=self._memory_importance(summary_content),
            token_estimate=compaction.token_estimate,
        )

        current_state = self.store.get_memory_state(context.session_id, context.user_id)
        self.store.upsert_memory_state(
            context.session_id, context.user_id,
            self._merge_memory_state(
                current_state["state"] if current_state is not None else {},
                summary_content,
            ),
        )

        return (
            compaction,
            persisted_message_ids[compaction.summarized_count:],
            latest_summary,
        )

    @staticmethod
    def _memory_keywords(content: str) -> list[str]:
        """
        从情节摘要中提取轻量检索关键词。

        用途：SQLite 中的关键词检索（不依赖外部向量服务）。

        提取规则：
          - 英文标识符：字母开头 + 后续字母/数字/符号，2~32 字符
          - 中文片段：连续 2~12 个中文字符
          - 过滤掉摘要格式词（"历史摘要"、"user"、"assistant" 等）
          - 去重，最多返回 30 项

        参数：
          content: 摘要文本

        返回：
          list[str] — 关键词列表
        """
        
        words = re.findall(
            r"[A-Za-z][A-Za-z0-9_.+-]{1,31}|[一-鿿]{2,12}", content
        )
        
        stop_words = {"历史摘要", "已有累计摘要", "新增历史事实", "assistant", "user", "system"}
        return [
            word
            for word in dict.fromkeys(words)  
            if word.casefold() not in stop_words
        ][:30]

    def _bounded_memory_context(self, payload: dict) -> str:
        """
        给检索记忆分配独立的 token 预算，避免记忆内容挤爆主上下文。

        预算规则：最多 effective_limit 的 1/4，且不超过 1024 token。
        超限时用二分查找裁剪序列化内容，但始终保留前缀标记。

        参数：
          payload: 记忆数据（state + episodes）

        返回：
          str — 带有 "[结构化会话记忆]" 前缀的文本
        """
        prefix = "[结构化会话记忆]\n"
        serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        token_budget = max(8, min(1_024, self.context_manager.effective_limit // 4))
        counter = self.context_manager.token_counter
        full = prefix + serialized
        if counter(full) <= token_budget:
            return full
        
        low, high = 0, len(serialized)
        while low < high:
            middle = (low + high + 1) // 2
            if counter(prefix + serialized[:middle]) <= token_budget:
                low = middle
            else:
                high = middle - 1
        return prefix + serialized[:low]

    @staticmethod
    def _memory_importance(content: str) -> float:
        """
        根据摘要中的关键词标记估算情节的重要度。

        启发式评分（0~1 之间）：
          基础分 0.45
          每命中一个标记词 +0.08
          标记词包括：决定、最终、必须、约束、错误、失败、未完成、文件、报告
          上限 1.0

        这是启发式分数，用于情节召回排序，不代表模型置信度。

        参数：
          content: 摘要文本

        返回：
          float — 重要度分数（0~1）
        """
        markers = ("决定", "最终", "必须", "约束", "错误", "失败", "未完成", "文件", "报告")
        return min(1.0, 0.45 + sum(marker in content for marker in markers) * 0.08)

    @classmethod
    def _merge_memory_state(
        cls,
        current: dict,
        summary: str,
    ) -> dict[str, list[str]]:
        '把新摘要中的稳定信息按规则归类并合并进结构化状态。'
        fields = {
            "goals": list(current.get("goals", [])),
            "constraints": list(current.get("constraints", [])),
            "facts": list(current.get("facts", [])),
            "decisions": list(current.get("decisions", [])),
            "tool_results": list(current.get("tool_results", [])),
            "errors": list(current.get("errors", [])),
            "open_items": list(current.get("open_items", [])),
            "artifacts": list(current.get("artifacts", [])),
        }
        rules = (
            ("goals", ("目标", "需求", "希望")),
            ("constraints", ("约束", "必须", "不得", "限制")),
            ("decisions", ("决定", "最终", "采用", "选择", "使用")),
            ("tool_results", ("工具", "检索结果", "查询结果")),
            ("errors", ("错误", "失败", "异常")),
            ("open_items", ("未完成", "待处理", "下一步", "仍需")),
            ("artifacts", ("artifact", ".md", ".pdf", "生成文件", "报告文件")),
        )
        
        lines = [
            line.strip(" \t-*")
            for line in summary.splitlines()
            if line.strip() and not line.startswith("[历史摘要]")
        ]
        for line in lines:
            
            target = "facts"
            
            for field, markers in rules:
                if any(marker.casefold() in line.casefold() for marker in markers):
                    target = field
                    break
            compact = line[:500]  
            if compact not in fields[target]:
                fields[target].append(compact)
                fields[target] = fields[target][-50:]  
        return fields

    @staticmethod
    def _requires_artifact(query: str) -> bool:
        """
        判断用户是否明确要求生成、导出或下载制品文件。

        仅提到文件类型不代表要求生成文件。例如“总结一下 PDF”是读取并概括
        已上传资料，不应因为出现 ``pdf`` 就强制检查 Artifact。只有“生成 PDF
        报告”“导出 Markdown 文件”等同时包含生成动作和文件对象的表达才返回 True。

        参数：
          query: 用户问题

        返回：
          bool — 是否需要生成制品文件
        """
        normalized = query.casefold()
        action_markers = (
            "生成", "导出", "保存", "下载", "制作", "创建",
            "撰写", "写一份", "写成", "输出", "转成", "转换",
        )
        artifact_markers = (
            "报告", "markdown", ".md", "pdf", "文件", "文档",
        )
        return (
            any(marker in normalized for marker in action_markers)
            and any(marker in normalized for marker in artifact_markers)
        )

    def _run_has_artifact(self, context: ExecutionContext) -> bool:
        """
        检查本次 Run 是否已经生成了制品文件。

        参数：
          context: 执行上下文

        返回：
          bool — 是否已有制品
        """
        return any(
            artifact.get("run_id") == context.run_id
            for artifact in self.store.list_artifacts(
                context.session_id, context.user_id, limit=100,
            )
        )

    async def _fail_if_running(
        self, context: ExecutionContext, status: str, message: str, error_code: str,
    ) -> None:
        """
        安全地将任务标记为失败（仅在任务仍在运行中时）。

        防止重复标记：如果任务已经是终态（completed/failed/cancelled/timed_out），
        就不再覆盖。

        参数：
          context: 执行上下文
          status: 要设置的任务状态
          message: 错误描述
          error_code: 错误码
        """
        task = self.store.get_task(context.task_id, context.user_id)
        if task["status"] in {"completed", "failed", "cancelled", "timed_out"}:
            return  
        self.store.update_task_status(context.task_id, context.user_id, status, error=message)
        self.store.finish_run(context.run_id, context.user_id, status)
        await self._event(context, "run_failed", {"error": error_code, "message": message})

    async def _lease_heartbeat(self, context: ExecutionContext) -> None:
        """
        租约心跳：定期续租，告诉系统"这个 Run 还在运行"。

        作用：
          崩溃恢复机制的一部分。如果心跳停止（进程崩溃），
          其他 Worker 可以通过检查过期租约来发现并接管未完成的任务。

        参数：
          context: 执行上下文
        """
        run = self.store.get_run(context.run_id, context.user_id)
        worker_id = run["worker_id"]
        
        lease_seconds = max(60, int(self.lease_heartbeat_seconds * 3))
        while True:
            await asyncio.sleep(self.lease_heartbeat_seconds)
            self.store.renew_run_lease(
                context.run_id, context.user_id, worker_id, lease_seconds=lease_seconds,
            )

    async def _event(
        self, context: ExecutionContext, event_type: str, payload: dict,
    ) -> None:
        """
        发送一条运行时事件（通过 EventBus 广播给所有订阅者）。

        参数：
          context: 执行上下文（提供 session/task/run/agent ID）
          event_type: 事件类型（如 "run_started", "tool_started", "run_completed"）
          payload: 事件负载数据（dict）
        """
        await self.event_bus.publish(RuntimeEvent(
            session_id=context.session_id, task_id=context.task_id, run_id=context.run_id,
            agent_id=context.agent_id, type=event_type, payload=payload,
        ), context.user_id)

    async def _generate_with_retry(
        self, context: ExecutionContext, messages: list[RuntimeMessage],
    ) -> ModelResponse:
        '调用模型生成回复（含自动重试）。'
        attempts = 0
        while True:
            attempts += 1
            try:
                return await self.model.generate(
                    messages,
                    self.registry.schemas_for_agent(context.agent_id),
                )
            except (TransientModelError, TimeoutError, ConnectionError) as exc:
                if attempts >= self.model_retry_policy.max_attempts:
                    raise  
                delay = self.model_retry_policy.delay_for(attempts)
                
                await self._event(context, "retry_scheduled", {
                    "target": "model",
                    "attempt": attempts + 1,
                    "delay_seconds": delay,
                    "error": type(exc).__name__,
                })
                await self.sleep(delay)  
