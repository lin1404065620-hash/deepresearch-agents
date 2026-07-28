from __future__ import annotations

import inspect
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from .models import ExecutionContext, RuntimeMessage

class ContextOverflowError(RuntimeError):
    """
    压缩已经尽力了，但上下文仍然超过输入预算。

    抛出时机：
      系统消息（system prompt）和工具定义（tool schemas）本身就太大了，
      即使把所有对话消息都删掉、摘要缩到最短，还是装不下。

    这种情况只能靠调整 max_tokens 或缩减 system prompt 来解决，
    压缩器已经无能为力。
    """
    pass

class MessageOrderError(ValueError):
    """
    system 消息出现在了对话中间（而不是开头），顺序不对。

    正常对话顺序应该是：
      [system prompt] [user] [assistant] [user] [assistant] ...
    如果出现：
      [user] [assistant] [system] [user] ...
    说明调用方把 system 消息插错位置了，继续重排会改变原始语义。
    """
    pass

@dataclass(frozen=True)
class CompactionResult:
    '一次上下文压缩的完整结果。'

    messages: list[RuntimeMessage]

    summary_content: str | None = None

    summarized_count: int = 0

    token_estimate: int = 0

    summary_source: str | None = None

    used_semantic_summary: bool = False

    requires_additional_compaction: bool = False

SummaryGenerator = Callable[[str], str | Awaitable[str]]

class ContextManager:
    '对话历史的智能压缩引擎。'

    SUMMARY_PREFIXES = ("[历史摘要]", "[持久化历史摘要]")

    def __init__(
        self,
        *,
        max_tokens: int = 8_000,
        keep_recent: int = 8,
        token_counter: Callable[[str], int] | None = None,
        output_budget: int = 0,
        safety_margin: int = 0,
        summary_generator: SummaryGenerator | None = None,
        summary_input_budget: int = 6_000,
        max_compaction_messages: int = 200,
    ):
        '参数： max_tokens: int = 8000 模型的总上下文窗口大小。默认 8000。 例如 GPT-4 可能是 8192 或 128000。'
        
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if keep_recent < 0:
            raise ValueError("keep_recent cannot be negative")
        if output_budget < 0 or safety_margin < 0:
            raise ValueError("output_budget and safety_margin cannot be negative")
        if output_budget + safety_margin >= max_tokens:
            raise ValueError("output_budget + safety_margin must be smaller than max_tokens")
        if summary_input_budget <= 0:
            raise ValueError("summary_input_budget must be positive")
        if max_compaction_messages <= 0:
            raise ValueError("max_compaction_messages must be positive")

        self.max_tokens = max_tokens
        self.keep_recent = keep_recent
        self.output_budget = output_budget
        self.safety_margin = safety_margin
        
        self.effective_limit = max_tokens - output_budget - safety_margin
        self.token_counter = token_counter or self._build_default_token_counter()
        self.summary_generator = summary_generator
        self.summary_input_budget = summary_input_budget
        self.max_compaction_messages = max_compaction_messages

    @staticmethod
    def _build_default_token_counter() -> Callable[[str], int]:
        '构造默认的 token 计数函数。'
        try:
            import tiktoken

            encoding = tiktoken.get_encoding("cl100k_base")
            return lambda text: max(1, len(encoding.encode(text)))
        except (ImportError, OSError, ValueError):
            return lambda text: max(1, (len(text.encode("utf-8")) + 2) // 3)

    def count_tokens(
        self,
        messages: list[RuntimeMessage],
        *,
        tool_schemas: list[dict[str, Any]] | None = None,
    ) -> int:
        '估算完整模型输入的 token 数，不只是统计消息正文。'
        total = 2  
        for message in messages:
            total += 4  
            total += self.token_counter(message.role)  
            total += self.token_counter(message.content)  
            if message.tool_call_id:
                
                total += self.token_counter(message.tool_call_id)
            if message.tool_calls:
                
                payload = [
                    call.model_dump(mode="json")
                    for call in message.tool_calls
                ]
                total += self.token_counter(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        if tool_schemas:
            
            total += self.token_counter(
                json.dumps(tool_schemas, ensure_ascii=False, sort_keys=True),
            )
        return total

    def compact(
        self,
        messages: list[RuntimeMessage],
        *,
        tool_schemas: list[dict[str, Any]] | None = None,
        context: ExecutionContext | None = None,
        previous_summary: str | None = None,
    ) -> list[RuntimeMessage]:
        '压缩消息列表（最简接口），只返回压缩后的消息。'
        return self.compact_with_metadata(
            messages,
            tool_schemas=tool_schemas,
            context=context,
            previous_summary=previous_summary,
        ).messages

    def compact_with_metadata(
        self,
        messages: list[RuntimeMessage],
        *,
        tool_schemas: list[dict[str, Any]] | None = None,
        context: ExecutionContext | None = None,
        previous_summary: str | None = None,
    ) -> CompactionResult:
        '按时间顺序对历史消息执行有界、连续的滚动压缩。\n\n注意：此方法始终使用确定性摘要（规则拼接），不会调用 LLM 语义摘要器。 如需语义摘要，使用 acompact_with_metadata()。'
        return self._compact(
            messages,
            tool_schemas=tool_schemas,
            context=context,
            previous_summary=previous_summary,
            generated_summary=None,  
        )

    async def acompact_with_metadata(
        self,
        messages: list[RuntimeMessage],
        *,
        tool_schemas: list[dict[str, Any]] | None = None,
        context: ExecutionContext | None = None,
        previous_summary: str | None = None,
        force_compaction: bool = False,
    ) -> CompactionResult:
        '异步压缩接口，优先使用 LLM 语义摘要。'
        
        deterministic = self._compact(
            messages,
            tool_schemas=tool_schemas,
            context=context,
            previous_summary=previous_summary,
            generated_summary=None,
            force_compaction=force_compaction,
        )

        if (
            self.summary_generator is None           
            or deterministic.summarized_count == 0    
            or not deterministic.summary_source       
        ):
            return deterministic

        try:
            generated = self.summary_generator(deterministic.summary_source)
            
            if inspect.isawaitable(generated):
                generated = await generated
            if not isinstance(generated, str) or not generated.strip():
                raise ValueError("摘要模型返回了空内容")
            
            return self._compact(
                messages,
                tool_schemas=tool_schemas,
                context=context,
                previous_summary=previous_summary,
                generated_summary=generated.strip(),
                force_compaction=force_compaction,
            )
        except Exception:
            
            return deterministic

    def _compact(
        self,
        messages: list[RuntimeMessage],
        *,
        tool_schemas: list[dict[str, Any]] | None,
        context: ExecutionContext | None,
        previous_summary: str | None,
        generated_summary: str | None,
        force_compaction: bool = False,
    ) -> CompactionResult:
        '按时间顺序执行有界、连续的滚动压缩。'
        
        self._validate_message_order(messages)
        identity = self._resolve_identity(messages, context)
        original = list(messages)

        non_system_count = sum(message.role != "system" for message in original)
        count_requires_compaction = (
            non_system_count > self.max_compaction_messages + self.keep_recent
        )
        if (
            self.count_tokens(original, tool_schemas=tool_schemas) <= self.effective_limit
            and not count_requires_compaction
            and not force_compaction
        ):
            return CompactionResult(messages=original)

        system_messages = [
            message for message in original
            if message.role == "system" and not self._is_summary(message)
        ]
        embedded_summaries = [
            self._strip_summary_prefix(message.content)
            for message in original
            if message.role == "system" and self._is_summary(message)
        ]
        if previous_summary is None and embedded_summaries:
            previous_summary = "\n".join(embedded_summaries)

        non_system = [message for message in original if message.role != "system"]
        if force_compaction:

            old = list(non_system)
            recent = []
        elif self.keep_recent == 0:
            
            old = list(non_system)
            recent: list[RuntimeMessage] = []
        else:
            recent = non_system[-self.keep_recent:]   
            old = non_system[:-self.keep_recent]       

        while recent and self._base_tokens(system_messages, recent, tool_schemas) > self.effective_limit:
            if len(recent) == 1:
                break  
            moved, recent = self._remove_oldest_turn(recent)
            old.extend(moved)

        all_old = old
        batch_size = min(len(all_old), self.max_compaction_messages)
        summary_source = None
        while batch_size > 0:
            try:
                summary_source = self._summary_source(
                    previous_summary,
                    all_old[:batch_size],
                )
                break
            except ContextOverflowError:
                batch_size //= 2
        if all_old and batch_size == 0:
            raise ContextOverflowError("摘要输入预算无法容纳一条旧消息")
        old = all_old[:batch_size]
        deferred = all_old[batch_size:]
        requires_additional = bool(deferred)

        summary_content = None
        if summary_source:

            summary_content = generated_summary or self._deterministic_summary(previous_summary, old)

        summary_message = (
            self._summary_message(identity, summary_content)
            if summary_content
            else None
        )
        compacted = (
            system_messages
            + ([summary_message] if summary_message else [])
            + deferred
            + recent
        )

        if requires_additional:
            return CompactionResult(
                messages=compacted,
                summary_content=summary_content,
                summarized_count=len(old),
                token_estimate=self.token_counter(summary_content) if summary_content else 0,
                summary_source=summary_source,
                used_semantic_summary=generated_summary is not None,
                requires_additional_compaction=True,
            )

        if (
            summary_message
            and recent
            and self.count_tokens(compacted, tool_schemas=tool_schemas) > self.effective_limit
        ):
            
            minimum_summary_budget = min(
                1_024,                                   
                max(128, self.effective_limit // 4),     
            )
            minimum_summary_budget = min(
                minimum_summary_budget,
                max(8, self.effective_limit // 2),       
            )
            
            fixed = system_messages + [self._summary_message(identity, "")] + recent[:-1]
            available_for_last = (
                self.effective_limit
                - self.count_tokens(fixed, tool_schemas=tool_schemas)
                - minimum_summary_budget
                - 6  
            )
            if available_for_last > 0:
                
                recent[-1] = recent[-1].model_copy(
                    update={
                        "content": self._truncate_preserving_ends(
                            recent[-1].content,
                            available_for_last,
                        ),
                    },
                )
                compacted = system_messages + [summary_message] + recent

        if summary_message and self.count_tokens(compacted, tool_schemas=tool_schemas) > self.effective_limit:
            available = self._available_summary_tokens(system_messages, recent, tool_schemas)
            if available > 0:
                shortened = (
                    self._fit_deterministic_summary(previous_summary, old, available)
                    if generated_summary is None
                    else self._truncate_preserving_ends(summary_message.content, available)
                )
                summary_message = summary_message.model_copy(update={"content": shortened})
                summary_content = shortened
                compacted = system_messages + [summary_message] + recent

        if self.count_tokens(compacted, tool_schemas=tool_schemas) > self.effective_limit and recent:
            last = recent[-1]
            prefix = system_messages + ([summary_message] if summary_message else []) + recent[:-1]
            available = self.effective_limit - self.count_tokens(prefix, tool_schemas=tool_schemas) - 6
            if available > 0:
                recent[-1] = last.model_copy(
                    update={"content": self._truncate_preserving_ends(last.content, available)},
                )
                compacted = system_messages + ([summary_message] if summary_message else []) + recent

        final_tokens = self.count_tokens(compacted, tool_schemas=tool_schemas)
        if final_tokens > self.effective_limit and recent:
            last = recent[-1]
            reduced_limit = max(
                0,
                self.token_counter(last.content) - (final_tokens - self.effective_limit) - 2,
            )
            recent[-1] = last.model_copy(
                update={
                    "content": self._truncate_preserving_ends(last.content, reduced_limit),
                },
            )
            compacted = system_messages + ([summary_message] if summary_message else []) + recent
            final_tokens = self.count_tokens(compacted, tool_schemas=tool_schemas)

        if final_tokens > self.effective_limit:
            raise ContextOverflowError(
                f"上下文压缩后仍超限: {final_tokens}>{self.effective_limit}",
            )

        return CompactionResult(
            messages=compacted,
            summary_content=summary_content,
            summarized_count=len(old),
            token_estimate=self.token_counter(summary_content) if summary_content else 0,
            summary_source=summary_source,
            used_semantic_summary=generated_summary is not None,
        )

    def _base_tokens(
        self,
        system_messages: list[RuntimeMessage],
        recent: list[RuntimeMessage],
        tool_schemas: list[dict[str, Any]] | None,
    ) -> int:
        """
        计算 system + recent 的基础 token 数（不含摘要）。

        用于判断"即使不写摘要，光 system 消息和最近消息就已经超了"的情况。
        """
        return self.count_tokens(system_messages + recent, tool_schemas=tool_schemas)

    def _available_summary_tokens(
        self,
        system_messages: list[RuntimeMessage],
        recent: list[RuntimeMessage],
        tool_schemas: list[dict[str, Any]] | None,
    ) -> int:
        '计算摘要还能占用多少 token。'
        empty_summary = self._summary_message(
            self._resolve_identity(system_messages + recent, None),
            "",
        )
        fixed = self.count_tokens(
            system_messages + [empty_summary] + recent,
            tool_schemas=tool_schemas,
        )
        return max(0, self.effective_limit - fixed)

    @staticmethod
    def _remove_oldest_turn(
        messages: list[RuntimeMessage],
    ) -> tuple[list[RuntimeMessage], list[RuntimeMessage]]:
        """
        从消息列表中移除最旧的一轮对话。

        一轮对话以 user 消息为界：
          例如 [user, assistant, tool, user, assistant] → 移除 [user, assistant, tool]

        参数：
          messages: 消息列表

        返回：
          (被移除的消息, 剩余的消息)
        """
        if not messages:
            return [], []
        
        boundary = 1
        while boundary < len(messages) and messages[boundary].role != "user":
            boundary += 1
        return messages[:boundary], messages[boundary:]

    def _deterministic_summary(
        self,
        previous_summary: str | None,
        old: list[RuntimeMessage],
    ) -> str:
        """
        生成确定性摘要（规则拼接，不依赖 LLM）。

        格式：
          [历史摘要]
          已有累计摘要：（如果有的话，截取前 600 字符）
          新增历史事实与决策：
          - user: 用户消息的快照（前 160 字符）
          - assistant: AI 回复的快照（前 160 字符）
          ...

        参数：
          previous_summary: 之前的累计摘要（在前缀之后追加）
          old: 要被压缩的旧消息列表

        返回：
          str — 格式化的摘要文本
        """
        sections = ["[历史摘要]"]
        if previous_summary:
            sections.append("已有累计摘要：\n" + self._snapshot(previous_summary, 600))
        if old:
            lines = [
                f"- {message.role}: {self._snapshot(message.content, 160)}"
                for message in old
            ]
            sections.append("新增历史事实与决策：\n" + "\n".join(lines))
        return "\n".join(sections)

    def _fit_deterministic_summary(
        self,
        previous_summary: str | None,
        old: list[RuntimeMessage],
        token_limit: int,
    ) -> str:
        """
        在指定的 token 预算内生成确定性摘要。

        策略：从宽到严逐步降低每条消息的快照长度限制（80 → 76 → ... → 8），
        找到第一个能在预算内的版本。

        这样做的好处：逐条缩短每条消息，而不是直接砍掉中间的消息，
        保证每条消息都有至少一点点代表信息。

        参数：
          previous_summary: 之前的累计摘要
          old: 旧的非 system 消息列表
          token_limit: 摘要可用的 token 上限

        返回：
          str — 符合预算的摘要文本
        """
        
        previous_lines = [
            line.strip()
            for line in (previous_summary or "").splitlines()
            if line.strip() and line.strip() != "[历史摘要]"
        ][-12:]

        for snapshot_limit in range(80, 7, -4):
            sections = ["[历史摘要]"]
            if previous_lines:
                sections.append("已有累计摘要：")
                sections.extend(
                    "  " + self._snapshot(line, snapshot_limit)
                    for line in previous_lines
                )
            sections.extend(
                f"- {message.role}: {self._snapshot(message.content, snapshot_limit)}"
                for message in old
            )
            candidate = "\n".join(sections)
            if self.token_counter(candidate) <= token_limit:
                return candidate

        minimal_lines = ["[历史摘要]"]
        minimal_lines.extend(
            "  " + self._snapshot(line, 12)
            for line in previous_lines
        )
        minimal_lines.extend(
            f"- {message.role}: {self._snapshot(message.content, 8)}"
            for message in old
        )
        minimal = "\n".join(minimal_lines)
        
        return self._truncate_preserving_ends(minimal, token_limit)

    def _summary_source(
        self,
        previous_summary: str | None,
        old: list[RuntimeMessage],
    ) -> str:
        '构造有硬预算的摘要原始材料（JSON 格式），供语义摘要器使用。'
        preserve = ["目标", "事实", "决策", "约束", "工具结果", "错误", "未完成事项"]
        minimum_snapshot_tokens = 8

        def build_payload(previous: str, snapshot_tokens: int) -> str:
            payload = {
                "已有摘要": previous,
                "消息": [
                    {
                        "role": message.role,
                        "content": (
                            self._truncate_preserving_ends(
                                message.content,
                                snapshot_tokens,
                            )
                            if message.content
                            else ""
                        ),
                    }
                    for message in old
                ],
                "保留": preserve,
            }
            return json.dumps(payload, ensure_ascii=False)

        minimum_source = build_payload("", minimum_snapshot_tokens)
        if self.token_counter(minimum_source) > self.summary_input_budget:
            raise ContextOverflowError("摘要输入预算无法容纳当前连续消息批次的最小有效快照")

        previous = ""
        if previous_summary:
            low = 0
            high = min(800, self.token_counter(previous_summary))
            while low < high:
                middle = (low + high + 1) // 2
                candidate = self._truncate_preserving_ends(previous_summary, middle)
                if (
                    self.token_counter(build_payload(candidate, minimum_snapshot_tokens))
                    <= self.summary_input_budget
                ):
                    low = middle
                else:
                    high = middle - 1
            previous = self._truncate_preserving_ends(previous_summary, low) if low else ""

        for snapshot_tokens in (240, 160, 80, 40, 20, minimum_snapshot_tokens):
            source = build_payload(previous, snapshot_tokens)
            if self.token_counter(source) <= self.summary_input_budget:
                return source
        return build_payload(previous, minimum_snapshot_tokens)

    @staticmethod
    def _snapshot(text: str, limit: int) -> str:
        '截取文本的快照：在 limit 个字符内保留首尾关键信息。'
        normalized = " ".join(text.split())  
        if len(normalized) <= limit:
            return normalized
        half = max(1, (limit - 1) // 2)  
        return normalized[:half] + "…" + normalized[-half:]

    def _truncate_text(self, text: str, token_limit: int) -> str:
        """
        按 token 上限裁剪文本（保留开头部分）。

        使用二分查找找到最大的子串长度，使 token 数不超限。

        参数：
          text: 原始文本
          token_limit: token 上限

        返回：
          str — 裁剪后的文本
        """
        if token_limit <= 0:
            return ""
        if self.token_counter(text) <= token_limit:
            return text
        
        low, high = 0, len(text)
        while low < high:
            middle = (low + high + 1) // 2
            candidate = text[:middle]
            if self.token_counter(candidate) <= token_limit:
                low = middle
            else:
                high = middle - 1
        return text[:low]

    def _truncate_preserving_ends(self, text: str, token_limit: int) -> str:
        '裁剪文本，同时保留开头和结尾（与 _snapshot 逻辑相同，但按 token 计算）。'
        if token_limit <= 0 or self.token_counter(text) <= token_limit:
            return text if token_limit > 0 else ""
        low, high = 0, len(text)
        while low < high:
            middle = (low + high + 1) // 2
            half = max(1, (middle - 1) // 2)
            candidate = text[:half] + "…" + text[-half:]
            if self.token_counter(candidate) <= token_limit:
                low = middle
            else:
                high = middle - 1
        half = max(1, (low - 1) // 2)
        return text[:half] + "…" + text[-half:]

    @classmethod
    def _is_summary(cls, message: RuntimeMessage) -> bool:
        '判断一条 system 消息是否为摘要消息（而非原始 system prompt）。'
        return message.content.startswith(cls.SUMMARY_PREFIXES)

    @classmethod
    def _strip_summary_prefix(cls, content: str) -> str:
        '去掉摘要消息的前缀标记，返回纯摘要内容。 这段 _strip_summary_prefix() 用来去掉摘要消息开头的来源标记并返回纯摘要正文：它依次检查 content 是否以 SUMMARY_PREFIXES 中的 某个前缀开头，例如 [历史摘要] 或 [持久化历史摘要]；匹配后就删除该前缀，并用 lstrip(" ") 清理前缀后多余的换行 和空格，例如把 "[历史摘要] 前面讨论了 SQLite" 处理成 "前面讨论了 SQLite"；如果没有匹配任何摘要前缀，则说明'
        for prefix in cls.SUMMARY_PREFIXES:
            if content.startswith(prefix):
                return content[len(prefix):].lstrip("\n ")
        return content

    @staticmethod
    def _summary_message(
        identity: tuple[str, str | None, str | None, str | None],
        content: str,
    ) -> RuntimeMessage:
        '构造一条摘要消息（role="system"）。 这段 _summary_message() 用来把一段摘要文本包装成一条标 准的 RuntimeMessage：它先从 identity 元组中拆出 session_id、task_id、run_id、agent_id，然后创建一条 role="system" 的消息，并把 传入的 content 作为摘要正文，同时补齐对应的会话、任务、运行和 Agent 标识，最终返回一条可以直接放入上下文中的系统摘要消息。'
        session_id, task_id, run_id, agent_id = identity
        return RuntimeMessage(
            session_id=session_id,
            task_id=task_id,
            run_id=run_id,
            agent_id=agent_id,
            role="system",
            content=content,
        )

    @staticmethod
    def _resolve_identity(
        messages: list[RuntimeMessage],
        context: ExecutionContext | None,
    ) -> tuple[str, str | None, str | None, str | None]:
        '确定消息的归属身份（session_id, task_id, run_id, agent_id）。'
        if context is not None:
            return context.session_id, context.task_id, context.run_id, context.agent_id
        if not messages:
            return "", None, None, None
        
        session_ids = {message.session_id for message in messages}
        if len(session_ids) != 1:
            raise ValueError("待压缩消息必须属于同一个 session")
        last = messages[-1]
        return last.session_id, last.task_id, last.run_id, last.agent_id

    @staticmethod
    def _validate_message_order(messages: list[RuntimeMessage]) -> None:
        '校验消息顺序：system 消息必须全部在对话开头。'
        conversation_started = False
        for message in messages:
            if message.role == "system":
                if conversation_started:
                    raise MessageOrderError("system 消息只能出现在对话开头")
            else:
                conversation_started = True
