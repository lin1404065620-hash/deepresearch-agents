from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

def new_id(prefix: str) -> str:
    '生成带类型前缀的唯一 ID。'
    return f"{prefix}_{uuid4().hex}"

class ExecutionContext(BaseModel):
    '执行上下文。代表"谁，在哪个会话里，用哪个 Agent，执行哪次任务"。'

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    user_id: str = Field(min_length=1)

    session_id: str = Field(min_length=1)

    task_id: str = Field(min_length=1)

    run_id: str = Field(min_length=1)

    agent_id: str = Field(min_length=1)

    workspace: Path

    permissions: frozenset[str] = frozenset()

    @model_validator(mode="after")
    def validate_distinct_execution_ids(self) -> "ExecutionContext":
        """
        Pydantic 模型校验器，在对象创建完成后自动执行。
        确保 session_id、task_id、run_id 三个 ID 互不相同。
        如果相同说明代码有 bug，直接报错终止。
        """
        values = (self.session_id, self.task_id, self.run_id)
        if len(set(values)) != len(values):  
            raise ValueError("session_id、task_id 和 run_id 必须相互独立")
        return self

    @classmethod
    def create(
        cls,
        user_id: str,
        session_id: str,
        workspace: Path,
        agent_id: str = "main",
    ) -> "ExecutionContext":
        '快捷创建方法。只需传最关键的几个参数，task_id 和 run_id 自动生成。'
        return cls(
            user_id=user_id,
            session_id=session_id,
            task_id=new_id("task"),
            run_id=new_id("run"),
            agent_id=agent_id,
            workspace=workspace,
        )

    def for_child_agent(self, agent_id: str, permissions: frozenset[str] | None = None) -> "ExecutionContext":
        '从当前（父 Agent）上下文，派生子 Agent 的上下文。'
        return self.model_copy(
            update={
                "agent_id": agent_id,
                "permissions": self.permissions if permissions is None else permissions,
            }
        )

class ToolCallRequest(BaseModel):
    """
    模型说"我要调用某个工具"时发出的请求。

    比如模型分析完用户问题后决定搜索互联网，会生成：
      ToolCallRequest(
          id="call_abc123",
          name="internet_search",
          arguments={"query": "北京今天天气", "topic": "general"}
      )

    含义：用编号 call_abc123，调用 internet_search 工具，搜索"北京今天天气"。
    """

    id: str = Field(default_factory=lambda: new_id("call"))

    name: str = Field(min_length=1)

    arguments: dict[str, Any] = Field(default_factory=dict)

class ModelResponse(BaseModel):
    '模型的一次回复。'

    content: str = ""

    tool_calls: list[ToolCallRequest] = Field(default_factory=list)

class ToolExecutionResult(BaseModel):
    '工具执行完成后的结果报告，即某一次工具调用执行完之后，Runtime 对这次执行结果生成的统一报告。'

    tool_call_id: str

    tool_name: str

    status: Literal["succeeded", "failed", "outcome_unknown"]

    output: Any = None

    error_code: str | None = None

    error_message: str | None = None

    attempts: int = Field(default=1, ge=1)

    duration_ms: int = Field(default=0, ge=0)

class RuntimeMessage(BaseModel):
    '对话历史中的一条消息。'

    id: str = Field(default_factory=lambda: new_id("msg"))

    session_id: str

    task_id: str | None = None

    run_id: str | None = None

    agent_id: str | None = None

    role: Literal["system", "user", "assistant", "tool"]

    content: str = ""

    tool_calls: list[ToolCallRequest] = Field(default_factory=list)

    tool_call_id: str | None = None

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class RuntimeEvent(BaseModel):
    '运行时事件，在 EventBus（事件总线）上传输的数据包。'

    event_id: int | None = None

    session_id: str

    task_id: str | None = None

    run_id: str | None = None

    agent_id: str | None = None

    type: str

    payload: dict[str, Any] = Field(default_factory=dict)

    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class ArtifactRef(BaseModel):
    """
    文件制品的引用信息。

    Agent 生成的 Markdown、PDF 等文件叫"制品"（Artifact）。
    ArtifactRef 只是制品的"身份证"（元信息），不包含文件内容本身。

    类比图书馆索书卡：
      卡片上写"《三体》→ 3楼 A区 12排 5层"，
      但卡片不是书本身。ArtifactRef 就是这张索书卡。

    前端通过 artifact_id → /api/artifacts/{id}/download 下载实际文件。
    """

    artifact_id: str

    name: str

    relative_path: str

    media_type: str | None = None

    size: int | None = None
