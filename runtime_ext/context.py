from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from collections.abc import Iterator

from .models import ExecutionContext

_runtime_context: ContextVar[ExecutionContext | None] = ContextVar(
    "runtime_execution_context", default=None
)

def get_runtime_context(*, required: bool = True) -> ExecutionContext | None:
    '获取当前协程的执行上下文（相当于"刷手环"）。'
    context = _runtime_context.get()
    if required and context is None:
        raise RuntimeError("当前调用不在 Agent Runtime 执行上下文中")
    return context

@contextmanager
def runtime_scope(context: ExecutionContext) -> Iterator[ExecutionContext]:
    '上下文管理器：临时设置当前协程的执行上下文，退出时自动恢复。'
    token = _runtime_context.set(context)
    try:
        yield context
    finally:
        _runtime_context.reset(token)
