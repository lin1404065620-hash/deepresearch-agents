from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path

from .models import ExecutionContext
from .api import MAIN_AGENT_PERMISSIONS
from .store import MemoryStore

class RecoveryCoordinator:
    """
    把 MemoryStore 生成的恢复 Run 重新提交给 Agent Runtime。

    这是任务级安全重跑，不恢复 Python 局部变量或模型调用现场。Store 已经负责
    排除非幂等结果未知和超过恢复次数的任务，因此这里只消费可安全重跑的 Run。
    """

    def __init__(
        self,
        *,
        store: MemoryStore,
        runner: Callable[[ExecutionContext, str], Awaitable[str]],
        workspace_root: Path,
        worker_id: str,
        max_recovery_attempts: int = 3,
    ):
        self.store = store
        self.runner = runner
        self.workspace_root = workspace_root
        self.worker_id = worker_id
        self.max_recovery_attempts = max_recovery_attempts
        self._background_tasks: set[asyncio.Task] = set()

    async def recover_once(self, *, background: bool = False) -> list[dict]:
        """扫描一次失联 Run，并同步执行或提交后台安全重跑。"""
        runs = self.store.recover_stale_runs(
            self.worker_id,
            max_recovery_attempts=self.max_recovery_attempts,
        )
        jobs = []
        for run in runs:
            task = self.store.get_task(run["task_id"], run["user_id"])
            context = ExecutionContext(
                user_id=run["user_id"],
                session_id=run["session_id"],
                task_id=run["task_id"],
                run_id=run["run_id"],
                agent_id="main",
                workspace=self.workspace_root / run["session_id"],
                
                permissions=MAIN_AGENT_PERMISSIONS,
            )
            job = self._run_safely(context, task["query"])
            if background:
                background_task = asyncio.create_task(
                    job, name=f"runtime-recovery-{run['run_id']}",
                )
                self._background_tasks.add(background_task)
                background_task.add_done_callback(self._background_tasks.discard)
            else:
                jobs.append(job)
        if jobs:
            await asyncio.gather(*jobs)
        return runs

    async def _run_safely(self, context: ExecutionContext, query: str) -> None:

        try:
            await self.runner(context, query)
        except asyncio.CancelledError:
            raise
        except Exception:
            return
