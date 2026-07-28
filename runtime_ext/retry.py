from __future__ import annotations

from dataclasses import dataclass

class TransientToolError(RuntimeError):
    '工具发生的"临时性"错误，重试可能成功。'
    pass

class TransientModelError(RuntimeError):
    '模型调用发生的"临时性"错误，重试可能成功。'
    pass

class OutcomeUnknownError(RuntimeError):
    '工具可能已经产生了副作用，但调用方没有收到确定的结果。'
    pass

@dataclass(frozen=True)
class RetryPolicy:
    '指数退避重试策略。'

    max_attempts: int = 1

    initial_delay: float = 0.25

    multiplier: float = 2.0

    max_delay: float = 4.0

    def delay_for(self, completed_attempts: int) -> float:
        '计算第 N 次重试前应该等待多少秒。'
        return min(
            self.initial_delay * self.multiplier ** max(0, completed_attempts - 1),
            self.max_delay,
        )
