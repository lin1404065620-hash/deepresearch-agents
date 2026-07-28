from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any

from .models import RuntimeEvent, new_id

def calculate_file_sha256(file_path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """分块计算文件 SHA-256，避免大文件一次性读入内存。"""
    hasher = hashlib.sha256()
    with file_path.open("rb") as file:
        while chunk := file.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()

class ResourceNotFound(LookupError):
    """资源不存在，或存在但不属于当前用户（租户隔离）。"""
    pass

class ArtifactIntegrityError(RuntimeError):
    """
    制品完整性校验失败。
    当 verify_artifact_file() 发现文件的 SHA256 哈希
    跟存入数据库时不一致时抛出，说明文件被篡改或损坏。
    """
    pass

class SessionBusyError(RuntimeError):
    """会话仍有运行中或恢复中的 Run，当前不能删除。"""
    pass

class SessionStatus(StrEnum):
    """会话状态"""
    ACTIVE = "active"       
    ARCHIVED = "archived"   

class TaskStatus(StrEnum):
    """任务状态"""
    PENDING = "pending"                         
    RUNNING = "running"                         
    COMPLETED = "completed"                     
    FAILED = "failed"                           
    CANCELLED = "cancelled"                     
    TIMED_OUT = "timed_out"                     
    FAILED_REQUIRES_REVIEW = "failed_requires_review"  

class RunStatus(StrEnum):
    """运行状态（一次 task 的一次执行）"""
    RUNNING = "running"                   
    RECOVERING = "recovering"             
    RECOVERING_CLAIMED = "recovering_claimed"  
    COMPLETED = "completed"               
    FAILED = "failed"                     
    CANCELLED = "cancelled"               
    TIMED_OUT = "timed_out"               
    INTERRUPTED = "interrupted"           

class MessageRole(StrEnum):
    """消息角色"""
    SYSTEM = "system"         
    USER = "user"             
    ASSISTANT = "assistant"   
    TOOL = "tool"             

class ToolCallStatus(StrEnum):
    """工具调用状态"""
    STARTED = "started"               
    SUCCEEDED = "succeeded"           
    FAILED = "failed"                 
    OUTCOME_UNKNOWN = "outcome_unknown"  

class StoredSideEffect(StrEnum):
    """工具副作用级别（存在数据库中）"""
    NONE = "none"                               
    READ = "read"                               
    IDEMPOTENT_WRITE = "idempotent_write"       
    NON_IDEMPOTENT_WRITE = "non_idempotent_write"  

def _now() -> str:
    """返回当前 UTC 时间的 ISO 格式字符串。"""
    return datetime.now(timezone.utc).isoformat()

def _enum_value(value: StrEnum | str, enum_type: type[StrEnum]) -> str:
    """
    把 StrEnum 或字符串统一转成标准字符串值。

    比如：
      _enum_value(TaskStatus.RUNNING, TaskStatus) → "running"
      _enum_value("completed", TaskStatus)         → "completed"
      _enum_value("invalid", TaskStatus)           → 抛错

    作用：兼容旧代码传字符串的情况，同时校验值是否合法。
    """
    return enum_type(value).value

class MemoryStore:
    """单 Worker SQLite 持久化层；同一实例的连接访问由 RLock 串行化。"""

    SCHEMA_VERSION = 4

    MAX_PAGE_SIZE = 500

    MAX_MEMORY_STATE_ITEMS = 50
    MAX_MEMORY_STATE_ITEM_LENGTH = 500

    def __init__(self, database_path: Path | str):
        '初始化数据库连接。'
        
        database_path = Path(database_path)
        
        database_path.parent.mkdir(parents=True, exist_ok=True)

        self.database_path = database_path
        
        self._lock = threading.RLock()

        self.connection = sqlite3.connect(database_path, check_same_thread=False, timeout=30)
        
        self.connection.row_factory = sqlite3.Row

        with self._lock:
            
            self.connection.execute("PRAGMA journal_mode=WAL")
            
            self.connection.execute("PRAGMA busy_timeout=30000")
            
            self.connection.execute("PRAGMA foreign_keys=ON")
            
            self._migrate()

    @staticmethod
    def _schema_sql(*, include_indexes: bool = True) -> str:
        '生成完整的建表 SQL。\n\n注意： 表中每个状态字段都有 CHECK(status IN (...)) 约束。 这意味着即使代码有 bug 传了非法状态值，SQLite 也会直接拒绝写入， 而不是默默存一个 garbage 值。'
        sql = """
        CREATE TABLE sessions (
            session_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            title TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'active'
                CHECK(status IN ('active','archived')),
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE(session_id, user_id)
        );
        CREATE TABLE tasks (
            task_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            query TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending'
                CHECK(status IN ('pending','running','completed','failed','cancelled','timed_out','failed_requires_review')),
            result TEXT,
            error TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE(task_id, user_id),
            UNIQUE(task_id, session_id, user_id),
            FOREIGN KEY(session_id, user_id) REFERENCES sessions(session_id, user_id) ON DELETE CASCADE
        );
        CREATE TABLE runs (
            run_id TEXT PRIMARY KEY,
            task_id TEXT NOT NULL,
            session_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            status TEXT NOT NULL
                CHECK(status IN ('running','recovering','recovering_claimed','completed','failed','cancelled','timed_out','interrupted')),
            worker_id TEXT NOT NULL,
            lease_expires_at TEXT NOT NULL,
            recovered_from_run_id TEXT,
            recovery_attempt INTEGER NOT NULL DEFAULT 0 CHECK(recovery_attempt >= 0),
            started_at TEXT NOT NULL,
            ended_at TEXT,
            UNIQUE(run_id, user_id),
            FOREIGN KEY(task_id, session_id, user_id)
                REFERENCES tasks(task_id, session_id, user_id) ON DELETE CASCADE,
            FOREIGN KEY(recovered_from_run_id) REFERENCES runs(run_id) ON DELETE SET NULL
        );
        CREATE TABLE messages (
            message_id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            task_id TEXT,
            run_id TEXT,
            agent_id TEXT,
            role TEXT NOT NULL CHECK(role IN ('system','user','assistant','tool')),
            content TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(session_id, user_id) REFERENCES sessions(session_id, user_id) ON DELETE CASCADE,
            FOREIGN KEY(task_id, user_id) REFERENCES tasks(task_id, user_id) ON DELETE CASCADE,
            FOREIGN KEY(run_id, user_id) REFERENCES runs(run_id, user_id) ON DELETE CASCADE
        );
        CREATE TABLE tool_calls (
            tool_call_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            tool_name TEXT NOT NULL,
            side_effect TEXT NOT NULL
                CHECK(side_effect IN ('none','read','idempotent_write','non_idempotent_write')),
            status TEXT NOT NULL
                CHECK(status IN ('started','succeeded','failed','outcome_unknown')),
            arguments_json TEXT,
            result_json TEXT,
            idempotency_key TEXT,
            attempts INTEGER NOT NULL DEFAULT 1 CHECK(attempts >= 1),
            started_at TEXT NOT NULL,
            ended_at TEXT,
            FOREIGN KEY(run_id, user_id) REFERENCES runs(run_id, user_id) ON DELETE CASCADE
        );
        CREATE TABLE events (
            event_id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            task_id TEXT,
            run_id TEXT,
            agent_id TEXT,
            type TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            FOREIGN KEY(session_id, user_id) REFERENCES sessions(session_id, user_id) ON DELETE CASCADE,
            FOREIGN KEY(task_id, user_id) REFERENCES tasks(task_id, user_id) ON DELETE CASCADE,
            FOREIGN KEY(run_id, user_id) REFERENCES runs(run_id, user_id) ON DELETE CASCADE
        );
        CREATE TABLE summaries (
            summary_id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            start_message_id INTEGER,
            end_message_id INTEGER,
            content TEXT NOT NULL,
            token_estimate INTEGER NOT NULL CHECK(token_estimate >= 0),
            created_at TEXT NOT NULL,
            FOREIGN KEY(session_id, user_id) REFERENCES sessions(session_id, user_id) ON DELETE CASCADE
        );
        CREATE TABLE artifacts (
            artifact_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            task_id TEXT,
            run_id TEXT,
            name TEXT NOT NULL,
            relative_path TEXT NOT NULL,
            media_type TEXT,
            size INTEGER CHECK(size IS NULL OR size >= 0),
            sha256 TEXT CHECK(sha256 IS NULL OR length(sha256) = 64),
            idempotency_key TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY(session_id, user_id) REFERENCES sessions(session_id, user_id) ON DELETE CASCADE,
            FOREIGN KEY(task_id, user_id) REFERENCES tasks(task_id, user_id) ON DELETE CASCADE,
            FOREIGN KEY(run_id, user_id) REFERENCES runs(run_id, user_id) ON DELETE CASCADE
        );
        CREATE TABLE memory_state (
            session_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            state_json TEXT NOT NULL,
            version INTEGER NOT NULL DEFAULT 1 CHECK(version >= 1),
            updated_at TEXT NOT NULL,
            PRIMARY KEY(session_id, user_id),
            FOREIGN KEY(session_id, user_id)
                REFERENCES sessions(session_id, user_id) ON DELETE CASCADE
        );
        CREATE TABLE memory_episodes (
            episode_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            start_message_id INTEGER NOT NULL,
            end_message_id INTEGER NOT NULL,
            summary TEXT NOT NULL,
            keywords_json TEXT NOT NULL,
            importance REAL NOT NULL DEFAULT 0.5
                CHECK(importance >= 0 AND importance <= 1),
            token_estimate INTEGER NOT NULL CHECK(token_estimate >= 0),
            created_at TEXT NOT NULL,
            UNIQUE(session_id, user_id, start_message_id, end_message_id),
            CHECK(start_message_id <= end_message_id),
            FOREIGN KEY(session_id, user_id)
                REFERENCES sessions(session_id, user_id) ON DELETE CASCADE
        );
        """
        if include_indexes:
            sql += MemoryStore._indexes_sql()
        return sql

    @staticmethod
    def _indexes_sql() -> str:
        '生成所有索引的 SQL。'
        return """
        CREATE INDEX idx_events_session_id ON events(session_id, user_id, event_id);
        CREATE INDEX idx_messages_session_id ON messages(session_id, user_id, message_id);
        CREATE INDEX idx_sessions_owner ON sessions(user_id, updated_at);
        CREATE INDEX idx_tool_calls_idempotency ON tool_calls(user_id, tool_name, idempotency_key, status);
        CREATE UNIQUE INDEX uq_artifacts_owner_idempotency
            ON artifacts(user_id, idempotency_key) WHERE idempotency_key IS NOT NULL;
        CREATE INDEX idx_memory_episodes_session
            ON memory_episodes(session_id, user_id, end_message_id DESC);
        """

    def _migrate(self) -> None:
        '自动建表或升级数据库结构。'
        
        existing = self.connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='sessions'"
        ).fetchone()
        
        version = self.connection.execute("PRAGMA user_version").fetchone()[0]

        if not existing:
            
            self.connection.executescript(self._schema_sql())
            self.connection.execute(f"PRAGMA user_version={self.SCHEMA_VERSION}")
            self.connection.commit()
            return

        if version < self.SCHEMA_VERSION:
            
            if version == 2:
                self._migrate_v2_to_v3()
                version = 3
            elif version < 2:
                self._migrate_legacy_to_v3()
                version = self.SCHEMA_VERSION
            if version == 3:
                self._migrate_v3_to_v4()

        violations = list(self.connection.execute("PRAGMA foreign_key_check"))
        if violations:
            raise sqlite3.IntegrityError(f"数据库外键校验失败: {violations[:5]}")

    def _migrate_legacy_to_v3(self) -> None:
        '从早期结构重建为当前最新结构；方法名为兼容既有调用保留。'
        
        tables = ("sessions", "tasks", "runs", "messages", "tool_calls", "events", "summaries", "artifacts")
        self.connection.commit()
        
        self.connection.execute("PRAGMA foreign_keys=OFF")
        try:
            
            self.connection.execute("BEGIN IMMEDIATE")
            
            for index in (
                "idx_events_session_id", "idx_messages_session_id", "idx_sessions_owner",
                "idx_tool_calls_idempotency", "uq_artifacts_owner_idempotency",
            ):
                self.connection.execute(f"DROP INDEX IF EXISTS {index}")
            
            for table in tables:
                if self.connection.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
                ).fetchone():
                    self.connection.execute(f"ALTER TABLE {table} RENAME TO legacy_{table}")
            
            self.connection.executescript(self._schema_sql(include_indexes=False))
            
            self.connection.execute("INSERT INTO sessions SELECT * FROM legacy_sessions")
            self.connection.execute("INSERT INTO tasks SELECT * FROM legacy_tasks")
            self.connection.execute(
                "INSERT INTO runs(run_id,task_id,session_id,user_id,status,worker_id,"
                "lease_expires_at,recovered_from_run_id,started_at,ended_at) "
                "SELECT run_id,task_id,session_id,user_id,status,worker_id,"
                "lease_expires_at,recovered_from_run_id,started_at,ended_at FROM legacy_runs"
            )
            self.connection.execute("INSERT INTO messages SELECT * FROM legacy_messages")
            
            self.connection.execute(
                "INSERT INTO tool_calls(tool_call_id,run_id,user_id,tool_name,side_effect,status,arguments_json,result_json,idempotency_key,attempts,started_at,ended_at) "
                "SELECT tc.tool_call_id,tc.run_id,r.user_id,tc.tool_name,tc.side_effect,tc.status,tc.arguments_json,tc.result_json,tc.idempotency_key,tc.attempts,tc.started_at,tc.ended_at "
                "FROM legacy_tool_calls tc JOIN legacy_runs r ON r.run_id=tc.run_id"
            )
            self.connection.execute("INSERT INTO events SELECT * FROM legacy_events")
            self.connection.execute("INSERT INTO summaries SELECT * FROM legacy_summaries")
            
            self.connection.execute("INSERT OR IGNORE INTO artifacts SELECT * FROM legacy_artifacts")
            
            for table in reversed(tables):
                self.connection.execute(f"DROP TABLE legacy_{table}")
            
            self.connection.executescript(self._indexes_sql())
            
            self.connection.execute(f"PRAGMA user_version={self.SCHEMA_VERSION}")
            self.connection.commit()
        except Exception:
            
            self.connection.rollback()
            raise
        finally:
            
            self.connection.execute("PRAGMA foreign_keys=ON")

    def _migrate_v2_to_v3(self) -> None:
        """v3 为 runs 增加恢复次数，用于限制反复崩溃的恢复运行。"""
        with self._lock, self.connection:
            columns = {
                row["name"]
                for row in self.connection.execute("PRAGMA table_info(runs)")
            }
            if "recovery_attempt" not in columns:
                self.connection.execute(
                    "ALTER TABLE runs ADD COLUMN recovery_attempt INTEGER NOT NULL DEFAULT 0 "
                    "CHECK(recovery_attempt >= 0)",
                )
            self.connection.execute("PRAGMA user_version=3")

    def _migrate_v3_to_v4(self) -> None:
        """
        将 v3 原地升级到 v4。

        只新增 memory_state、memory_episodes 及索引，不重写原来的会话、
        消息和滚动摘要，因此已有数据和旧接口保持兼容。
        """
        with self._lock, self.connection:
            self.connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS memory_state (
                    session_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    state_json TEXT NOT NULL,
                    version INTEGER NOT NULL DEFAULT 1 CHECK(version >= 1),
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(session_id, user_id),
                    FOREIGN KEY(session_id, user_id)
                        REFERENCES sessions(session_id, user_id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS memory_episodes (
                    episode_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    start_message_id INTEGER NOT NULL,
                    end_message_id INTEGER NOT NULL,
                    summary TEXT NOT NULL,
                    keywords_json TEXT NOT NULL,
                    importance REAL NOT NULL DEFAULT 0.5
                        CHECK(importance >= 0 AND importance <= 1),
                    token_estimate INTEGER NOT NULL CHECK(token_estimate >= 0),
                    created_at TEXT NOT NULL,
                    UNIQUE(session_id, user_id, start_message_id, end_message_id),
                    CHECK(start_message_id <= end_message_id),
                    FOREIGN KEY(session_id, user_id)
                        REFERENCES sessions(session_id, user_id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_memory_episodes_session
                    ON memory_episodes(session_id, user_id, end_message_id DESC);
                """
            )
            self.connection.execute(f"PRAGMA user_version={self.SCHEMA_VERSION}")

    def close(self) -> None:
        """关闭数据库连接，释放文件句柄。"""
        with self._lock:
            self.connection.close()

    @staticmethod
    def _dict(row: sqlite3.Row | None) -> dict[str, Any]:
        """
        把 sqlite3.Row 转成普通 dict。

        如果查不到结果（row is None），抛出 ResourceNotFound 异常。
        上层调用方不需要每个方法都写 if row is None 的判断。

        _dict() 负责判断查询结果是否为空，并统一处理返回值或抛出异常。
        而且 _dict() 不只给会话使用，也可以给任务、运行、工具调用、文件制品等查询方法使用。
        """
        if row is None:
            raise ResourceNotFound("资源不存在")
        return dict(row)

    @classmethod
    def _limit(cls, limit: int) -> int:
        """
        把传入的 limit 限制在安全范围内。

        limit < 1        → 报错
        limit > 500      → 返回 500（防止一次查太多数据）
        limit 在 1~500   → 原样返回
        """
        if limit < 1:
            raise ValueError("limit 必须大于 0")
        return min(limit, cls.MAX_PAGE_SIZE)

    def create_session(self, session_id: str, user_id: str, title: str) -> dict[str, Any]:
        """
        创建新会话。

        调用时机：前端 POST /api/sessions 时调用。

        参数：
          session_id：唯一标识（如 "session_abc123"）
          user_id：哪个用户创建的
          title：会话标题（如 "药品销售分析"）

        返回：创建后的完整会话信息（dict）
        """
        now = _now()

        with self._lock, self.connection:
            self.connection.execute(
                "INSERT INTO sessions(session_id,user_id,title,created_at,updated_at) VALUES(?,?,?,?,?)",
                (session_id, user_id, title, now, now),
            )
            return self.get_session(session_id, user_id)

    def get_session(self, session_id: str, user_id: str) -> dict[str, Any]:
        """
        查询单个会话。同时校验 session_id 和 user_id（租户隔离）。
        不存在 → 抛出 ResourceNotFound。
        """
        with self._lock:
            return self._dict(self.connection.execute(
                "SELECT * FROM sessions WHERE session_id=? AND user_id=?", (session_id, user_id)
            ).fetchone())

    def list_sessions(
        self, user_id: str, *, before_updated_at: str | None = None, limit: int = 100,
    ) -> list[dict[str, Any]]:
        '列出某个用户的所有会话，按更新时间倒序。'
        with self._lock:
            limit = self._limit(limit)
            if before_updated_at:
                rows = self.connection.execute(
                    "SELECT * FROM sessions WHERE user_id=? AND updated_at<? ORDER BY updated_at DESC LIMIT ?",
                    (user_id, before_updated_at, limit),
                )
            else:
                rows = self.connection.execute(
                    "SELECT * FROM sessions WHERE user_id=? ORDER BY updated_at DESC LIMIT ?", (user_id, limit)
                )
            return [dict(row) for row in rows]

    def delete_session(self, session_id: str, user_id: str) -> None:
        '删除一个会话。'
        with self._lock, self.connection:
            self.get_session(session_id, user_id)
            busy = self.connection.execute(
                "SELECT 1 FROM runs WHERE session_id=? AND user_id=? "
                "AND status IN ('running','recovering') LIMIT 1",
                (session_id, user_id),
            ).fetchone()
            if busy is not None:
                raise SessionBusyError("会话仍有运行中的任务")
            cursor = self.connection.execute(
                "DELETE FROM sessions WHERE session_id=? AND user_id=?", (session_id, user_id)
            )
            if cursor.rowcount != 1:
                raise ResourceNotFound("会话不存在")

    def create_task(self, task_id: str, session_id: str, user_id: str, query: str) -> dict[str, Any]:
        """
        创建新任务。

        调用时机：用户在前端输入问题，POST /api/sessions/{id}/tasks 时调用。

        参数：
          task_id：唯一标识
          session_id：所属会话
          user_id：用户 ID
          query：用户的问题内容（如 "帮我分析今年药品销售情况"）

        创建前会先验证 session 存在，session 不存在 → 抛出 ResourceNotFound。
        """
        now = _now()
        with self._lock, self.connection:
            
            self.get_session(session_id, user_id)
            self.connection.execute(
                "INSERT INTO tasks(task_id,session_id,user_id,query,created_at,updated_at) VALUES(?,?,?,?,?,?)",
                (task_id, session_id, user_id, query, now, now),
            )
            return self.get_task(task_id, user_id)

    def get_task(self, task_id: str, user_id: str) -> dict[str, Any]:
        """查询单个任务。不存在 → 抛出 ResourceNotFound。"""
        with self._lock:
            return self._dict(self.connection.execute(
                "SELECT * FROM tasks WHERE task_id=? AND user_id=?", (task_id, user_id)
            ).fetchone())

    def update_task_status(
        self, task_id: str, user_id: str, status: TaskStatus | str,
        *, result: str | None = None, error: str | None = None,
    ) -> None:
        '更新任务状态。'
        
        value = _enum_value(status, TaskStatus)
        with self._lock, self.connection:
            cursor = self.connection.execute(
                "UPDATE tasks SET status=?,result=COALESCE(?,result),error=COALESCE(?,error),updated_at=? "
                "WHERE task_id=? AND user_id=?",
                (value, result, error, _now(), task_id, user_id),
            )
            if cursor.rowcount != 1:
                raise ResourceNotFound("任务不存在")

    def create_run(
        self, run_id: str, task_id: str, *args: str, lease_seconds: int = 60,
        recovered_from_run_id: str | None = None, recovery_attempt: int = 0,
        status: RunStatus | str = RunStatus.RUNNING,
    ) -> dict[str, Any]:
        '创建一条运行记录，同时把 task 状态改为 running。'
        
        if len(args) == 2:
            user_id, worker_id = args
            supplied_session_id = None
        elif len(args) == 3:
            supplied_session_id, user_id, worker_id = args
        else:
            raise TypeError("create_run 需要 user_id, worker_id")

        value = _enum_value(status, RunStatus)
        with self._lock, self.connection:
            
            task = self.get_task(task_id, user_id)
            session_id = task["session_id"]
            if supplied_session_id is not None and supplied_session_id != session_id:
                raise ValueError("run 的 session_id 必须与 task 一致")

            lease = datetime.now(timezone.utc) + timedelta(seconds=lease_seconds)

            self.connection.execute(
                "INSERT INTO runs(run_id,task_id,session_id,user_id,status,worker_id,"
                "lease_expires_at,recovered_from_run_id,recovery_attempt,started_at) "
                "VALUES(?,?,?,?,?,?,?,?,?,?)",
                (run_id, task_id, session_id, user_id, value, worker_id, lease.isoformat(),
                 recovered_from_run_id, recovery_attempt, _now()),
            )
            
            self.connection.execute(
                "UPDATE tasks SET status='running',updated_at=? WHERE task_id=? AND user_id=?",
                (_now(), task_id, user_id),
            )
            return self.get_run(run_id, user_id)

    def get_run(self, run_id: str, user_id: str) -> dict[str, Any]:
        """查询单条运行记录。不存在 → ResourceNotFound。"""
        with self._lock:
            return self._dict(self.connection.execute(
                "SELECT * FROM runs WHERE run_id=? AND user_id=?", (run_id, user_id)
            ).fetchone())

    def finish_run(self, run_id: str, user_id: str, status: RunStatus | str) -> None:
        """
        结束一次运行（写入 ended_at 和终态）。

        status 必须是 completed / failed / cancelled / timed_out / interrupted 之一。
        """
        value = _enum_value(status, RunStatus)
        terminal_statuses = {
            RunStatus.COMPLETED.value,
            RunStatus.FAILED.value,
            RunStatus.CANCELLED.value,
            RunStatus.TIMED_OUT.value,
            RunStatus.INTERRUPTED.value,
        }
        if value not in terminal_statuses:
            raise ValueError(f"finish_run 只接受终态，收到: {value}")
        with self._lock, self.connection:
            cursor = self.connection.execute(
                "UPDATE runs SET status=?,ended_at=? WHERE run_id=? AND user_id=? "
                "AND status IN ('running','recovering','recovering_claimed')",
                (value, _now(), run_id, user_id),
            )
            if cursor.rowcount != 1:
                raise ResourceNotFound("运行不存在")

    def renew_run_lease(
        self, run_id: str, user_id: str, worker_id: str, *, lease_seconds: int = 60,
    ) -> str:
        '续租：把运行租约延长 lease_seconds 秒。'
        lease = (datetime.now(timezone.utc) + timedelta(seconds=lease_seconds)).isoformat()
        with self._lock, self.connection:
            cursor = self.connection.execute(
                "UPDATE runs SET lease_expires_at=? WHERE run_id=? AND user_id=? AND worker_id=? "
                "AND status IN ('running','recovering')",
                (lease, run_id, user_id, worker_id),
            )
            if cursor.rowcount != 1:
                raise ResourceNotFound("可续租的运行不存在")
            return lease

    def append_message(
        self, session_id: str, user_id: str, *, role: MessageRole | str, content: str,
        task_id: str | None = None, run_id: str | None = None, agent_id: str | None = None,
    ) -> int:
        '向对话历史追加一条消息。'
        value = _enum_value(role, MessageRole)
        with self._lock, self.connection:
            
            self.get_session(session_id, user_id)
            cursor = self.connection.execute(
                "INSERT INTO messages(session_id,user_id,task_id,run_id,agent_id,role,content,created_at) "
                "VALUES(?,?,?,?,?,?,?,?)",
                (session_id, user_id, task_id, run_id, agent_id, value, content, _now()),
            )
            return int(cursor.lastrowid)

    def list_messages(
        self, session_id: str, user_id: str, *, after_message_id: int | None = 0,
        before_message_id: int | None = None, limit: int = 100,
    ) -> list[dict[str, Any]]:
        """
        查询会话消息，按 message_id 正序（旧→新）。

        参数：
          after_message_id：只取 id > 此值，用于增量加载新消息
          before_message_id：只取 id < 此值，用于向前加载更早历史
          limit：每次返回上限

        用途：前端加载对话历史、动态翻页。
        """
        with self._lock:
            self.get_session(session_id, user_id)
            limit = self._limit(limit)
            if before_message_id is not None and after_message_id not in (None, 0):
                raise ValueError("after_message_id 与 before_message_id 不能同时使用")
            if before_message_id is not None:
                rows = list(self.connection.execute(
                    "SELECT * FROM messages WHERE session_id=? AND user_id=? AND message_id<? "
                    "ORDER BY message_id DESC LIMIT ?",
                    (session_id, user_id, before_message_id, limit),
                ))
                rows.reverse()
                return [dict(row) for row in rows]
            cursor = 0 if after_message_id is None else after_message_id
            return [dict(row) for row in self.connection.execute(
                "SELECT * FROM messages WHERE session_id=? AND user_id=? AND message_id>? "
                "ORDER BY message_id LIMIT ?", (session_id, user_id, cursor, limit)
            )]

    def count_messages_after(
        self,
        session_id: str,
        user_id: str,
        after_message_id: int = 0,
    ) -> int:
        """统计累计摘要终点之后仍保存为原文的消息数量。"""
        with self._lock:
            self.get_session(session_id, user_id)
            row = self.connection.execute(
                "SELECT COUNT(*) FROM messages "
                "WHERE session_id=? AND user_id=? AND message_id>?",
                (session_id, user_id, after_message_id),
            ).fetchone()
            return int(row[0])

    def list_recent_messages_after(
        self,
        session_id: str,
        user_id: str,
        *,
        after_message_id: int = 0,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """
        读取摘要终点之后最近的原始消息，并按旧到新返回。

        SQL 先按 message_id 倒序取最近 N 条，再在内存中反转。该方法只负责
        模型上下文的有界读取，不会删除消息，也不会改变累计摘要覆盖终点。
        """
        with self._lock:
            self.get_session(session_id, user_id)
            limit = self._limit(limit)
            rows = list(self.connection.execute(
                "SELECT * FROM messages "
                "WHERE session_id=? AND user_id=? AND message_id>? "
                "ORDER BY message_id DESC LIMIT ?",
                (session_id, user_id, after_message_id, limit),
            ))
            rows.reverse()
            return [dict(row) for row in rows]

    def start_tool_call(
        self, tool_call_id: str, run_id: str, user_id: str, tool_name: str,
        side_effect: StoredSideEffect | str, *, arguments: Any = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        "记录工具调用开始（只写 status='started'）。"
        effect = _enum_value(side_effect, StoredSideEffect)
        with self._lock, self.connection:
            
            self.get_run(run_id, user_id)
            self.connection.execute(
                "INSERT INTO tool_calls(tool_call_id,run_id,user_id,tool_name,side_effect,status,arguments_json,idempotency_key,started_at) "
                "VALUES(?,?,?,?,?,'started',?,?,?)",
                (tool_call_id, run_id, user_id, tool_name, effect,
                 json.dumps(arguments, ensure_ascii=False), idempotency_key, _now()),
            )
            return self.get_tool_call(tool_call_id, user_id)

    def update_tool_call(
        self, tool_call_id: str, user_id: str, *, status: ToolCallStatus | str,
        result: Any = None, attempts: int = 1,
    ) -> dict[str, Any]:
        '更新工具调用为终态（写 status + result_json + ended_at）。'
        value = _enum_value(status, ToolCallStatus)
        
        if value == ToolCallStatus.STARTED:
            raise ValueError("update_tool_call 只能写入终态")
        with self._lock, self.connection:
            cursor = self.connection.execute(
                "UPDATE tool_calls SET status=?,result_json=?,attempts=?,ended_at=? "
                "WHERE tool_call_id=? AND user_id=? AND status='started'",
                (value, json.dumps(result, ensure_ascii=False), attempts, _now(), tool_call_id, user_id),
            )
            if cursor.rowcount != 1:
                raise ResourceNotFound("started 工具调用不存在")
            return self.get_tool_call(tool_call_id, user_id)

    def record_tool_call(
        self, tool_call_id: str, run_id: str, tool_name: str, side_effect: StoredSideEffect | str,
        status: ToolCallStatus | str, *, user_id: str | None = None, arguments: Any = None,
        result: Any = None, idempotency_key: str | None = None,
    ) -> None:
        '旧兼容方法：一次性写入工具调用（先 start 再 update）。'
        with self._lock:
            
            if user_id is None:
                row = self.connection.execute("SELECT user_id FROM runs WHERE run_id=?", (run_id,)).fetchone()
                if row is None:
                    raise ResourceNotFound("运行不存在")
                user_id = row["user_id"]
            value = _enum_value(status, ToolCallStatus)
            
            self.start_tool_call(
                tool_call_id, run_id, user_id, tool_name, side_effect,
                arguments=arguments, idempotency_key=idempotency_key,
            )
            
            if value != ToolCallStatus.STARTED:
                attempts = result.get("attempts", 1) if isinstance(result, dict) else 1
                self.update_tool_call(tool_call_id, user_id, status=value, result=result, attempts=attempts)

    def get_tool_call(self, tool_call_id: str, user_id: str) -> dict[str, Any]:
        """查询单条工具调用记录。"""
        with self._lock:
            return self._dict(self.connection.execute(
                "SELECT * FROM tool_calls WHERE tool_call_id=? AND user_id=?", (tool_call_id, user_id)
            ).fetchone())

    def exists_failed_tool_call(self, run_id: str, user_id: str) -> bool:
        """当前用户的指定 run 是否包含失败或结果未知的工具调用。"""
        with self._lock:
            row = self.connection.execute(
                "SELECT 1 FROM tool_calls WHERE user_id=? AND run_id=? "
                "AND status IN ('failed','outcome_unknown') LIMIT 1",
                (user_id, run_id),
            ).fetchone()
            return row is not None

    def find_succeeded_tool_call(
        self, tool_name: str, idempotency_key: str, user_id: str,
    ) -> dict[str, Any] | None:
        """
        幂等查询：同名工具 + 同幂等键是否已经成功执行过？

        如果有 → 返回之前的成功记录（可跳过执行，直接复用结果）
        如果没有 → 返回 None（正常执行）

        这是"幂等性"的数据库层面实现：同一个操作调用两次不会产生重复效果。
        """
        with self._lock:
            row = self.connection.execute(
                "SELECT * FROM tool_calls WHERE user_id=? AND tool_name=? AND idempotency_key=? "
                "AND status='succeeded' ORDER BY ended_at DESC LIMIT 1",
                (user_id, tool_name, idempotency_key),
            ).fetchone()
            return dict(row) if row is not None else None

    def recover_stale_runs(
        self,
        worker_id: str,
        *,
        max_recovery_attempts: int = 3,
    ) -> list[dict[str, Any]]:
        '扫描所有"租约已过期但仍显示 running"的 run，尝试恢复。'
        recovered: list[dict[str, Any]] = []
        with self._lock:
            
            candidates = [dict(row) for row in self.connection.execute(
                "SELECT r.* FROM runs r JOIN tasks t "
                "ON t.task_id=r.task_id AND t.user_id=r.user_id "
                "WHERE r.status IN ('running','recovering') AND r.lease_expires_at<? "
                "AND t.status NOT IN ('cancelled','completed','failed','timed_out','failed_requires_review') "
                "ORDER BY r.started_at",
                (_now(),),
            )]
            for row in candidates:
                next_attempt = int(row.get("recovery_attempt") or 0) + 1
                if next_attempt > max_recovery_attempts:
                    self.update_task_status(
                        row["task_id"],
                        row["user_id"],
                        TaskStatus.FAILED_REQUIRES_REVIEW,
                        error="超过最大恢复次数",
                    )
                    continue
                with self.connection:

                    claimed = self.connection.execute(
                        "UPDATE runs SET status='recovering_claimed',worker_id=? WHERE run_id=? "
                        "AND status IN ('running','recovering') AND lease_expires_at<?",
                        (worker_id, row["run_id"], _now()),
                    )
                    if claimed.rowcount != 1:
                        continue  

                    unknown = self.connection.execute(
                        "SELECT tool_call_id FROM tool_calls WHERE run_id=? AND user_id=? "
                        "AND status='started' AND side_effect='non_idempotent_write'",
                        (row["run_id"], row["user_id"]),
                    ).fetchall()

                    self.connection.execute(
                        "UPDATE runs SET status='interrupted',ended_at=? WHERE run_id=? AND user_id=?",
                        (_now(), row["run_id"], row["user_id"]),
                    )
                    
                    for call in unknown:
                        self.connection.execute(
                            "UPDATE tool_calls SET status='outcome_unknown',ended_at=? "
                            "WHERE tool_call_id=? AND user_id=? AND status='started'",
                            (_now(), call["tool_call_id"], row["user_id"]),
                        )

                if unknown:
                    
                    self.update_task_status(
                        row["task_id"], row["user_id"], TaskStatus.FAILED_REQUIRES_REVIEW,
                        error="非幂等工具结果未知",
                    )
                    continue
                
                new_run = self.create_run(
                    new_id("run"), row["task_id"], row["user_id"], worker_id,
                    recovered_from_run_id=row["run_id"], status=RunStatus.RECOVERING,
                    recovery_attempt=next_attempt,
                )
                recovered.append(new_run)
        return recovered

    def append_event(self, event: RuntimeEvent, user_id: str) -> RuntimeEvent:
        """
        持久化一条事件，并返回带 event_id 的 event 对象。

        传入的 event.event_id 是 None（内存中生成），
        存入 SQLite 后由 AUTOINCREMENT 分配真正的 event_id，
        返回的对象中 event_id 已填充。

        这是 EventBus.publish() 的第一步：先落盘，再广播。
        """
        with self._lock, self.connection:
            
            self.get_session(event.session_id, user_id)
            cursor = self.connection.execute(
                "INSERT INTO events(session_id,user_id,task_id,run_id,agent_id,type,payload_json,timestamp) "
                "VALUES(?,?,?,?,?,?,?,?)",
                (event.session_id, user_id, event.task_id, event.run_id, event.agent_id,
                 event.type, json.dumps(event.payload, ensure_ascii=False), event.timestamp.isoformat()),
            )
            
            return event.model_copy(update={"event_id": int(cursor.lastrowid)})

    def list_events(
        self, session_id: str, user_id: str, *, after_event_id: int = 0, limit: int = 200,
    ) -> list[dict[str, Any]]:
        """
        查询会话事件，按 event_id 正序。

        参数：
          after_event_id：游标分页，只取 id > 此值的（断线重连时补推）
          limit：每次返回上限

        用途：
          1. 初次加载时拉取历史事件
          2. 前端断线重连时传 after_event_id，只补推缺失的
        """
        with self._lock:
            self.get_session(session_id, user_id)
            limit = self._limit(limit)
            rows = self.connection.execute(
                "SELECT * FROM events WHERE session_id=? AND user_id=? AND event_id>? "
                "ORDER BY event_id LIMIT ?", (session_id, user_id, after_event_id, limit)
            )
            result = []
            for row in rows:
                item = dict(row)
                
                item["payload"] = json.loads(item.pop("payload_json"))
                result.append(item)
            return result

    def append_summary(
        self, session_id: str, user_id: str, *, content: str, token_estimate: int,
        start_message_id: int | None = None, end_message_id: int | None = None,
    ) -> int:
        '记录一条对话摘要。'

        if (start_message_id is None) != (end_message_id is None):
            raise ValueError("摘要起止消息 ID 必须同时提供")
        if (
            start_message_id is not None
            and end_message_id is not None
            and start_message_id > end_message_id
        ):
            raise ValueError("摘要起始消息 ID 不能大于结束消息 ID")

        with self._lock, self.connection:
            self.get_session(session_id, user_id)
            if start_message_id is not None and end_message_id is not None:
                endpoints = self.connection.execute(
                    "SELECT COUNT(*) FROM messages WHERE session_id=? AND user_id=? "
                    "AND message_id IN (?,?)",
                    (session_id, user_id, start_message_id, end_message_id),
                ).fetchone()[0]
                expected_endpoints = 1 if start_message_id == end_message_id else 2
                if endpoints != expected_endpoints:
                    raise ValueError("摘要区间必须引用当前会话中真实存在的消息")
                overlap = self.connection.execute(
                    "SELECT 1 FROM summaries WHERE session_id=? AND user_id=? "
                    "AND start_message_id IS NOT NULL AND end_message_id IS NOT NULL "
                    "AND NOT(end_message_id<? OR start_message_id>?) LIMIT 1",
                    (session_id, user_id, start_message_id, end_message_id),
                ).fetchone()
                if overlap is not None:
                    raise ValueError("摘要覆盖区间不能与已有摘要重叠")
            cursor = self.connection.execute(
                "INSERT INTO summaries(session_id,user_id,start_message_id,end_message_id,content,token_estimate,created_at) "
                "VALUES(?,?,?,?,?,?,?)",
                (session_id, user_id, start_message_id, end_message_id, content, token_estimate, _now()),
            )
            return int(cursor.lastrowid)

    def list_summaries(
        self, session_id: str, user_id: str, *, after_summary_id: int = 0, limit: int = 50,
    ) -> list[dict[str, Any]]:
        '查询会话摘要列表。 `list_summaries()` 用于分页读取某个用户指定会话中的摘要记录。函数先加锁保证查询过程 的线程安全，再调用 `get_session()` 验证该会话存在并且属于当前用户，然后通过 `_limit()` 校验并限 制本次最多返回多少条摘要。参数 `after_summary_id` 表示“从哪一条摘要之后开始查询”，例如 传入 `after_summary_id=0` 时会从最早的摘要开始读取；如果上一批最后一条摘要的 `summary_i'
        with self._lock:
            self.get_session(session_id, user_id)
            limit = self._limit(limit)
            return [dict(row) for row in self.connection.execute(
                "SELECT * FROM summaries WHERE session_id=? AND user_id=? AND summary_id>? "
                "ORDER BY summary_id LIMIT ?", (session_id, user_id, after_summary_id, limit)
            )]

    def get_latest_summary(self, session_id: str, user_id: str) -> dict[str, Any] | None:
        '返回当前会话最新的摘要；尚未生成摘要时返回 None。 `get_latest_summary()` 用于读取某个用户指定会话中最新生成的一条摘要。函数先通过线程锁保证查 询过程安全，再调用 `get_session()` 验证该会话真实存在并且属于当前用户；随后 从 SQLite 的 `summaries` 表中筛选当前 `session_id` 和 `user_id` 对应的摘要，按照 `summary_id` 从大到 小排序，并通过 `LIMIT 1` 只取 ID 最大'
        with self._lock:
            self.get_session(session_id, user_id)
            row = self.connection.execute(
                "SELECT * FROM summaries WHERE session_id=? AND user_id=? "
                "ORDER BY summary_id DESC LIMIT 1",
                (session_id, user_id),
            ).fetchone()
            return dict(row) if row is not None else None

    def update_rolling_summary(
        self,
        summary_id: int,
        session_id: str,
        user_id: str,
        *,
        content: str,
        token_estimate: int,
        end_message_id: int,
    ) -> dict[str, Any]:
        '用累计摘要替换当前摘要内容，并把覆盖终点向后推进。'
        with self._lock, self.connection:
            current = self.connection.execute(
                "SELECT * FROM summaries WHERE summary_id=? AND session_id=? AND user_id=?",
                (summary_id, session_id, user_id),
            ).fetchone()
            if current is None:
                raise ResourceNotFound("摘要不存在")
            endpoint = self.connection.execute(
                "SELECT 1 FROM messages WHERE message_id=? AND session_id=? AND user_id=?",
                (end_message_id, session_id, user_id),
            ).fetchone()
            if endpoint is None:
                raise ValueError("摘要结束消息必须属于当前会话")
            if current["end_message_id"] is not None and end_message_id < current["end_message_id"]:
                raise ValueError("滚动摘要覆盖终点不能向前移动")
            self.connection.execute(
                "UPDATE summaries SET content=?,token_estimate=?,end_message_id=?,created_at=? "
                "WHERE summary_id=? AND session_id=? AND user_id=?",
                (
                    content,
                    token_estimate,
                    end_message_id,
                    _now(),
                    summary_id,
                    session_id,
                    user_id,
                ),
            )
            row = self.connection.execute(
                "SELECT * FROM summaries WHERE summary_id=? AND session_id=? AND user_id=?",
                (summary_id, session_id, user_id),
            ).fetchone()
            return dict(row)

    _MEMORY_STATE_FIELDS = (
        "goals", "constraints", "facts", "decisions", "tool_results",
        "errors", "open_items", "artifacts",
    )

    @classmethod
    def _normalize_memory_state(cls, state: dict[str, Any]) -> dict[str, list[str]]:
        '把任意状态字典整理为固定的八类字符串列表。'
        normalized: dict[str, list[str]] = {}
        for field in cls._MEMORY_STATE_FIELDS:
            raw = state.get(field, [])
            values = raw if isinstance(raw, list) else [raw]
            unique: list[str] = []
            for value in values:
                text = str(value).strip()
                if text and text not in unique:
                    unique.append(text[:cls.MAX_MEMORY_STATE_ITEM_LENGTH])
            normalized[field] = unique[-cls.MAX_MEMORY_STATE_ITEMS:]
        return normalized

    def get_memory_state(self, session_id: str, user_id: str) -> dict[str, Any] | None:
        '读取会话唯一的结构化长期状态；尚未生成时返回 None。'
        with self._lock:
            self.get_session(session_id, user_id)
            row = self.connection.execute(
                "SELECT * FROM memory_state WHERE session_id=? AND user_id=?",
                (session_id, user_id),
            ).fetchone()
            if row is None:
                return None
            item = dict(row)
            try:
                decoded = json.loads(item.pop("state_json"))
            except (TypeError, ValueError):
                decoded = {}
            item["state"] = self._normalize_memory_state(
                decoded if isinstance(decoded, dict) else {}
            )
            return item

    def upsert_memory_state(
        self, session_id: str, user_id: str, state: dict[str, Any],
    ) -> dict[str, Any]:
        '原子新增或滚动更新会话唯一的结构化状态。'
        normalized = self._normalize_memory_state(state)
        with self._lock, self.connection:
            self.get_session(session_id, user_id)
            self.connection.execute(
                "INSERT INTO memory_state(session_id,user_id,state_json,version,updated_at) "
                "VALUES(?,?,?,1,?) ON CONFLICT(session_id,user_id) DO UPDATE SET "
                "state_json=excluded.state_json,version=memory_state.version+1,"
                "updated_at=excluded.updated_at",
                (session_id, user_id, json.dumps(normalized, ensure_ascii=False), _now()),
            )
            result = self.get_memory_state(session_id, user_id)
            if result is None:  # pragma: no cover
                raise RuntimeError("结构化记忆写入失败")
            return result

    @staticmethod
    def _decode_memory_episode(row: sqlite3.Row | None) -> dict[str, Any]:
        '把情节行转为字典，并将 keywords_json 解析为列表。'
        if row is None:
            raise ResourceNotFound("情节记忆不存在")
        item = dict(row)
        try:
            keywords = json.loads(item.pop("keywords_json"))
        except (TypeError, ValueError):
            keywords = []
        item["keywords"] = keywords if isinstance(keywords, list) else []
        return item

    def record_memory_episode(
        self,
        episode_id: str,
        session_id: str,
        user_id: str,
        *,
        start_message_id: int,
        end_message_id: int,
        summary: str,
        keywords: list[str] | None = None,
        importance: float = 0.5,
        token_estimate: int = 0,
    ) -> dict[str, Any]:
        '幂等记录一次压缩区间形成的不可变历史情节。'
        if start_message_id > end_message_id:
            raise ValueError("情节起始消息 ID 不能大于结束消息 ID")
        if not 0 <= importance <= 1:
            raise ValueError("importance 必须位于 0 到 1 之间")
        if token_estimate < 0:
            raise ValueError("token_estimate 不能为负数")
        normalized_keywords = list(dict.fromkeys(
            str(keyword).strip()[:100]
            for keyword in (keywords or [])
            if str(keyword).strip()
        ))[:30]
        with self._lock, self.connection:
            self.get_session(session_id, user_id)
            endpoints = self.connection.execute(
                "SELECT COUNT(*) FROM messages WHERE session_id=? AND user_id=? "
                "AND message_id IN (?,?)",
                (session_id, user_id, start_message_id, end_message_id),
            ).fetchone()[0]
            expected = 1 if start_message_id == end_message_id else 2
            if endpoints != expected:
                raise ValueError("情节区间必须引用当前会话中真实存在的消息")
            existing = self.connection.execute(
                "SELECT * FROM memory_episodes WHERE session_id=? AND user_id=? "
                "AND start_message_id=? AND end_message_id=?",
                (session_id, user_id, start_message_id, end_message_id),
            ).fetchone()
            if existing is None:
                self.connection.execute(
                    "INSERT INTO memory_episodes("
                    "episode_id,session_id,user_id,start_message_id,end_message_id,"
                    "summary,keywords_json,importance,token_estimate,created_at"
                    ") VALUES(?,?,?,?,?,?,?,?,?,?)",
                    (
                        episode_id, session_id, user_id, start_message_id, end_message_id,
                        summary, json.dumps(normalized_keywords, ensure_ascii=False),
                        importance, token_estimate, _now(),
                    ),
                )
                existing = self.connection.execute(
                    "SELECT * FROM memory_episodes WHERE episode_id=? AND user_id=?",
                    (episode_id, user_id),
                ).fetchone()
            return self._decode_memory_episode(existing)

    def list_memory_episodes(
        self,
        session_id: str,
        user_id: str,
        *,
        before_end_message_id: int | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        '按覆盖终点倒序分页读取情节；游标用于加载更早的一页。 list_memory_episodes() 用于分页读取某个用户在指定会话中的历史 Episode：它先确认会话存在且属于该用户，再限制每页最多读 取多少条，然后从 memory_episodes 表中按 end_message_id 从大到小查询，也就是优先返回最近形成的记 忆；如果传入 before_end_message_id，就只读取结束消息 ID 小于这个值的更早记录，方便继续翻下一页；最后把 每条数据库记录通'
        with self._lock:
            self.get_session(session_id, user_id)
            limit = self._limit(limit)
            sql = "SELECT * FROM memory_episodes WHERE session_id=? AND user_id=?"
            params: list[Any] = [session_id, user_id]
            if before_end_message_id is not None:
                sql += " AND end_message_id<?"
                params.append(before_end_message_id)
            sql += " ORDER BY end_message_id DESC LIMIT ?"
            params.append(limit)
            return [
                self._decode_memory_episode(row)
                for row in self.connection.execute(sql, params)
            ]

    def search_memory_episodes(
        self,
        session_id: str,
        user_id: str,
        query: str,
        *,
        limit: int = 3,
        candidate_limit: int = 500,
    ) -> list[dict[str, Any]]:
        '用关键词、字符二元组重合度和 importance 召回 Top-K 历史情节。'

        if candidate_limit <= 0:
            raise ValueError("candidate_limit must be positive")
        candidates = self.list_memory_episodes(
            session_id,
            user_id,
            limit=min(candidate_limit, 500),
        )
        query_text = query.casefold()
        query_bigrams = {
            query_text[index:index + 2]
            for index in range(max(0, len(query_text) - 1))
            if not query_text[index:index + 2].isspace()
        }

        def score(item: dict[str, Any]) -> tuple[float, int]:
            summary = str(item["summary"]).casefold()
            keyword_score = sum(
                2.0
                for keyword in item["keywords"]
                if str(keyword).casefold() in query_text
                or (str(keyword).casefold() in summary and query_text in summary)
            )
            summary_bigrams = {
                summary[index:index + 2]
                for index in range(max(0, len(summary) - 1))
            }
            overlap = len(query_bigrams & summary_bigrams) / max(1, len(query_bigrams))
            return (
                keyword_score + overlap + float(item["importance"]) * 0.25,
                int(item["end_message_id"]),
            )

        return sorted(candidates, key=score, reverse=True)[: self._limit(limit)]

    def record_artifact(
        self, artifact_id: str, session_id: str, user_id: str, *, name: str,
        relative_path: str, task_id: str | None = None, run_id: str | None = None,
        media_type: str | None = None, size: int | None = None, sha256: str | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        '记录一个文件制品（存的是元信息，不存文件内容）。'
        
        if sha256 is not None and (len(sha256) != 64 or any(c not in "0123456789abcdefABCDEF" for c in sha256)):
            raise ValueError("sha256 必须是 64 位十六进制")

        with self._lock, self.connection:
            self.get_session(session_id, user_id)

            if idempotency_key:
                existing = self.connection.execute(
                    "SELECT artifact_id FROM artifacts WHERE user_id=? AND idempotency_key=?",
                    (user_id, idempotency_key),
                ).fetchone()
                if existing:
                    return self.get_artifact(existing["artifact_id"], user_id)

            try:
                self.connection.execute(
                    "INSERT INTO artifacts(artifact_id,session_id,user_id,task_id,run_id,name,relative_path,media_type,size,sha256,idempotency_key,created_at) "
                    "VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                    (artifact_id, session_id, user_id, task_id, run_id, name, relative_path,
                     media_type, size, sha256.lower() if sha256 else None, idempotency_key, _now()),
                )
            except sqlite3.IntegrityError:
                
                if not idempotency_key:
                    raise
                existing = self.connection.execute(
                    "SELECT artifact_id FROM artifacts WHERE user_id=? AND idempotency_key=?",
                    (user_id, idempotency_key),
                ).fetchone()
                if existing is None:
                    raise
                return self.get_artifact(existing["artifact_id"], user_id)

            return self.get_artifact(artifact_id, user_id)

    def get_artifact(self, artifact_id: str, user_id: str) -> dict[str, Any]:
        """查询单个制品的元信息。"""
        with self._lock:
            return self._dict(self.connection.execute(
                "SELECT artifact_id,session_id,task_id,run_id,name,relative_path,media_type,size,sha256,idempotency_key,created_at "
                "FROM artifacts WHERE artifact_id=? AND user_id=?", (artifact_id, user_id)
            ).fetchone())

    def list_artifacts(
        self, session_id: str, user_id: str, *, before_created_at: str | None = None, limit: int = 50,
    ) -> list[dict[str, Any]]:
        """
        列出某个会话的所有制品，按创建时间倒序（新的在最上面）。

        用途：前端右侧面板展示生成的文件列表，也可下拉翻页。
        """
        with self._lock:
            self.get_session(session_id, user_id)
            limit = self._limit(limit)
            columns = "artifact_id,session_id,task_id,run_id,name,relative_path,media_type,size,sha256,idempotency_key,created_at"
            if before_created_at:
                rows = self.connection.execute(
                    f"SELECT {columns} FROM artifacts WHERE session_id=? AND user_id=? AND created_at<? "
                    "ORDER BY created_at DESC LIMIT ?", (session_id, user_id, before_created_at, limit)
                )
            else:
                rows = self.connection.execute(
                    f"SELECT {columns} FROM artifacts WHERE session_id=? AND user_id=? "
                    "ORDER BY created_at DESC LIMIT ?", (session_id, user_id, limit)
                )
            return [dict(row) for row in rows]

    def verify_artifact_file(self, artifact_id: str, user_id: str, file_path: Path) -> None:
        '校验制品文件的完整性。'
        artifact = self.get_artifact(artifact_id, user_id)
        expected = artifact.get("sha256")
        
        if not expected:
            return
        
        actual = hashlib.sha256(file_path.read_bytes()).hexdigest()
        if actual != expected:
            raise ArtifactIntegrityError(f"制品 {artifact_id} 完整性校验失败")
