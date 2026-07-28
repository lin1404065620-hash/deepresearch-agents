from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

from .models import ArtifactRef

class SecurityViolation(ValueError):
    '安全违规异常。继承自 ValueError。'
    pass

class SQLGuard:
    'SQL 安全检查器。'

    _allowed_starts = {"SELECT", "WITH", "SHOW", "DESCRIBE", "DESC", "EXPLAIN"}

    _forbidden_words = {
        "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "TRUNCATE",
        "REPLACE", "MERGE", "CALL", "GRANT", "REVOKE", "COMMIT", "ROLLBACK",
        "START", "LOCK", "UNLOCK", "LOAD_FILE", "SLEEP", "BENCHMARK", "OUTFILE",
        "DUMPFILE",
    }

    def __init__(self, allowed_tables: Iterable[str]):
        """
        初始化 SQL 守卫。

        参数：
          allowed_tables：表名白名单。只有这些表可以被查询。

        示例：
          SQLGuard(["drugs", "sales_records", "suppliers"])
        注意：表名会被 .casefold() 转小写，实现大小写不敏感匹配。
        """
        self.allowed_tables = {name.casefold() for name in allowed_tables}

    def validate(self, query: str) -> str:
        '校验一条 SQL 语句是否安全。\n\n返回： 如果安全，返回原 SQL 字符串（可直接传给 MySQL 执行）'
        
        sql = query.strip()

        if not sql:
            raise SecurityViolation("SQL 不能为空")

        if ";" in sql or "--" in sql or "#" in sql or "/*" in sql or "*/" in sql:
            raise SecurityViolation("禁止多语句或 SQL 注释")

        tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_$]*", sql)
        if not tokens or tokens[0].upper() not in self._allowed_starts:
            raise SecurityViolation("只允许只读 SQL")

        upper_tokens = {token.upper() for token in tokens}
        forbidden = upper_tokens & self._forbidden_words
        if forbidden:
            raise SecurityViolation(f"SQL 包含禁止操作: {', '.join(sorted(forbidden))}")

        cte_aliases = {
            match.group(1).casefold()
            for match in re.finditer(r"(?:\bWITH\b|,)\s*([A-Za-z_][\w$]*)\s+AS\s*\(", sql, re.I)
        }

        referenced = {
            match.group(1).split(".")[-1].strip("`").casefold()
            for match in re.finditer(
                r"\b(?:FROM|JOIN|DESCRIBE|DESC)\s+(`?[A-Za-z_][\w$]*`?(?:\.`?[A-Za-z_][\w$]*`?)?)",
                sql,
                re.I,
            )
        }

        external_tables = referenced - cte_aliases

        unknown = external_tables - self.allowed_tables
        if unknown:
            raise SecurityViolation(f"SQL 访问了未授权数据表: {', '.join(sorted(unknown))}")

        return sql

class PathGuard:
    '文件路径安全守卫。'

    capabilities = {"inputs", "artifacts", "temp"}

    def __init__(
        self,
        workspace: Path,
        *,
        allowed_upload_extensions: set[str] | None = None,
    ):
        '初始化路径守卫。'
        
        self.workspace = workspace.resolve()

        self.inputs = self.workspace / "inputs"       
        self.artifacts = self.workspace / "artifacts" 
        self.temp = self.workspace / "temp"           

        for directory in (self.inputs, self.artifacts, self.temp):
            directory.mkdir(parents=True, exist_ok=True)

        default_extensions = {".txt", ".md", ".pdf", ".docx", ".xlsx", ".csv", ".json"}
        self.allowed_upload_extensions = {
            suffix.casefold() for suffix in (allowed_upload_extensions or default_extensions)
        }

    def _root_for(self, capability: str) -> Path:
        '根据文件能力获取对应的根目录。'
        if capability not in self.capabilities:
            raise SecurityViolation(f"未知文件能力: {capability}")
        
        return getattr(self, capability)

    def resolve(self, relative_path: str, *, capability: str, for_write: bool = False) -> Path:
        '将用户/AI 传入的相对路径解析为安全的绝对路径。\n\n返回： 解析后的安全绝对路径'
        
        del for_write

        if not relative_path or "\x00" in relative_path:
            raise SecurityViolation("文件路径无效")

        candidate = Path(relative_path)

        if candidate.is_absolute() or candidate.drive:
            raise SecurityViolation("禁止使用服务器绝对路径")

        root = self._root_for(capability).resolve()

        resolved = (root / candidate).resolve(strict=False)

        if resolved != root and not resolved.is_relative_to(root):
            raise SecurityViolation("文件路径超出授权目录")

        return resolved

    def safe_upload_name(self, original_name: str) -> str:
        '校验上传文件的文件名是否安全，并返回安全的文件名。'
        
        if not original_name or "\x00" in original_name or original_name in {".", ".."}:
            raise SecurityViolation("上传文件名无效")

        normalized = original_name.replace("\\", "/")
        safe_name = Path(normalized).name

        if safe_name != normalized or "/" in normalized:
            raise SecurityViolation("上传文件名不得包含目录")

        if Path(safe_name).suffix.casefold() not in self.allowed_upload_extensions:
            raise SecurityViolation("不允许上传该文件类型")

        return safe_name

    def artifact_ref(self, artifact_id: str, file_path: Path, media_type: str | None = None) -> ArtifactRef:
        '为已生成的文件构建 ArtifactRef（文件索引卡片）。'
        
        resolved = file_path.resolve(strict=False)
        if resolved != self.artifacts and not resolved.is_relative_to(self.artifacts):
            raise SecurityViolation("文件不是当前会话的 artifact")

        return ArtifactRef(
            artifact_id=artifact_id,
            name=resolved.name,                                          
            relative_path=resolved.relative_to(self.workspace).as_posix(), 
            media_type=media_type,
            size=resolved.stat().st_size if resolved.exists() else None,   
        )
