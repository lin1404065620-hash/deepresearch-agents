from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader

from .security import PathGuard, SecurityViolation

SUPPORTED_INPUT_SUFFIXES = {".md", ".txt", ".pdf"}
MAX_EXTRACTED_CHARACTERS = 200_000

def list_input_files(workspace: Path) -> list[dict[str, int | str]]:
    """列出当前会话可供 Agent 读取的输入文件，不向模型或前端暴露服务器路径。"""
    guard = PathGuard(workspace)
    return [
        {"name": path.name, "size": path.stat().st_size}
        for path in sorted(guard.inputs.iterdir(), key=lambda item: item.name.casefold())
        if path.is_file() and path.suffix.casefold() in SUPPORTED_INPUT_SUFFIXES
    ]

def read_input_file(workspace: Path, filename: str) -> str:
    """从当前会话 inputs 目录安全提取文本；不接受目录、绝对路径或路径穿越。"""
    if not filename or Path(filename).name != filename:
        raise SecurityViolation("输入文件名无效")
    guard = PathGuard(workspace)
    target = guard.resolve(filename, capability="inputs")
    if not target.is_file():
        raise FileNotFoundError(f"输入文件不存在: {filename}")

    suffix = target.suffix.casefold()
    if suffix not in SUPPORTED_INPUT_SUFFIXES:
        raise ValueError(f"不支持的输入文件格式: {suffix or '无扩展名'}")
    if suffix == ".pdf":
        text = "\n".join(page.extract_text() or "" for page in PdfReader(str(target)).pages)
    else:
        text = target.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"输入文件没有可提取的文本: {filename}")
    return text[:MAX_EXTRACTED_CHARACTERS]
