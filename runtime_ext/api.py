from __future__ import annotations

import asyncio
import shutil
from collections.abc import Awaitable, Callable
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect, status
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .events import EventBus
from .models import ExecutionContext, new_id
from .security import PathGuard, SecurityViolation
from .store import (
    ArtifactIntegrityError,
    MemoryStore,
    ResourceNotFound,
    SessionBusyError,
)

MAIN_AGENT_PERMISSIONS = frozenset({
    "delegate:network",     
    "delegate:database",    
    "delegate:rag",         
    "delegate:report",      
    "tool:generate_report", 
})

class SessionCreate(BaseModel):
    """
    创建会话的请求体。

    FastAPI 会自动根据这个模型校验请求 JSON：
      - title 必须是 1~200 字符的字符串
      - 不传 title 时默认 "新会话"

    字段：
      title: 会话标题，显示在前端会话列表中
    """
    title: str = Field(default="新会话", min_length=1, max_length=200)

class TaskCreate(BaseModel):
    """
    创建任务的请求体。

    字段：
      query: 用户输入的问题/任务描述
             最少 1 个字符，最多 10 万字符（约一部长篇小说的长度）
    """
    query: str = Field(min_length=1, max_length=100_000)

def create_runtime_router(
    *,
    store: MemoryStore,
    event_bus: EventBus,
    current_user: Callable[[], str],
    workspace_root: Path,
    runner: Callable[[ExecutionContext, str], Awaitable[str]] | None = None,
) -> APIRouter:
    '创建配置好的 FastAPI 路由（工厂函数）。'

    router = APIRouter(prefix="/api")

    def owner(user_id: str = Depends(current_user)) -> str:
        'FastAPI 依赖注入：获取当前认证用户。'
        if not user_id:
            raise HTTPException(status_code=401, detail="未认证")
        return user_id

    def owned_session(session_id: str, user_id: str) -> dict:
        """
        获取会话并验证所有权（辅助函数，减少重复代码）。

        参数：
          session_id: 会话 ID
          user_id: 当前用户 ID

        返回：
          dict — 会话记录

        抛出：
          HTTPException(404) — 会话不存在（不论是不存在还是不属于当前用户，统一返回 404）
        """
        try:
            return store.get_session(session_id, user_id)
        except ResourceNotFound as exc:
            raise HTTPException(status_code=404, detail="会话不存在") from exc

    def delete_owned_session_files_and_data(session_id: str, user_id: str) -> None:
        """用可恢复的 Workspace 回收流程删除一条已确认归属的会话。"""
        root = workspace_root.resolve()
        workspace = (root / session_id).resolve()
        if workspace.parent != root:
            raise SecurityViolation("会话路径无效")

        quarantined: Path | None = None
        if workspace.exists():
            trash_root = root / ".trash"
            trash_root.mkdir(parents=True, exist_ok=True)
            quarantined = trash_root / f"{session_id}-{new_id('trash')}"
            workspace.replace(quarantined)

        try:
            store.delete_session(session_id, user_id)
        except Exception:
            if quarantined is not None and quarantined.exists():
                quarantined.replace(workspace)
            raise

        if quarantined is not None and quarantined.exists():
            shutil.rmtree(quarantined)

    @router.post("/sessions", status_code=status.HTTP_201_CREATED)
    async def create_session(request: SessionCreate, user_id: str = Depends(owner)):
        'POST /api/sessions'
        session_id = new_id("session")
        
        PathGuard(workspace_root / session_id)
        return store.create_session(session_id, user_id, request.title)

    @router.get("/sessions")
    async def list_sessions(
        before: str | None = None,
        limit: int = 100,
        user_id: str = Depends(owner),
    ):
        'GET /api/sessions?before=2024-01-01T00:00:00&limit=100'
        return {
            "sessions": store.list_sessions(
                user_id, before_updated_at=before, limit=limit,
            ),
        }

    @router.delete("/sessions")
    async def delete_all_sessions(user_id: str = Depends(owner)) -> dict:
        """
        删除当前用户全部空闲会话；运行中会话保留，单条失败不影响后续会话。
        """
        sessions: list[dict] = []
        before: str | None = None
        while True:
            page = store.list_sessions(
                user_id,
                before_updated_at=before,
                limit=500,
            )
            if not page:
                break
            sessions.extend(page)
            if len(page) < 500:
                break
            before = page[-1]["updated_at"]

        deleted: list[str] = []
        retained: list[str] = []
        failed: list[str] = []
        for session in sessions:
            session_id = session["session_id"]
            try:
                delete_owned_session_files_and_data(session_id, user_id)
            except SessionBusyError:
                retained.append(session_id)
            except Exception:
                failed.append(session_id)
            else:
                deleted.append(session_id)

        return {
            "deleted_count": len(deleted),
            "retained_count": len(retained),
            "deleted_session_ids": deleted,
            "retained_session_ids": retained,
            "failed_session_ids": failed,
        }

    @router.get("/sessions/{session_id}")
    async def get_session(
        session_id: str,
        after_message_id: int | None = None,
        before_message_id: int | None = None,
        message_limit: int = 100,
        user_id: str = Depends(owner),
    ):
        'GET /api/sessions/{session_id}?after_message_id=10&message_limit=50'
        session = owned_session(session_id, user_id)
        return {
            "session": session,
            "messages": store.list_messages(
                session_id, user_id,
                after_message_id=after_message_id,
                before_message_id=before_message_id,
                limit=message_limit,
            ),
        }

    @router.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
    async def delete_session(
        session_id: str,
        user_id: str = Depends(owner),
    ) -> None:
        """
        删除当前用户拥有的空闲会话及其受控 Workspace。

        Workspace 会先移动到根目录下的临时回收区；数据库删除失败时恢复目录，
        成功时再清理回收副本，避免前端接触或传入服务器绝对路径。
        """
        owned_session(session_id, user_id)
        try:
            delete_owned_session_files_and_data(session_id, user_id)
        except SessionBusyError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="会话仍在运行，请等待任务结束后再删除",
            ) from exc

    @router.post("/sessions/{session_id}/tasks", status_code=status.HTTP_202_ACCEPTED)
    async def create_task(
        session_id: str,
        request: TaskCreate,
        user_id: str = Depends(owner),
    ):
        'POST /api/sessions/{session_id}/tasks'
        
        owned_session(session_id, user_id)

        task_id = new_id("task")
        run_id = new_id("run")

        store.create_task(task_id, session_id, user_id, request.query)
        store.create_run(run_id, task_id, user_id, "single-worker")

        context = ExecutionContext(
            user_id=user_id,
            session_id=session_id,
            task_id=task_id,
            run_id=run_id,
            agent_id="main",
            workspace=workspace_root / session_id,
            permissions=MAIN_AGENT_PERMISSIONS,
        )

        if runner is not None:
            async def run_safely() -> None:
                """
                安全地运行 Agent，确保异常不会导致未读取的异常泄漏。

                Runtime 内部已经负责将失败状态和事件持久化，
                这里只需要吞掉异常，不让 asyncio 报 "Task exception was never retrieved"。
                """
                try:
                    await runner(context, request.query)
                except asyncio.CancelledError:
                    raise  
                except Exception:
                    
                    return

            asyncio.create_task(run_safely(), name=f"runtime-{run_id}")

        return {
            "status": "started",
            "session_id": session_id,
            "task_id": task_id,
            "run_id": run_id,
        }

    @router.get("/sessions/{session_id}/events")
    async def list_events(
        session_id: str,
        after: int = 0,
        limit: int = 200,
        user_id: str = Depends(owner),
    ):
        'GET /api/sessions/{session_id}/events?after=42&limit=200'
        owned_session(session_id, user_id)
        return {
            "events": store.list_events(
                session_id, user_id, after_event_id=after, limit=limit,
            ),
        }

    @router.post("/sessions/{session_id}/upload")
    async def upload_inputs(
        session_id: str,
        files: list[UploadFile] = File(...),
        user_id: str = Depends(owner),
    ):
        'POST /api/sessions/{session_id}/upload'
        owned_session(session_id, user_id)
        guard = PathGuard(workspace_root / session_id)
        saved = []

        for upload in files:
            
            try:
                name = guard.safe_upload_name(upload.filename or "")
                target = guard.resolve(name, capability="inputs", for_write=True)
            except SecurityViolation as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

            content = await upload.read(20 * 1024 * 1024 + 1)  
            if len(content) > 20 * 1024 * 1024:
                raise HTTPException(status_code=413, detail="上传文件超过 20MB")

            target.write_bytes(content)
            saved.append({
                "input_id": new_id("input"),
                "name": name,
                "size": len(content),
            })

        return {"files": saved}

    @router.get("/sessions/{session_id}/artifacts")
    async def list_artifacts(
        session_id: str,
        before: str | None = None,
        limit: int = 50,
        user_id: str = Depends(owner),
    ):
        'GET /api/sessions/{session_id}/artifacts?before=2024-01-01T00:00:00&limit=50'
        owned_session(session_id, user_id)
        return {
            "artifacts": store.list_artifacts(
                session_id, user_id, before_created_at=before, limit=limit,
            ),
        }

    @router.get("/artifacts/{artifact_id}/download")
    async def download_artifact(artifact_id: str, user_id: str = Depends(owner)):
        'GET /api/artifacts/{artifact_id}/download'
        
        try:
            artifact = store.get_artifact(artifact_id, user_id)
        except ResourceNotFound as exc:
            raise HTTPException(status_code=404, detail="文件不存在") from exc

        guard = PathGuard(workspace_root / artifact["session_id"])
        relative = Path(artifact["relative_path"])
        if not relative.parts or relative.parts[0] != "artifacts":
            raise HTTPException(status_code=404, detail="文件不存在")
        try:
            target = guard.resolve(
                Path(*relative.parts[1:]).as_posix(), capability="artifacts",
            )
        except SecurityViolation as exc:
            raise HTTPException(status_code=404, detail="文件不存在") from exc

        if not target.is_file():
            raise HTTPException(status_code=404, detail="文件不存在")

        try:
            store.verify_artifact_file(artifact_id, user_id, target)
        except ArtifactIntegrityError as exc:
            raise HTTPException(status_code=409, detail="文件完整性校验失败") from exc

        return FileResponse(
            target,
            filename=artifact["name"],
            media_type=artifact["media_type"],
        )

    @router.post("/tasks/{task_id}/cancel")
    async def cancel_task(task_id: str, user_id: str = Depends(owner)):
        'POST /api/tasks/{task_id}/cancel'
        try:
            store.get_task(task_id, user_id)
        except ResourceNotFound as exc:
            raise HTTPException(status_code=404, detail="任务不存在") from exc
        store.update_task_status(task_id, user_id, "cancelled")
        return {"task_id": task_id, "status": "cancelled"}

    @router.websocket("/ws/runtime/{session_id}")
    async def runtime_websocket(
        websocket: WebSocket,
        session_id: str,
        after: int = 0,
        user_id: str = Depends(current_user),
    ):
        'WS /api/ws/runtime/{session_id}?after=42'
        
        try:
            store.get_session(session_id, user_id)
            subscription = await event_bus.subscribe(
                session_id, user_id, after_event_id=after,
            )
        except ResourceNotFound:
            await websocket.close(code=4404)  
            return

        await websocket.accept()

        async def send_events() -> None:
            while True:
                event = await subscription.get()     
                await websocket.send_json(           
                    event.model_dump(mode="json"),
                )

        sender = asyncio.create_task(send_events())

        try:
            
            while True:
                message = await websocket.receive_text()
                if message == "ping":
                    await websocket.send_json({"type": "pong"})
        except WebSocketDisconnect:
            pass  
        finally:
            
            sender.cancel()
            subscription.close()

    return router
