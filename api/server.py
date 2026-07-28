import uuid
import asyncio
import uvicorn
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import shutil
import os
import re
import logging

import sys
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

from api.monitor import manager
from runtime_ext.api import create_runtime_router
from runtime_ext.events import EventBus
from runtime_ext.store import MemoryStore
from runtime_ext.bootstrap import build_default_runtime
from runtime_ext.recovery import RecoveryCoordinator

app = FastAPI(title="DeepAgents API")
logger = logging.getLogger(__name__)
frontend_file = project_root / "frontend" / "index.html"

runtime_store = MemoryStore(project_root / "data" / "agent_runtime.db")
runtime_event_bus = EventBus(runtime_store)
runtime_workspace_root = project_root / "runtime_workspaces"
runtime_workspace_root.mkdir(parents=True, exist_ok=True)
runtime_services = build_default_runtime(runtime_store, runtime_event_bus)
runtime_recovery = RecoveryCoordinator(
    store=runtime_store,
    runner=runtime_services.run,
    workspace_root=runtime_workspace_root,
    worker_id="single-worker",
    max_recovery_attempts=3,
)
legacy_agent_api_enabled = os.getenv("ENABLE_LEGACY_AGENT_API", "").lower() in {"1", "true", "yes"}
legacy_path_api_enabled = os.getenv("ENABLE_LEGACY_PATH_API", "").lower() in {"1", "true", "yes"}

def get_runtime_user() -> str:
    return os.getenv("RUNTIME_DEV_USER_ID", "local-developer")

app.include_router(create_runtime_router(
    store=runtime_store,
    event_bus=runtime_event_bus,
    current_user=get_runtime_user,
    workspace_root=runtime_workspace_root,
    runner=runtime_services.run,
))

@app.get("/", include_in_schema=False)
async def frontend_home():
    return FileResponse(frontend_file)

output_dir = project_root / "output"   
output_dir.mkdir(exist_ok=True)

updated_dir = project_root / "updated"
updated_dir.mkdir(exist_ok=True)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        origin.strip()
        for origin in os.getenv("CORS_ALLOW_ORIGINS", "http://localhost:8000").split(",")
        if origin.strip()
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TaskRequest(BaseModel):
    query: str
    thread_id: str = None

@app.on_event("startup")
async def startup_event():
    """
    服务启动时，获取当前运行的事件循环，并绑定到 WebSocket 管理器。
    确保后台线程能通过 run_coroutine_threadsafe 准确投递消息。
    """
    loop = asyncio.get_running_loop()   
    manager.set_loop(loop)  

    await runtime_recovery.recover_once(background=True)
    logger.info("WebSocket manager bound to event loop")

@app.post("/api/task")  
async def run_task(request: TaskRequest):
    if not legacy_agent_api_enabled:
        raise HTTPException(status_code=410, detail="旧任务接口已停用，请使用会话级任务接口")
    from agent.main_agent import run_deep_agent

    thread_id = request.thread_id or str(uuid.uuid4())

    asyncio.create_task(run_deep_agent(request.query, thread_id))

    return {"status": "started", "thread_id": thread_id}

@app.post("/api/upload")
async def upload_files(files: List[UploadFile] = File(...), thread_id: str = Form(...)):
    if not legacy_path_api_enabled:
        raise HTTPException(status_code=410, detail="旧路径上传接口已停用，请使用会话输入接口")
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", thread_id):
        raise HTTPException(status_code=400, detail="无效的 thread_id")
    
    target_dir = updated_dir / f"session_{thread_id}"
    target_dir.mkdir(parents=True, exist_ok=True)

    saved_files = []
    
    for file in files:
        safe_name = Path(file.filename or "").name
        if not safe_name or safe_name != file.filename:
            raise HTTPException(status_code=400, detail="无效的文件名")
        file_path = target_dir / safe_name

        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        saved_files.append(file.filename)

    return {"status": "uploaded", "files": saved_files}

@app.get("/api/download")
async def download_file(path: str):
    if not legacy_path_api_enabled:
        raise HTTPException(status_code=410, detail="旧路径下载接口已停用，请使用 artifact ID 接口")
    
    try:
        abs_path = Path(path).resolve()
        output_abs = output_dir.resolve()

        if not abs_path.is_relative_to(output_abs):
            return {"error": "拒绝访问: 只能下载输出目录下的文件"}
    except Exception:
        return {"error": "无效的路径参数"}
    
    if not abs_path.exists():
        return {"error": "文件不存在"}

    return FileResponse(abs_path, filename=abs_path.name)

@app.get("/api/files")
async def list_files(path: str):
    if not legacy_path_api_enabled:
        raise HTTPException(status_code=410, detail="旧路径文件接口已停用，请使用制品列表接口")
    try:
        
        abs_path = Path(path).resolve()
        output_abs = output_dir.resolve()

        if not abs_path.is_relative_to(output_abs):
            return {"error": "拒绝访问: 只能访问输出目录下的文件"}

    except Exception:
        return {"error": "路径无效"}

    if not abs_path.exists():
        return {"error": "目录不存在"}

    files = []
    try:
        
        for file_path in abs_path.rglob("*"):
            if file_path.is_file():
                
                stat = file_path.stat()
                files.append({
                    "name": file_path.name,
                    "type": "file",
                    "path": file_path.relative_to(output_abs).as_posix(),
                    "size": stat.st_size,
                    "mtime": stat.st_mtime
                })

    except Exception:
        logger.exception("Failed to enumerate legacy output files")
        return {"error": "文件遍历失败"}

    files.sort(key=lambda x: x.get("mtime", 0), reverse=True)
    return {"files": files}

@app.websocket("/ws/{thread_id}")
async def websocket_endpoint(websocket: WebSocket, thread_id: str):
    
    await manager.connect(websocket, thread_id)

    try:
        
        while True:
            
            data = await websocket.receive_text()

            await websocket.send_json({
                "type": "pong",
                "message": f"服务端已收到: {data}"
            })

    except WebSocketDisconnect:
        
        manager.disconnect(websocket, thread_id)
        logger.info("Legacy WebSocket client disconnected")

    except Exception:
        logger.exception("Legacy WebSocket connection failed")
        manager.disconnect(websocket, thread_id)

if __name__ == "__main__":
    uvicorn.run("api.server:app", host="0.0.0.0", port=8000, reload=True)
