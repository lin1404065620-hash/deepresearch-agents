import datetime
import asyncio
from typing import Any, Dict, Optional
from fastapi import WebSocket
from api.context import get_thread_context

try:
    import builtins
except ImportError:
    builtins = None

class ToolMonitor:
    '工具监控类，用于在工具执行过程中上报进度和状态。 设计为单例模式，可在任何工具中直接导入使用。 兼容 FastAPI WebSocket 和 脚本运行时的 stream_writer。'
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ToolMonitor, cls).__new__(cls)
            cls._instance.websocket_manager = None  
        return cls._instance

    def set_websocket_manager(self, manager):
        """设置 FastAPI 的 WebSocket 管理器"""
        self.websocket_manager = manager

    def _emit(self, event_type: str, message: str, data: Optional[Dict[str, Any]] = None):
        """内部发送方法"""
        payload = {
            "type": "monitor_event",
            "event": event_type,
            "message": message,
            "data": data or {},
            "timestamp": datetime.datetime.now().isoformat()
        }

        if self.websocket_manager:
            try:
                
                thread_id = get_thread_context()

                manager_loop = self.websocket_manager.loop

                if manager_loop:
                    if thread_id:
                        
                        try:
                            
                            current_loop = asyncio.get_running_loop()
                            print(f"对比是不是同一个event_loop:{manager_loop == current_loop}")
                        except RuntimeError:
                            current_loop = None

                        if current_loop and current_loop == manager_loop:
                            
                            current_loop.create_task(
                                self.websocket_manager.send_to_thread(payload, thread_id)
                            )
                        else:

                            asyncio.run_coroutine_threadsafe(
                                self.websocket_manager.send_to_thread(payload, thread_id),
                                manager_loop
                            )
                    else:
                        
                        pass
            except Exception as e:
                print(f"[Monitor] WebSocket send failed: {e}")

        if builtins and hasattr(builtins, 'runtime') and hasattr(builtins.runtime, 'stream_writer'):
            try:
                builtins.runtime.stream_writer(payload)
            except Exception:
                pass

        print(f"\n[Monitor:{event_type}] {message}")

    def report_tool(self, tool_name: str, args: Dict[str, Any] = None):
        """报告工具开始执行"""
        self._emit("tool_start", f"开始执行工具: {tool_name}", {"tool_name": tool_name, "args": args})

    def report_assistant(self, assistant_name: str, args: Dict[str, Any] = None):
        """报告正在调用的子智能体进度"""
        self._emit("assistant_call", f"正在调用助手: {assistant_name}",
                   {"assistant_name": assistant_name, "args": args})

    def report_task_result(self, result: str):
        """报告任务最终结果"""
        self._emit("task_result", "任务执行完成", {"result": result})

    def report_session_dir(self, path: str):
        """报告任务工作目录"""
        self._emit("session_created", f"工作目录已创建: {path}", {"path": path})

monitor = ToolMonitor()

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        
        self.loop = None

    def set_loop(self, loop):
        """显式设置事件循环"""
        self.loop = loop
        monitor.set_websocket_manager(self)
        print(f"[Monitor] ConnectionManager manually bound to loop: {id(self.loop)}")

    async def connect(self, websocket: WebSocket, thread_id: str):
        await websocket.accept()
        print(f"存储当前会话id:{thread_id}对应的:{websocket}")
        self.active_connections[thread_id] = websocket
        print(f"Client connected: {thread_id}")

    def disconnect(self, websocket: WebSocket, thread_id: str):
        if thread_id in self.active_connections:
            del self.active_connections[thread_id]
        print(f"Client disconnected: {thread_id}")

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def send_to_thread(self, message: dict, thread_id: str):
        if thread_id in self.active_connections:
            websocket = self.active_connections[thread_id]
            await websocket.send_json(message)

manager = ConnectionManager()