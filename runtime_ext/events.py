from __future__ import annotations

import asyncio
from collections import defaultdict

from .models import RuntimeEvent
from .store import MemoryStore

class EventSubscription:
    '代表"一个订阅者"和 EventBus 之间的连接。'

    def __init__(self, bus: "EventBus", session_id: str, queue: asyncio.Queue[RuntimeEvent]):
        """
        由 EventBus.subscribe() 内部调用，外部不需要直接创建。

        参数：
          bus: EventBus — 所属的事件总线，close 时需要通知它
          session_id: str — 订阅的是哪个会话的事件
          queue: asyncio.Queue[RuntimeEvent] — 这个订阅者专属的事件队列
        """
        self._bus = bus
        self._session_id = session_id
        self._queue = queue

    async def get(self) -> RuntimeEvent:
        """
        阻塞等待并返回下一个事件。

        这是一个 async 方法，调用它会"等着"，直到有新事件到来。
        类比：手机收到微信消息前，屏幕是黑的；消息一来，屏幕就亮了。

        返回：
          RuntimeEvent — 下一个到达的事件对象

        注意：
          如果 EventBus 的 publish() 发了事件，这里会立即收到。
          如果没有事件，协程会在这里挂起，不会浪费 CPU。
        """
        return await self._queue.get()

    def close(self) -> None:
        '取消订阅，关闭这个连接。'
        self._bus._subscribers[self._session_id].discard(self._queue)

class EventBus:
    '事件总线：管理所有订阅者，负责事件的"先存后发"。'

    def __init__(self, store: MemoryStore, *, queue_size: int = 100):
        """
        参数：
          store: MemoryStore — 数据库存储层，事件需要先持久化再广播
          queue_size: int = 100 — 每个订阅者队列的最大容量。
            如果订阅者处理太慢导致队列满了，publish 会丢弃这个订阅者（防止内存爆炸）。
        """
        self.store = store
        self.queue_size = queue_size

        self._subscribers: dict[str, set[asyncio.Queue[RuntimeEvent]]] = defaultdict(set)

    async def subscribe(
        self, session_id: str, user_id: str, *, after_event_id: int | None = None
    ) -> EventSubscription:
        '订阅某个会话的事件流。'
        
        self.store.get_session(session_id, user_id)

        queue: asyncio.Queue[RuntimeEvent] = asyncio.Queue(maxsize=self.queue_size)

        if after_event_id is not None:
            for item in self.store.list_events(session_id, user_id, after_event_id=after_event_id):
                
                queue.put_nowait(RuntimeEvent(
                    event_id=item["event_id"], session_id=item["session_id"],
                    task_id=item["task_id"], run_id=item["run_id"],
                    agent_id=item["agent_id"], type=item["type"],
                    payload=item["payload"], timestamp=item["timestamp"],
                ))

        self._subscribers[session_id].add(queue)
        return EventSubscription(self, session_id, queue)

    async def publish(self, event: RuntimeEvent, user_id: str | None = None) -> RuntimeEvent:
        '发布一个事件：先持久化到数据库，再广播给所有订阅者。'
        
        if user_id is None:
            
            user_id = self.store.connection.execute(
                "SELECT user_id FROM sessions WHERE session_id=?", (event.session_id,)
            ).fetchone()[0]

        persisted = self.store.append_event(event, user_id)

        dead: list[asyncio.Queue[RuntimeEvent]] = []
        for queue in self._subscribers.get(event.session_id, set()):
            try:

                queue.put_nowait(persisted)
            except asyncio.QueueFull:
                dead.append(queue)

        for queue in dead:
            self._subscribers[event.session_id].discard(queue)

        return persisted
