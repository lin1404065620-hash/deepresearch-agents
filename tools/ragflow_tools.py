

import os
import sys
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from langchain_core.tools import tool

from ragflow_sdk import RAGFlow 
from ragflow_sdk.modules.chat import Chat

from api.monitor import monitor
from rawflow.rag_config import _load_ragflow_env

api_key , base_url =_load_ragflow_env()
ragflow_client = RAGFlow(api_key=api_key,base_url=base_url)

def _list_chats(**filters):
    """兼容 RAGFlow 服务端返回分页对象、SDK 仍按列表解析的情况。"""
    params = {
        "page": 1,
        "page_size": 30,
        "orderby": "create_time",
        "desc": True,
        **filters,
    }
    response = ragflow_client.get("/chats", params)
    payload = response.json()
    if payload.get("code") != 0:
        raise RuntimeError(payload.get("message", "RAGFlow 聊天列表请求失败"))

    data = payload.get("data", [])
    chat_data = data.get("chats", []) if isinstance(data, dict) else data
    return [Chat(ragflow_client, item) for item in chat_data]

@tool
def get_assistant_list() -> str:
    """
    调用此工具，可以查询ragflow服务器中有哪些助手和助手关联的知识库信息！
    供模型参考，可以从哪个助手获取对应的内部文档信息！
    强调：想向某个助手提问，必须想要调用此工具查询助手的信息和名称
    返回结果： 有-> 名称:助手名称,助手描述：xxxx,关联的知识库：知识库的名、知识库的名字、
             没有 -> 没有任何可用助手
             异常 -> 抛出 RAGFlow 查询异常
    :return:
    """

    monitor.report_tool(tool_name="ragflow聊天助手列表查询工具：get_assistant_list")

    try:
        
        chat_list = _list_chats()
        if not chat_list:
            return "没有任何可用助手"
        
        count_chat_info = "" 
        for chat in chat_list:
            dataset_names = []
            
            kb_names = getattr(chat, "kb_names", [])
            if isinstance(kb_names, list):
                dataset_names.extend(kb_names)

            count_chat_info += f"助手名称:{chat.name};功能介绍：{chat.description}; 关联的知识库：{'、'.join(dataset_names)} \n"
        return count_chat_info
    except Exception as e:
        raise RuntimeError("RAGFlow助手列表查询失败") from e

@tool
def create_ask_delete(chat_name,question)->str:
    """
    想某个助手，创建单次会话进行提问，提问完毕以后会关闭会话！
    主要查询ragflow中相关的信息！
    注意：调用此工具之前，必须先调用 get_assistant_list工具明确查询助手的名字和对应的问题
    :param chat_name: 助手的名字！上一个工具get_assistant_list告诉大模型的只有名字
    :param question: 本次提问的问题
    :return: 返回提问的结果
    """
    
    monitor.report_tool(tool_name="ragflow提问助手工具：create_ask_delete",args={"chat_name":chat_name,"question":question})

    try:
        chats = _list_chats(name=chat_name)
        if not chats:
            return f"未找到助手：{chat_name}"

        use_chat = chats[0] 
        session = None
        try:
            
            session = use_chat.create_session(name="temp_session_ask")
            
            response = session.ask(question=question, stream=True)
            result_parts = []
            for part in response:
                content = getattr(part, "content", "")
                if content:
                    result_parts.append(content)
            result = "".join(result_parts)

            return (
                f"本次调用助手：{use_chat.name}\n"
                f"助手 ID：{use_chat.id}\n"
                f"回答内容：{result}"
            )
        finally:
            
            if session is not None:
                use_chat.delete_sessions(ids=[session.id])
    except Exception as e:
        raise RuntimeError("RAGFlow助手提问失败") from e

if __name__ == '__main__':
    chat_name = os.getenv("RAGFLOW_CHAT_NAME")
    question = os.getenv("RAGFLOW_TEST_QUESTION")
    if not chat_name or not question:
        raise SystemExit(
            "请先设置 RAGFLOW_CHAT_NAME 和 RAGFLOW_TEST_QUESTION 再运行此模块"
        )
    print(create_ask_delete.invoke({
        "chat_name": chat_name,
        "question": question,
    }))

