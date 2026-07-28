
from ragflow_sdk import RAGFlow 
from ragflow_sdk.modules.chat import Chat

try:
    
    from rawflow.rag_config import _load_ragflow_env
except ModuleNotFoundError as exc:
    if exc.name != "rawflow":
        raise
    
    from rag_config import _load_ragflow_env

api_key , base_url =_load_ragflow_env()
ragflow_client = RAGFlow(api_key=api_key,base_url=base_url)

def _list_chats(**filters):
    """兼容服务端返回分页对象、SDK 仍按旧版列表解析的情况。"""
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

def get_assistant_list():

    chat_list = _list_chats()
    
    count_chat_info = "" 
    for chat in chat_list:
        dataset_names = []  
        
        kb_names = getattr(chat, "kb_names", [])
        if isinstance(kb_names, list):
            dataset_names.extend(kb_names)

        count_chat_info += f"助手名称:{chat.name};功能介绍：{chat.description}; 关联的知识库：{'、'.join(dataset_names)} \n"

    return count_chat_info

def ask_question(chat_name,question):
    """
    向某个助手发起提问： 1. 创建一个会话 2.提问 3.关闭会话！
    :param chat_name: 助手的名字！上一个工具get_assistant_list告诉大模型的只有名字
    :param question: 本次提问的问题
    :return: 返回提问的结果
    """

    chats = _list_chats(name=chat_name)
    use_chat = chats[0] 
    
    session = use_chat.create_session(name="temp_session_ask")

    response = session.ask(question = question,stream=True)
    
    result = ""
    
    for part in response:
        
        result = part.content

    use_chat.delete_sessions(ids=[session.id])
    
    return result

