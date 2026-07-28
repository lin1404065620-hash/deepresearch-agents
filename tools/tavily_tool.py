

from typing import  Literal

from langchain_core.tools import tool

from tavily import TavilyClient

import os  
from dotenv import load_dotenv  

from api.monitor import monitor

load_dotenv()

def _get_tavily_client() -> TavilyClient:
    """按需创建客户端，避免未启用网络搜索时强制要求 Tavily 配置。"""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError("未配置 TAVILY_API_KEY，无法使用网络搜索工具")
    return TavilyClient(api_key=api_key)

@tool
def internet_search(
        query: str,
        topic: Literal[ "news",  "finance",  "general"] = "general",
        max_results: int = 5,
        include_raw_content: bool = False
):
    """
    根据用户问题，进行网络信息收！ 
    注意：主要搜索公开的网络信息！如果指定查询数据库或者rag不能使用此工具！
    :param query: 用户的查询信息
    :param topic: 查询的类型
    :param max_results: 返回的最大条数 
    :param include_raw_content: 是否返回原内容 False 精简 True 详细
    :return: 
    """

    monitor.report_tool(tool_name="网络搜索工具",
                        args={"query": query, "topic": topic, "max_results": max_results,
                              "include_raw_content": include_raw_content})

    return _get_tavily_client().search(
        query=query,
        topic=topic,
        max_results=max_results,
        include_raw_content=include_raw_content,
    )

