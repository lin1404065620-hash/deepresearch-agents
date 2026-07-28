import sys
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from agent.subagents.knowledge_base_agent import knowledge_base_agent
from agent.subagents.database_query_agent import database_query_agent
from agent.subagents.network_search_agent import network_search_agent
from langgraph.checkpoint.memory import InMemorySaver

from tools.markdown_tools import generate_markdown
from tools.pdf_tools import convert_md_to_pdf
from tools.upload_file_read_tool import read_file_content

from deepagents import create_deep_agent

from agent.llm import model
from agent.prompts import main_agent_content

from api.monitor import monitor
import asyncio
import uuid
import shutil

from api.context import set_session_context, reset_session_context, set_thread_context

from langchain_core.messages import AIMessage
main_agent = create_deep_agent(
   model = model,
   system_prompt=main_agent_content['system_prompt'],
   tools= [generate_markdown,convert_md_to_pdf,read_file_content],
   checkpointer=InMemorySaver(),
   subagents=[
       database_query_agent,
       network_search_agent,
       knowledge_base_agent
   ]
)

project_root_path = Path(__file__).parents[1].resolve() 

async def run_deep_agent(task_query,session_id):
    """
    定义流式+异步执行主智能体！！
    执行过程中，返回  会话文件化返回  调用子智能体  调用最终结果 （monitor）
    task_query: 前端提问的问题
    session_id: 每个前端会话对应的标识 （1.存储session_id ContextVars 2.session_id 给他创建对应的output输出地址）
    """
    print(f"当前会话的main_agent开始执行了！ 会话id:{session_id}")

    session_dir = project_root_path / "output" / f"session_{session_id}"
    
    session_dir.mkdir(parents=True, exist_ok=True)
    
    session_dir_str = str(session_dir).replace("\\","/")

    relative_session_dir_str = str(session_dir.relative_to(project_root_path)).replace("\\","/")

    updated_dir_path = project_root_path / "updated" / f"session_{session_id}"
    updated_info_prompt = "" 
    if updated_dir_path.exists():
        
        files = [ f.name  for f in updated_dir_path.iterdir()  if f.is_file()]
        
        if files:
            for filename in files:
                
                shutil.copy2(updated_dir_path / filename, session_dir / filename)
            
            updated_info_prompt = (f"\n    [已上传文件] 已加载到工作目录:\n" +
                             "\n".join([f"    - {f}" for f in files]) +
                             "\n    请优先使用工具（read_file_content）读取并参考这些文件。")

    session_dir_token = set_session_context(session_dir_str)  
    session_id_token = set_thread_context(session_id)  

    monitor.report_session_dir(session_dir_str)  

    config = {
        "configurable":{
            "thread_id":session_id
        }
    }

    path_instruction = f"""
    【工作环境指令】
    工作目录: {relative_session_dir_str}
    {updated_info_prompt}

    规则：
    1. 新生成文件必须保存到工作目录：'{relative_session_dir_str}/filename'
    2. 读取已上传的文件时，请直接将文件名（例如：'开篇.txt'）作为 filename 参数传入（read_file_content）读取工具，不要带上任何目录前缀。
    3. 使用相对路径，禁止使用绝对路径
    4. 若存在上传文件，请先分析内容
    """
    
    try:
        
        async for chunk in main_agent.astream({
            "messages":[
                {
                    "role":"user","content":task_query+path_instruction
                }
            ]
        },config=config):
            
            for node_name,state in chunk.items():
                if not state or "messages" not in state: continue
                messages = state["messages"]
                if messages and isinstance(messages,list):
                    last_msg = messages[-1]
                    if node_name == 'model':
                        if last_msg.tool_calls:
                            
                            for tool_call in last_msg.tool_calls:
                                """
                                  tool_call = {
                                      name: task
                                      args:{
                                          subagent_type:子智能体的名字
                                          description:子智能体的描述
                                      }
                                  }                                
                                """
                                if tool_call['name'] == 'task':
                                    
                                    monitor.report_assistant(tool_call['args']['subagent_type'],{'description':tool_call['args']['description']})
                        elif last_msg.content:
                            
                            print(f"主智能体执行结果，最终结果：{last_msg.content[:100]}")
                            monitor.report_task_result(last_msg.content)

    except Exception as e :
        
        monitor._emit("error",f"执行主智能发生异常信息：{str(e)}")
    finally:
        
        reset_session_context(session_dir_token, session_id_token)

