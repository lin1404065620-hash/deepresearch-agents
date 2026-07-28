from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfgen import canvas

from agent.llm import runtime_model
from agent.prompts import main_agent_content, sub_agents_content
from tools.db_tools import execute_sql_query, get_table_data, list_sql_tables
from tools.ragflow_tools import create_ask_delete, get_assistant_list
from tools.tavily_tool import internet_search

from .adapters import LangChainModelAdapter, structured_tool_spec
from .context import get_runtime_context
from .context_manager import ContextManager
from .events import EventBus
from .input_files import list_input_files, read_input_file
from .models import ExecutionContext, RuntimeMessage, new_id
from .orchestration import AgentDefinition, SubAgentExecutor
from .retry import RetryPolicy
from .runtime import AgentRuntime
from .security import PathGuard, SQLGuard, SecurityViolation
from .store import MemoryStore, calculate_file_sha256
from .tools import SideEffect, ToolRegistry, ToolSpec

REPORT_AGENT_MAX_TOKENS = 16_000

REPORT_AGENT_SYSTEM_PROMPT = """
你是研究报告生成专家。根据任务已有证据和当前会话上传的资料，生成内容完整、结构清晰、
可核验的 Markdown 报告。

处理上传资料时必须遵守：
1. 先调用 list_uploaded_inputs，再逐个调用 read_uploaded_input 读取与任务有关的文件。
2. 必须综合文件全文，不得只摘取文件末尾、开头或文件名，不得把“已读取文件”当作完成。
3. 先识别原文栏目和关键信息，再组织报告；原文没有的信息必须标明“原文未提供”，不得编造。
4. 对简历类文件，报告至少覆盖：候选人概况、教育背景、论文与科研成果、科研经历、
   项目经历、技术能力、综合优势、与目标岗位或申请方向的匹配度、风险与改进建议。
5. 重要成果应保留可核验的数量、角色、时间、方法和结果；避免只写“能力较强”等空泛评价。
6. 完整报告应包含一个一级标题和多个二级标题，并提供执行摘要、分项分析和总结建议。

完成内容后必须调用 write_markdown 生成 Markdown；用户明确要求 PDF 时，再调用
markdown_to_pdf。write_markdown 会拒绝标题加一句话式的过短报告，失败时应根据已读取的
完整资料扩写后重试。最后返回 artifact_id、200～400 字的核心摘要和已覆盖栏目，不返回
服务器绝对路径。
""".strip()

def validate_markdown_report(content: str) -> None:
    """
    在报告落盘前执行最低完整性校验。

    该校验不判断报告观点是否正确，只拦截“一个标题加一句泛化评价”这类明显不完整的
    输出，让报告 Agent 能看到工具错误并基于已经读取的资料扩写后重试。
    """
    meaningful_length = len(re.sub(r"\s+", "", content))
    heading_count = sum(
        1 for line in content.splitlines()
        if re.match(r"^\s{0,3}#{1,6}\s+\S", line)
    )
    if meaningful_length < 500 or heading_count < 3:
        raise ValueError(
            "报告内容过于简略：至少需要 500 个非空白字符和 3 个 Markdown 标题，"
            "请综合已读取的完整资料扩写后重新调用 write_markdown"
        )

@dataclass
class RuntimeServices:
    '组装完成的 Runtime 服务集合（成品出厂）。'

    runtime: AgentRuntime

    store: MemoryStore

    event_bus: EventBus

    async def run(self, context: ExecutionContext, query: str) -> str:
        """
        启动一次 Agent 运行（委托给 AgentRuntime.run()）。

        参数：
          context: 执行上下文
          query: 用户问题

        返回：
          str — Agent 的最终文本回复

          context 和 query 是运行所需的输入，self.runtime.run() 负责真正执行 Agent，而
            await 负责异步等待整个 Agent 流程完成，并取得最终回复。
        """
        return await self.runtime.run(context, query)

def build_default_runtime(store: MemoryStore, event_bus: EventBus) -> RuntimeServices:
    '构建默认配置的完整 Agent Runtime。'

    adapter = LangChainModelAdapter(runtime_model)

    async def semantic_summary(source: str) -> str:
        '调用 LLM 生成语义摘要。'
        response = await adapter.generate([
            RuntimeMessage(
                session_id="summary",
                role="system",
                content=(
                    "将历史上下文压缩为结构化中文摘要。只保留：用户目标、已确认约束、"
                    "关键事实、技术决策、工具结果、错误与解决方案、未完成事项。"
                    "不得补充输入中不存在的信息。"
                ),
            ),
            RuntimeMessage(
                session_id="summary",
                role="user",
                content=source,
            ),
        ], [])
        return response.content

    summary_generator = (
        semantic_summary
        if os.getenv("RUNTIME_LLM_SUMMARY_ENABLED", "").casefold() in {"1", "true", "yes"}
        else None
    )

    specialist_tools = ToolRegistry()

    specialist_tools.register(structured_tool_spec(
        internet_search,
        side_effect=SideEffect.READ,          
        allowed_agents={"network"},            
        retry_policy=RetryPolicy(max_attempts=3),  
        timeout_seconds=90,                    
    ))

    specialist_tools.register(structured_tool_spec(
        list_sql_tables,
        side_effect=SideEffect.READ,
        allowed_agents={"database"},
    ))

    allowed_tables = {
        name.strip()
        for name in os.getenv(
            "RUNTIME_ALLOWED_TABLES", "drugs,sales_records,suppliers"
        ).split(",")
        if name.strip()
    }
    sql_guard = SQLGuard(allowed_tables)

    async def safe_table_data(table_name: str) -> str:
        '安全地读取表数据（含表名白名单校验）。'
        
        if table_name.casefold() not in {name.casefold() for name in allowed_tables}:
            raise SecurityViolation("数据表不在 Runtime 白名单中")
        return await get_table_data.ainvoke({"table_name": table_name})

    async def safe_sql_query(query: str) -> str:
        '安全地执行 SQL 查询（含 5 步 SQL 注入检测）。'

        return await execute_sql_query.ainvoke({"query": sql_guard.validate(query)})

    specialist_tools.register(ToolSpec.from_callable(
        safe_table_data, side_effect=SideEffect.READ, allowed_agents={"database"},
    ))
    specialist_tools.register(ToolSpec.from_callable(
        safe_sql_query, side_effect=SideEffect.READ, allowed_agents={"database"},
    ))

    specialist_tools.register(structured_tool_spec(
        get_assistant_list,
        side_effect=SideEffect.READ,
        allowed_agents={"rag"},
    ))
    specialist_tools.register(structured_tool_spec(
        create_ask_delete,
        side_effect=SideEffect.READ,
        allowed_agents={"rag"},
        retry_policy=RetryPolicy(max_attempts=3),  
        timeout_seconds=120,                        
    ))

    async def list_uploaded_inputs() -> list[dict[str, int | str]]:
        '列出当前会话已上传且可读取的输入文件。'
        context = get_runtime_context()
        return list_input_files(context.workspace)

    async def read_uploaded_input(filename: str) -> str:
        '读取当前会话上传的 Markdown、文本或 PDF 文件并提取文本。'

        context = get_runtime_context()
        return read_input_file(context.workspace, filename)

    specialist_tools.register(ToolSpec.from_callable(
        list_uploaded_inputs, side_effect=SideEffect.READ, allowed_agents={"report"},
    ))
    specialist_tools.register(ToolSpec.from_callable(
        read_uploaded_input, side_effect=SideEffect.READ, allowed_agents={"report"},
    ))

    async def write_markdown(content: str, filename: str) -> dict[str, Any]:
        '生成 Markdown 报告文件。'
        validate_markdown_report(content)
        context = get_runtime_context()
        guard = PathGuard(context.workspace)

        safe_name = Path(filename).name
        if not safe_name.casefold().endswith(".md"):
            safe_name += ".md"

        target = guard.resolve(safe_name, capability="artifacts", for_write=True)
        temporary = guard.resolve(f"{new_id('temp')}.md", capability="temp", for_write=True)
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(target)  

        digest = calculate_file_sha256(target)
        artifact_id = new_id("artifact")
        reference = guard.artifact_ref(artifact_id, target, "text/markdown")

        store.record_artifact(
            artifact_id, context.session_id, context.user_id,
            name=reference.name, relative_path=reference.relative_path,
            task_id=context.task_id, run_id=context.run_id,
            media_type=reference.media_type, size=reference.size, sha256=digest,
            idempotency_key=f"markdown:{context.session_id}:{digest}",  
        )
        return reference.model_dump(mode="json")

    async def markdown_to_pdf(
        markdown_filename: str, pdf_filename: str | None = None,
    ) -> dict[str, Any]:
        """
        将 Markdown 文件转换为 PDF。

        参数：
          markdown_filename: 源 Markdown 文件名（必须在 artifacts 目录下）
          pdf_filename: 输出 PDF 文件名（不传则自动从源文件名推导）

        返回：
          dict — ArtifactRef 的 JSON 表示

        实现：
          使用 reportlab 库生成 PDF，每行最多 55 个字符，自动换页。
          中文字体使用 STSong-Light（宋体）。
        """
        context = get_runtime_context()
        guard = PathGuard(context.workspace)

        source = guard.resolve(Path(markdown_filename).name, capability="artifacts")
        if not source.exists():
            raise FileNotFoundError("Markdown artifact 不存在")

        target_name = (
            Path(pdf_filename).name if pdf_filename
            else source.with_suffix(".pdf").name
        )
        if not target_name.casefold().endswith(".pdf"):
            target_name += ".pdf"

        target = guard.resolve(target_name, capability="artifacts", for_write=True)
        temporary = guard.resolve(
            f"{new_id('temp')}.pdf", capability="temp", for_write=True,
        )

        pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
        document = canvas.Canvas(str(temporary))
        document.setFont("STSong-Light", 11)

        y = 800  
        for source_line in source.read_text(encoding="utf-8").splitlines():
            
            chunks = [
                source_line[index:index + 55]
                for index in range(0, len(source_line), 55)
            ] or [""]
            for line in chunks:
                if y < 50:  
                    document.showPage()
                    document.setFont("STSong-Light", 11)
                    y = 800
                document.drawString(45, y, line)
                y -= 16  
        document.save()

        temporary.replace(target)
        digest = calculate_file_sha256(target)
        artifact_id = new_id("artifact")
        reference = guard.artifact_ref(artifact_id, target, "application/pdf")
        store.record_artifact(
            artifact_id, context.session_id, context.user_id,
            name=reference.name, relative_path=reference.relative_path,
            task_id=context.task_id, run_id=context.run_id,
            media_type=reference.media_type, size=reference.size, sha256=digest,
            idempotency_key=f"pdf:{context.session_id}:{digest}",
        )
        return reference.model_dump(mode="json")

    specialist_tools.register(ToolSpec.from_callable(
        write_markdown,
        side_effect=SideEffect.IDEMPOTENT_WRITE,   
        allowed_agents={"report"},
        supports_idempotency=True,                  
    ))
    specialist_tools.register(ToolSpec.from_callable(
        markdown_to_pdf,
        side_effect=SideEffect.IDEMPOTENT_WRITE,
        allowed_agents={"report"},
        supports_idempotency=True,
    ))

    definitions = {

        "network": AgentDefinition(
            "network",
            "网络搜索专家",
            adapter,
            frozenset({"internet_search"}),
            sub_agents_content["tavily"]["system_prompt"],
            required_tool="internet_search",  
            summary_generator=summary_generator,
        ),

        "database": AgentDefinition(
            "database",
            "数据库查询专家",
            adapter,
            frozenset({"list_sql_tables", "safe_table_data", "safe_sql_query"}),
            sub_agents_content["db"]["system_prompt"],
            summary_generator=summary_generator,
        ),

        "rag": AgentDefinition(
            "rag",
            "RAG 知识库专家",
            adapter,
            frozenset({"get_assistant_list", "create_ask_delete"}),
            sub_agents_content["ragflow"]["system_prompt"],
            summary_generator=summary_generator,
        ),

        "report": AgentDefinition(
            "report",
            "研究报告生成专家",
            adapter,
            frozenset({
                "list_uploaded_inputs", "read_uploaded_input",
                "write_markdown", "markdown_to_pdf",
            }),
            REPORT_AGENT_SYSTEM_PROMPT,
            max_tokens=REPORT_AGENT_MAX_TOKENS,
            summary_generator=summary_generator,
        ),
    }

    executor = SubAgentExecutor(definitions, specialist_tools, store, event_bus)

    main_tools = ToolRegistry()

    def delegate_tool(agent_id: str):
        '动态创建委派工具函数。'
        async def delegate(description: str) -> dict[str, Any]:
            context = get_runtime_context()
            result = await executor.delegate(context, agent_id, description)
            if result.get("status") != "succeeded":
                raise RuntimeError("子 Agent 执行失败")
            return result
        return delegate

    for agent_id in definitions:
        main_tools.register(ToolSpec.from_callable(
            delegate_tool(agent_id),
            name=f"delegate_{agent_id}",                          
            description=f"将任务委派给{definitions[agent_id].description}",  
            side_effect=(
                SideEffect.READ if agent_id != "report"
                else SideEffect.IDEMPOTENT_WRITE                   
            ),
            allowed_agents={"main"},                               
            supports_idempotency=(agent_id == "report"),           
            timeout_seconds={
                "network": 120,
                "rag": 240,
                "report": 120,
            }.get(agent_id, 60),  
        ))

    main_prompt = main_agent_content["system_prompt"] + """

【动态检索与兜底规则】
- 使用 delegate_network、delegate_database、delegate_rag 获取证据；需要文件时最后调用 delegate_report。
- 用户要求总结上传的 PDF、附件或其他上传文件时，必须调用 delegate_report，让报告专家通过
  list_uploaded_inputs 和 read_uploaded_input 查找并读取当前会话文件；不得要求用户提供服务器绝对路径。
- “总结/概括上传文件”只要求返回内容时，可以直接汇总报告专家的结论；只有用户明确要求生成、导出、
  保存或下载 Markdown/PDF/报告文件时，才要求报告专家生成可下载 Artifact。
- 用户询问当前时间、天气、新闻、价格、政策、人物职务等实时信息或最新事实时，必须调用 delegate_network 后再回答。
- 对任何你无法确定、知识可能过期、或者现有上下文没有可靠证据的问题，必须调用 delegate_network 搜索，不能凭记忆猜测。
- 不得直接回复无法获取实时信息、请查看手机/电脑，或以业务范围为由拒绝；应先让网络搜索助手查找答案。
- 网络搜索没有得到可靠结果时，才可以如实说明未检索到，并简要列出已尝试的检索方向。
- 不得伪造工具结果或服务器路径。

这段代码完成了主 Agent 的最终配置和封装：首先，main_prompt 中的“动态检索与兜底规
则”是在系统提示词层面要求主 Agent根据任务
调用 delegate_network、delegate_database、delegate_rag 获取证据，需要输出文件时再
调用 delegate_report；其中实时信息、可能过期的信息或缺少可靠依据的问题必须先进行网
络搜索，搜索无可靠结果后才能如实说明，且禁止伪造工具结果和服务器路径。需要注意，这些规
则主要是对模型的行为约束，并不是单独的 Python 强制校验逻辑，真正可调用哪些工具仍由前面
注册的 main_tools 和权限配置决定。随后代码创建 AgentRuntime：adapter 负责调用
模型，main_tools 提供主 Agent 的委派工具，store 保存运行状态、记忆和制品记录，event_bus 负责发送
运行事件；ContextManager 管理本次对话送入模型的上下文，配置的上下文预算为 8000 token、预留
约 1024 token 给模型输出，并设置 256 token 的安全余量，内容过长时可使用前面配置
的可选 summary_generator 做压缩；max_steps=20 限制一次 Agent 执行循环最多进行 20 个步骤，防止无
限调用工具，system_prompt=main_prompt 则把上述规则交给主 Agent。最后，代码把创建
好的 runtime、store 和 event_bus 封装进 RuntimeServices 返回，方便外部通过一个服务对象统一启
动 Agent、访问存储并监听事件。
"""

    runtime = AgentRuntime(
        adapter,
        main_tools,
        store,
        event_bus,
        context_manager=ContextManager(
            max_tokens=8_000,
            output_budget=1_024,
            safety_margin=256,
            summary_generator=summary_generator,
        ),
        max_steps=20,
        system_prompt=main_prompt,
    )

    return RuntimeServices(runtime, store, event_bus)
