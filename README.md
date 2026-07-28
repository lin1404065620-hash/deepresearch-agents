# Deep Search Pro

Deep Search Pro 是一个面向研究型任务的多智能体报告生成系统。主 Agent 可以把网络搜索、
MySQL 查询、RAGFlow 检索和报告生成任务委派给专用子 Agent，并在显式 Agent Loop 中完成
工具调用、结果回填、失败重试、上下文压缩和最终汇总。

## 核心能力

- 主 Agent 与网络、数据库、RAG、报告子 Agent 协作
- ToolSpec / ToolRegistry 工具注册、JSON Schema 生成和权限隔离
- 模型与工具分级重试、幂等键和副作用声明
- Session、Task、Run 分离的 SQLite 持久化
- 滚动摘要、结构化记忆和相关情节召回
- WebSocket EventBus 实时进度推送和断线事件补发
- SQL 只读校验、路径穿越防护和会话所有权校验
- 上传 PDF、Markdown、文本和办公文档后生成 Markdown/PDF 报告

## 项目结构

```text
.
├── agent/          # 模型、提示词和兼容的 DeepAgents 入口
├── api/            # FastAPI 服务入口与 WebSocket 监控
├── frontend/       # 单页 Web 客户端
├── mysql/          # 可选的 MySQL 数据导入脚本
├── prompt/         # YAML 提示词配置
├── rawflow/        # RAGFlow SDK 适配
├── runtime_ext/    # 轻量 Agent Runtime 核心
├── tools/          # 网络、数据库、RAG 和文件工具
├── utils/          # 路径与文档转换辅助函数
├── .env.example    # 安全的环境变量模板
└── requirements.txt
```

## 环境要求

- Python 3.11 或 3.12
- 可访问的 OpenAI 兼容模型服务
- 可选：Tavily、RAGFlow、MySQL
- Windows 下如需通过 Microsoft Word 转换 PDF，需要安装 Word

## 安装

建议在虚拟环境中安装依赖。中国大陆网络环境可使用清华 PyPI 镜像：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

复制环境变量模板并填入自己的服务配置：

```powershell
Copy-Item .env.example .env
```

不要提交 `.env`。未使用的 MySQL、RAGFlow 或 Tavily 功能可以暂不配置，但调用对应 Agent
前必须提供相关环境变量。

## 启动

在项目根目录运行：

```powershell
python -m api.server
```

默认服务地址为 `http://localhost:8000`，根路径会返回前端页面，接口说明可在本地
`/docs` 路径查看。

## 主要运行流程

```text
用户创建会话并提交问题
        ↓
主 Agent 加载会话记忆与最近消息
        ↓
按任务委派 network / database / rag / report
        ↓
子 Agent 在隔离上下文中调用获授权工具
        ↓
工具结果写回 Agent Loop，主 Agent 继续决策
        ↓
生成回答或 Markdown/PDF Artifact
        ↓
SQLite 持久化 + WebSocket 推送完成事件
```

## 数据与安全边界

- `.env`、数据库、上传文件和生成报告均被 `.gitignore` 排除。
- 前端只接触会话 ID、Artifact ID 和受控下载接口，不接触服务器绝对路径。
- MySQL 工具只允许单条只读查询，并对表名和危险关键字进行校验。
- 文件操作由受控 Workspace 和 PathGuard 限制，防止 `../` 路径穿越。
- 当前 SQLite 实现定位于单机、中小规模部署；多机部署需要替换为共享数据库、消息队列
  和对象存储。

## 可选数据导入

将自己的 Excel 文件放在本地，并通过 `MYSQL_IMPORT_FILE` 指定路径：

```powershell
python -m mysql.excel_to_mysql
```

示例数据不随仓库发布，数据库连接信息必须通过环境变量提供。

## 发布前检查

提交前至少确认：

```powershell
python -m compileall agent api rawflow runtime_ext tools utils
python -c "from runtime_ext.store import MemoryStore; from runtime_ext.security import PathGuard; print('import ok')"
```

本仓库不包含真实 API Key、数据库凭据、服务器地址、用户上传文件或本地运行数据。
