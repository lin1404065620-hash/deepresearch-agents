import os
from dotenv import load_dotenv
from api.monitor import monitor
from mysql.connector import connect, Error
from langchain_core.tools import tool

load_dotenv()

def get_db_config():
    """从环境变量获取数据库连接配置"""
    config = {
        "host": os.getenv("MYSQL_HOST", "localhost"),
        "port": int(os.getenv("MYSQL_PORT", "3306")),      
        "user": os.getenv("MYSQL_USER"),                    
        "password": os.getenv("MYSQL_PASSWORD"),            
        "database": os.getenv("MYSQL_DATABASE"),            
        "charset": os.getenv("MYSQL_CHARSET", "utf8mb4"),   
        "collation": os.getenv("MYSQL_COLLATION", "utf8mb4_unicode_ci"),  
        "autocommit": True,                                 
        "sql_mode": os.getenv("MYSQL_SQL_MODE", "TRADITIONAL"),  
        "use_pure": True,                                    
    }
    
    config = {k: v for k, v in config.items() if v is not None}

    required_keys = ["user", "password", "database"]
    missing_keys = [k for k in required_keys if k not in config]
    if missing_keys:
        raise ValueError(f"缺失数据库核心配置：{', '.join(missing_keys)}")

    return config

@tool
def list_sql_tables()->str:
    """
    查询当前库中所有可用的表！
    作用：为了模型识别有哪些可用的表！方便进行后续的自定义sql查询
    :return: 有表： 可用的表有：表1,表2,表3....  没有表: 没有可用的表
    """

    monitor.report_tool(tool_name="数据库表名查询工具：list_sql_tables", args={})
    
    config = get_db_config()

    try:
        
        with connect(**config) as  conn: 
            with conn.cursor() as cursor: 
                sql = "show tables" 
                cursor.execute(sql)

                tables = cursor.fetchall() 
                if not tables:
                    return "没有可用的表"

                table_names = [table[0] for table in tables]
                return f"可用的表有：{', '.join(table_names)}" 
    
    except Exception as e:
        raise RuntimeError("数据库表列表查询失败") from e

@tool
def get_table_data(table_name)->str:
    '根据传入的表名，查询该表前 100 条数据，并把查询结果整理成类似 CSV 的字符串返回给 Agent。'

    monitor.report_tool(tool_name="数据库表数据查询工具：get_table_data", args={"table_name":table_name})

    config = get_db_config()

    try:
        
        with connect(**config) as  conn:
            
            with conn.cursor() as cursor:
                
                sql = f"select * from {table_name} limit 100"
                cursor.execute(sql)

                description = cursor.description
                if not description:
                    return f"数据表：{table_name}为空没有数据！"

                columns = [ desc[0] for desc in description ] 

                rows = cursor.fetchall()

                results = [ ",".join(map(str,row)) for row in rows]

                header_str = ",".join(columns)
                
                data_str = "\n".join(results)
                return f"{header_str}\n{data_str}"
    except Error as e:
        return f"查询出现异常：{str(e)}"

@tool
def execute_sql_query(query)->str:
    '接收一条自定义 SQL 查询语句，连接 MySQL 执行查询，再把结果整理成类似 CSV 的字符串返回给 Agent。 它和前面的 get_table_data() 不同：get_table_data() 只能查询指定表的前 100 条数据，而 execute_sql_query() 可以执行联表、筛选、分组、统计、排序等更复杂的 SQL。'
    
    monitor.report_tool(tool_name="数据库表数据查询工具：execute_sql_query", args={"query":query})

    config = get_db_config()

    try:
        
        with connect(**config) as  conn:
            
            with conn.cursor() as cursor:
                
                cursor.execute(query)

                description = cursor.description
                if not description:
                    return f"执行自定义SQL语句查询没有结果，sql为：{query}！"

                columns = [ desc[0] for desc in description ] 

                rows = cursor.fetchall()

                results = [ ",".join(map(str,row)) for row in rows]

                header_str = ",".join(columns)
                
                data_str = "\n".join(results)
                return f"{header_str}\n{data_str}"
    except Error as e:
        return f"查询出现异常：{str(e)}"

