import os
import asyncio
from typing import Optional
import dashscope
from qwen_agent.agents import Assistant
from qwen_agent.gui import WebUI
import pandas as pd
from sqlalchemy import create_engine, text
from qwen_agent.tools.base import BaseTool, register_tool

# 定义资源文件根目录
ROOT_RESOURCE = os.path.join(os.path.dirname(__file__), 'resource')

# 配置 DashScope
dashscope.api_key = os.getenv('DASHSCOPE_API_KEY', '')  # 从环境变量获取 API Key
dashscope.timeout = 30  # 设置超时时间为 30 秒

# ====== 门票助手 system prompt 和函数描述 ======
system_prompt = """我是门票助手，以下是关于门票订单表相关的字段，我可能会编写对应的SQL(运行环境是MySQL8.0)，对数据进行查询
-- 门票订单表
CREATE TABLE tkt_orders (
    order_time DATETIME,             -- 订单日期
    account_id INT,                  -- 预定用户ID
    gov_id VARCHAR(18),              -- 商品使用人ID（身份证号）
    gender VARCHAR(10),              -- 使用人性别
    age INT,                         -- 年龄
    province VARCHAR(30),           -- 使用人省份
    SKU VARCHAR(100),                -- 商品SKU名
    product_serial_no VARCHAR(30),  -- 商品ID
    eco_main_order_id VARCHAR(20),  -- 订单ID
    sales_channel VARCHAR(20),      -- 销售渠道
    status VARCHAR(30),             -- 商品状态
    order_value DECIMAL(10,2),       -- 订单金额
    quantity INT                     -- 商品数量
);
一日门票，对应多种SKU：
Universal Studios Beijing One-Day Dated Ticket-Standard
Universal Studios Beijing One-Day Dated Ticket-Child
Universal Studios Beijing One-Day Dated Ticket-Senior
二日门票，对应多种SKU：
USB 1.5-Day Dated Ticket Standard
USB 1.5-Day Dated Ticket Discounted
一日门票、二日门票查询
SUM(CASE WHEN SKU LIKE 'Universal Studios Beijing One-Day%' THEN quantity ELSE 0 END) AS one_day_ticket_sales,
SUM(CASE WHEN SKU LIKE 'USB%' THEN quantity ELSE 0 END) AS two_day_ticket_sales
我将回答用户关于门票相关的问题

【工具使用强制规则】
你必须使用 exc_sql 工具来查询数据库，流程如下：
1. 根据用户问题，结合上面的表结构，编写一条 MySQL 查询 SQL；
2. 调用 exc_sql 工具，将 SQL 作为 sql_input 参数传入执行；
3. 仅基于 exc_sql 返回的真实数据回答用户。
严禁：(a) 自己编写 Python/pandas 代码进行计算；(b) 使用任何模拟、虚构或编造的数据；(c) 在未调用 exc_sql 的情况下给出任何统计结果。
"""

# ====== exc_sql 工具类实现 ======
@register_tool('exc_sql')
class ExcSQLTool(BaseTool):
    """
    SQL查询工具，执行传入的SQL语句并返回结果。
    """
    description = '对于生成的SQL，进行SQL查询'
    parameters = {
        'type': 'object',
        'properties': {
            'sql_input': {
                'type': 'string',
                'description': '生成的SQL语句',
            },
            'database': {
                'type': 'string',
                'description': '数据库名，默认 ubr',
            },
        },
        'required': ['sql_input'],
    }

    def call(self, params: str, **kwargs) -> str:
        import json
        import re
        # 容错解析：模型有时返回非标准 JSON（如含未转义换行/引号），避免单次畸形参数中断整个对话
        if isinstance(params, dict):
            args = params
        else:
            try:
                args = json.loads(params)
            except (json.JSONDecodeError, TypeError):
                m = re.search(r'"sql_input"\s*:\s*"(.*)"\s*}', params, re.DOTALL)
                args = {'sql_input': m.group(1)} if m else {'sql_input': params.strip()}
        sql_input = (args.get('sql_input') or '').strip()
        if not sql_input:
            return '错误：未提供有效的 sql_input 参数。'
        database = args.get('database', 'ubr')
        # 在控制台 / 终端打印 SQL，便于调试（TUI 模式也能看到）
        print(f"[回传的SQL] {sql_input}, 数据库: {database}")
        # 同时写入日志文件：GUI(WebUI)模式下 stdout 可能被框架缓冲/重定向，
        # 文件日志无论哪种模式都能可靠查看每次生成的 SQL
        try:
            from datetime import datetime
            log_path = os.path.join(os.path.dirname(__file__), 'sql_calls.log')
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {sql_input}\n")
        except Exception:
            pass
        # 创建数据库连接
        engine = create_engine(
            f"mysql+pymysql://student123:student321@rm-uf6z891lon6dxuqblqo.mysql.rds.aliyuncs.com:3306/{database}?charset=utf8mb4",
            connect_args={"connect_timeout": 10},
            pool_size=10,
            max_overflow=20,
        )
        try:
            df = pd.read_sql(text(sql_input), engine)
            # 在工具回传内容前附上执行的 SQL，UI 上即可看到模型生成的 SQL
            sql_block = f"> **模型生成的 SQL：**\n```sql\n{sql_input}\n```\n\n"
            # 返回前10行，防止数据过多
            return sql_block + df.head(10).to_markdown(index=False)
        except Exception as e:
            # 解开 SQLAlchemy 的包装，露出真正的底层错误（否则只显示无用的 "Failed raising error"）
            orig = getattr(e, 'orig', None) or getattr(e, '__cause__', None)
            real = str(orig) if orig else str(e)
            return f"SQL执行出错: {real}\n\n执行的SQL:\n```sql\n{sql_input}\n```"

# ====== 初始化门票助手服务 ======
def init_agent_service():
    """初始化门票助手服务"""
    llm_cfg = {
        # 'model': 'qwen-max',  # 课程原模型；对 Function Calling 支持不稳定
        'model': 'qwen-max',  # 确定支持 Function Calling，解决“工具未被调用 / 日志为空”的问题
        'timeout': 30,
        'retry_count': 3,
    }
    try:
        bot = Assistant(
            llm=llm_cfg,
            name='门票助手',
            description='门票查询与订单分析',
            system_message=system_prompt,
            function_list=['exc_sql'],  # 只传工具名字符串
        )
        print(f"助手初始化成功！当前模型: {llm_cfg['model']}")
        return bot
    except Exception as e:
        print(f"助手初始化失败: {str(e)}")
        raise

def app_tui():
    """终端交互模式
    
    提供命令行交互界面，支持：
    - 连续对话
    - 文件输入
    - 实时响应
    """
    try:
        # 初始化助手
        bot = init_agent_service()

        # 对话历史
        messages = []
        while True:
            try:
                # 获取用户输入
                query = input('user question: ')
                # 获取可选的文件输入
                file = input('file url (press enter if no file): ').strip()
                
                # 输入验证
                if not query:
                    print('user question cannot be empty！')
                    continue
                    
                # 构建消息
                if not file:
                    messages.append({'role': 'user', 'content': query})
                else:
                    messages.append({'role': 'user', 'content': [{'text': query}, {'file': file}]})

                print("正在处理您的请求...")
                # 运行助手并处理响应
                response = []
                for response in bot.run(messages):
                    print('bot response:', response)
                messages.extend(response)
            except Exception as e:
                print(f"处理请求时出错: {str(e)}")
                print("请重试或输入新的问题")
    except Exception as e:
        print(f"启动终端模式失败: {str(e)}")


def app_gui():
    """图形界面模式，提供 Web 图形界面"""
    try:
        print("正在启动 Web 界面...")
        # 初始化助手
        bot = init_agent_service()
        # 配置聊天界面，列举3个典型门票查询问题
        chatbot_config = {
            'prompt.suggestions': [
                '2023年4、5、6月一日门票，二日门票的销量多少？帮我按照周进行统计',
                '2023年7月的不同省份的入园人数统计',
                '帮我查看2023年10月1-7日销售渠道订单金额排名',
            ]
        }
        print("Web 界面准备就绪，正在启动服务...")
        # 启动 Web 界面
        WebUI(
            bot,
            chatbot_config=chatbot_config
        ).run()
    except Exception as e:
        print(f"启动 Web 界面失败: {str(e)}")
        print("请检查网络连接和 API Key 配置")


if __name__ == '__main__':
    # 运行模式选择
    app_gui()          # 图形界面模式（默认）