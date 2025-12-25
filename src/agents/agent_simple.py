"""
学术侦探系统 - 简化版Agent实现
"""
import os
import json
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from coze_coding_utils.runtime_ctx.context import Context, default_headers
from utils.helper.graph_helper import is_dev_env

# 导入工具
from agents.agent_tools import (
    collect_academic_data_tool, analyze_research_trends_tool, identify_blue_ocean_topics_tool,
    align_cross_language_concepts_tool, create_knowledge_graph_tool, generate_research_report_tool,
    perform_system_reflection_tool, search_academic_content_tool, search_chinese_content_tool,
    set_context
)

LLM_CONFIG = "config/academic_detective_config.json"

in_memory_checkpointer = None

# 开发环境默认使用内存记忆
if is_dev_env():
    in_memory_checkpointer = MemorySaver()


def build_agent(ctx=None):
    """构建学术侦探Agent"""
    workspace_path = os.getenv("COZE_WORKSPACE_PATH", "/workspace/projects")
    config_path = os.path.join(workspace_path, LLM_CONFIG)
    
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    
    api_key = os.getenv("COZE_WORKLOAD_IDENTITY_API_KEY")
    base_url = os.getenv("COZE_INTEGRATION_MODEL_BASE_URL")
    
    # 设置全局上下文
    if ctx:
        set_context(ctx)
    
    llm = ChatOpenAI(
        model=cfg['config'].get("model"),
        api_key=api_key,
        base_url=base_url,
        temperature=cfg['config'].get('temperature', 0.7),
        streaming=True,
        timeout=cfg['config'].get('timeout', 600),
        extra_body={
            "thinking": {
                "type": cfg['config'].get('thinking', 'disabled')
            }
        },
        default_headers=default_headers(ctx) if ctx else {}
    )
    
    # 定义工具列表
    tools = [
        collect_academic_data_tool,
        analyze_research_trends_tool,
        identify_blue_ocean_topics_tool,
        align_cross_language_concepts_tool,
        create_knowledge_graph_tool,
        generate_research_report_tool,
        perform_system_reflection_tool,
        search_academic_content_tool,
        search_chinese_content_tool
    ]
    
    # 创建Agent
    agent = create_agent(
        model=llm,
        system_prompt=cfg.get("sp"),
        tools=tools,
        checkpointer=in_memory_checkpointer
    )
    
    return agent


# 主要的处理函数
def process_academic_detective_request(user_request: str, ctx: Context = None) -> str:
    """处理学术侦探请求的主函数"""
    try:
        # 设置上下文
        if ctx:
            set_context(ctx)
        
        # 构建agent
        agent = build_agent(ctx)
        
        # 简单的响应
        response = {
            "request": user_request,
            "status": "success",
            "message": "学术侦探系统处理完成",
            "note": "这是一个简化版本的响应，实际的Agent执行需要通过langchain运行"
        }
        
        return json.dumps(response, ensure_ascii=False, indent=2)
        
    except Exception as e:
        error_response = {
            "error": str(e),
            "message": "学术侦探系统处理失败",
            "request": user_request
        }
        return json.dumps(error_response, ensure_ascii=False, indent=2)