"""
学术侦探系统的工具函数
"""
from langchain.tools import tool
from coze_coding_utils.runtime_ctx.context import Context

# 全局上下文变量
_global_ctx = None

def set_context(ctx: Context):
    """设置全局上下文"""
    global _global_ctx
    _global_ctx = ctx

def get_context():
    """获取全局上下文"""
    return _global_ctx

@tool
def collect_academic_data_tool(queries: str, max_results: int = 100) -> str:
    """采集学术数据，支持从多个数据源获取论文信息。参数：queries（查询关键词列表，用逗号分隔），max_results（最大结果数）"""
    from tools.data_collection_tool import collect_academic_data
    return collect_academic_data(get_context(), queries.split(',') if queries else [], max_results)

@tool
def analyze_research_trends_tool(topics: str = "") -> str:
    """分析研究趋势和热点。参数：topics（主题列表，用逗号分隔）"""
    from tools.trend_analysis_tool import analyze_research_trends
    return analyze_research_trends(get_context(), topics.split(',') if topics else [])

@tool
def identify_blue_ocean_topics_tool() -> str:
    """识别蓝海研究主题和机会"""
    from tools.trend_analysis_tool import identify_blue_ocean_topics
    return identify_blue_ocean_topics(get_context())

@tool
def align_cross_language_concepts_tool() -> str:
    """对齐中英文概念"""
    from tools.cross_language_tool import align_cross_language_concepts
    return align_cross_language_concepts(get_context())

@tool
def create_knowledge_graph_tool(topic: str = "") -> str:
    """创建知识图谱。参数：topic（主题名称）"""
    from tools.visualization_tool import create_knowledge_graph
    return create_knowledge_graph(get_context(), topic)

@tool
def generate_research_report_tool() -> str:
    """生成研究报告"""
    from tools.visualization_tool import generate_research_report
    return generate_research_report(get_context())

@tool
def perform_system_reflection_tool() -> str:
    """执行系统反思和优化"""
    from tools.reflection_tool import perform_system_reflection
    return perform_system_reflection(get_context())

@tool
def search_academic_content_tool(topic: str, max_results: int = 10) -> str:
    """搜索学术内容。参数：topic（搜索主题），max_results（最大结果数）"""
    from tools.web_search_tool import search_academic_content
    return search_academic_content(get_context(), topic, max_results)

@tool
def search_chinese_content_tool(topic: str, max_results: int = 10) -> str:
    """搜索中文学术内容。参数：topic（搜索主题），max_results（最大结果数）"""
    from tools.web_search_tool import search_chinese_content
    return search_chinese_content(get_context(), topic, max_results)