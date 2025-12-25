"""
学术侦探系统 - 主要Agent实现
"""
import os
import json
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional, Callable
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from coze_coding_utils.runtime_ctx.context import Context, default_headers
from utils.helper.graph_helper import is_dev_env
from langchain_community.tools import Tool

# 导入工具
from tools.data_collection_tool import collect_academic_data
from tools.trend_analysis_tool import analyze_research_trends, identify_blue_ocean_topics
from tools.cross_language_tool import align_cross_language_concepts, translate_and_map_concept
from tools.visualization_tool import create_knowledge_graph, generate_research_report, create_trend_visualization
from tools.reflection_tool import perform_system_reflection, evaluate_model_performance, analyze_system_bottlenecks
from tools.web_search_tool import search_academic_content, search_chinese_content

LLM_CONFIG = "config/academic_detective_config.json"

in_memory_checkpointer = None

# 开发环境默认使用内存记忆
if is_dev_env():
    in_memory_checkpointer = MemorySaver()


class AcademicDetectiveOrchestrator:
    """学术侦探系统协调者"""
    
    def __init__(self, ctx: Context = None):
        self.ctx = ctx
        self.config = self._load_config()
        self.llm = self._init_llm()
        self.task_queue = []
        self.active_tasks = {}
        
    def _load_config(self) -> Dict[str, Any]:
        """加载配置"""
        workspace_path = os.getenv("COZE_WORKSPACE_PATH", "/workspace/projects")
        config_path = os.path.join(workspace_path, LLM_CONFIG)
        
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _init_llm(self) -> ChatOpenAI:
        """初始化大模型"""
        api_key = os.getenv("COZE_WORKLOAD_IDENTITY_API_KEY")
        base_url = os.getenv("COZE_INTEGRATION_MODEL_BASE_URL")
        
        return ChatOpenAI(
            model=self.config['config'].get("model"),
            api_key=api_key,
            base_url=base_url,
            temperature=self.config['config'].get('temperature', 0.7),
            streaming=True,
            timeout=self.config['config'].get('timeout', 600),
            extra_body={
                "thinking": {
                    "type": self.config['config'].get('thinking', 'disabled')
                }
            },
            default_headers=default_headers(self.ctx) if self.ctx else {}
        )
    
    def plan_workflow(self, user_request: str) -> List[Dict[str, Any]]:
        """规划工作流程"""
        # 分析用户请求并生成执行计划
        workflow_steps = []
        
        # 基于关键词识别需要的操作
        request_lower = user_request.lower()
        
        if any(keyword in request_lower for keyword in ["采集", "收集", "数据", "抓取"]):
            workflow_steps.append({
                "step": 1,
                "agent": "data_collector",
                "action": "collect_academic_data",
                "parameters": {
                    "queries": self._extract_search_queries(user_request),
                    "max_results": 100
                },
                "description": "采集学术数据"
            })
        
        if any(keyword in request_lower for keyword in ["趋势", "分析", "热点", "机会"]):
            workflow_steps.append({
                "step": len(workflow_steps) + 1,
                "agent": "trend_analyst",
                "action": "analyze_research_trends",
                "parameters": {},
                "description": "分析研究趋势和机会"
            })
        
        if any(keyword in request_lower for keyword in ["蓝海", "机会", "潜力"]):
            workflow_steps.append({
                "step": len(workflow_steps) + 1,
                "agent": "trend_analyst",
                "action": "identify_blue_ocean_topics",
                "parameters": {},
                "description": "识别蓝海研究主题"
            })
        
        if any(keyword in request_lower for keyword in ["中英文", "跨语言", "概念", "对齐"]):
            workflow_steps.append({
                "step": len(workflow_steps) + 1,
                "agent": "cross_language",
                "action": "align_cross_language_concepts",
                "parameters": {},
                "description": "跨语言概念对齐"
            })
        
        if any(keyword in request_lower for keyword in ["图谱", "可视化", "报告", "展示"]):
            workflow_steps.append({
                "step": len(workflow_steps) + 1,
                "agent": "visualizer",
                "action": "create_knowledge_graph",
                "parameters": {},
                "description": "生成知识图谱"
            })
            
            workflow_steps.append({
                "step": len(workflow_steps) + 1,
                "agent": "visualizer",
                "action": "generate_research_report",
                "parameters": {},
                "description": "生成研究报告"
            })
        
        if any(keyword in request_lower for keyword in ["反思", "评估", "优化", "性能"]):
            workflow_steps.append({
                "step": len(workflow_steps) + 1,
                "agent": "reflector",
                "action": "perform_system_reflection",
                "parameters": {},
                "description": "执行系统反思"
            })
        
        # 如果没有特定需求，执行默认的完整分析流程
        if not workflow_steps:
            workflow_steps = [
                {
                    "step": 1,
                    "agent": "data_collector",
                    "action": "collect_academic_data",
                    "parameters": {"queries": ["artificial intelligence", "machine learning"], "max_results": 50},
                    "description": "采集学术数据"
                },
                {
                    "step": 2,
                    "agent": "trend_analyst",
                    "action": "analyze_research_trends",
                    "parameters": {},
                    "description": "分析研究趋势"
                },
                {
                    "step": 3,
                    "agent": "visualizer",
                    "action": "create_knowledge_graph",
                    "parameters": {},
                    "description": "生成知识图谱"
                },
                {
                    "step": 4,
                    "agent": "visualizer",
                    "action": "generate_research_report",
                    "parameters": {},
                    "description": "生成研究报告"
                }
            ]
        
        return workflow_steps
    
    def _extract_search_queries(self, user_request: str) -> List[str]:
        """从用户请求中提取搜索关键词"""
        # 简单的关键词提取，可以进一步优化
        common_keywords = [
            "artificial intelligence", "machine learning", "deep learning",
            "neural network", "natural language processing", "computer vision",
            "reinforcement learning", "generative ai", "large language model"
        ]
        
        request_lower = user_request.lower()
        extracted = []
        
        for keyword in common_keywords:
            if keyword in request_lower:
                extracted.append(keyword)
        
        # 如果没有匹配到关键词，使用默认查询
        if not extracted:
            extracted = ["artificial intelligence", "machine learning"]
        
        return extracted
    
    def execute_workflow(self, workflow_steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """执行工作流程"""
        execution_results = []
        current_step = 0
        
        for step in workflow_steps:
            current_step += 1
            step_start_time = datetime.now()
            
            try:
                # 执行步骤
                result = self._execute_step(step)
                
                step_result = {
                    "step": current_step,
                    "agent": step["agent"],
                    "action": step["action"],
                    "description": step["description"],
                    "status": "completed",
                    "start_time": step_start_time.isoformat(),
                    "end_time": datetime.now().isoformat(),
                    "duration": (datetime.now() - step_start_time).total_seconds(),
                    "result": result
                }
                
                execution_results.append(step_result)
                
            except Exception as e:
                step_result = {
                    "step": current_step,
                    "agent": step["agent"],
                    "action": step["action"],
                    "description": step["description"],
                    "status": "failed",
                    "start_time": step_start_time.isoformat(),
                    "end_time": datetime.now().isoformat(),
                    "error": str(e)
                }
                
                execution_results.append(step_result)
                
                # 决定是否继续执行
                if self._should_continue_on_error(step, str(e)):
                    continue
                else:
                    break
        
        # 生成执行摘要
        summary = self._generate_execution_summary(execution_results)
        
        return {
            "status": "completed" if all(r["status"] == "completed" for r in execution_results) else "partial",
            "workflow_id": str(uuid.uuid4()),
            "execution_date": datetime.now().isoformat(),
            "total_steps": len(workflow_steps),
            "completed_steps": len([r for r in execution_results if r["status"] == "completed"]),
            "failed_steps": len([r for r in execution_results if r["status"] == "failed"]),
            "execution_results": execution_results,
            "summary": summary
        }
    
    def _execute_step(self, step: Dict[str, Any]) -> str:
        """执行单个步骤"""
        agent = step["agent"]
        action = step["action"]
        parameters = step.get("parameters", {})
        
        # 调用相应的工具函数
        if agent == "data_collector":
            if action == "collect_academic_data":
                return collect_academic_data(self.ctx, parameters.get("queries", []), parameters.get("max_results", 100))
        
        elif agent == "trend_analyst":
            if action == "analyze_research_trends":
                return analyze_research_trends(self.ctx, parameters.get("topics"))
            elif action == "identify_blue_ocean_topics":
                return identify_blue_ocean_topics(self.ctx)
        
        elif agent == "cross_language":
            if action == "align_cross_language_concepts":
                return align_cross_language_concepts(self.ctx)
            elif action == "translate_and_map_concept":
                return translate_and_map_concept(self.ctx, parameters.get("concept"), parameters.get("target_lang", "zh"))
        
        elif agent == "visualizer":
            if action == "create_knowledge_graph":
                return create_knowledge_graph(self.ctx, parameters.get("topic"))
            elif action == "generate_research_report":
                return generate_research_report(self.ctx)
            elif action == "create_trend_visualization":
                return create_trend_visualization(self.ctx, parameters.get("days_back", 365))
        
        elif agent == "reflector":
            if action == "perform_system_reflection":
                return perform_system_reflection(self.ctx)
            elif action == "evaluate_model_performance":
                return evaluate_model_performance(self.ctx, parameters.get("days_back", 30))
            elif action == "analyze_system_bottlenecks":
                return analyze_system_bottlenecks(self.ctx)
        
        else:
            raise ValueError(f"未知的Agent或动作: {agent}.{action}")
    
    def _should_continue_on_error(self, step: Dict[str, Any], error: str) -> bool:
        """判断出错时是否继续执行"""
        # 数据采集失败不应该影响后续分析
        if step["agent"] == "data_collector":
            return True
        
        # 可视化失败不应该影响报告生成
        if step["agent"] == "visualizer" and "graph" in step["action"]:
            return True
        
        # 反思失败不应该影响主要功能
        if step["agent"] == "reflector":
            return True
        
        # 其他情况根据错误类型判断
        if "timeout" in error.lower() or "network" in error.lower():
            return True
        
        return False
    
    def _generate_execution_summary(self, execution_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """生成执行摘要"""
        completed_steps = [r for r in execution_results if r["status"] == "completed"]
        failed_steps = [r for r in execution_results if r["status"] == "failed"]
        
        # 提取关键结果
        key_findings = []
        papers_collected = 0
        trends_analyzed = 0
        blue_ocean_topics = 0
        visualizations_created = 0
        
        for result in completed_steps:
            if result["agent"] == "data_collector":
                # 解析采集的论文数量
                try:
                    data = json.loads(result["result"])
                    if data.get("status") == "success":
                        collection_stats = data.get("collection_stats", {})
                        papers_collected = collection_stats.get("total_saved", 0)
                        key_findings.append(f"成功采集 {papers_collected} 篇论文")
                except:
                    pass
            
            elif result["agent"] == "trend_analyst":
                if result["action"] == "analyze_research_trends":
                    try:
                        data = json.loads(result["result"])
                        if data.get("status") == "success":
                            trends_analyzed = data.get("topics_analyzed", 0)
                            key_findings.append(f"分析了 {trends_analyzed} 个研究主题")
                    except:
                        pass
                elif result["action"] == "identify_blue_ocean_topics":
                    try:
                        data = json.loads(result["result"])
                        if data.get("status") == "success":
                            blue_ocean_topics = len(data.get("blue_ocean_topics", []))
                            key_findings.append(f"识别出 {blue_ocean_topics} 个蓝海主题")
                    except:
                        pass
            
            elif result["agent"] == "visualizer":
                visualizations_created += 1
                if result["action"] == "create_knowledge_graph":
                    key_findings.append("生成了知识图谱")
                elif result["action"] == "generate_research_report":
                    key_findings.append("生成了研究报告")
        
        total_duration = sum(r.get("duration", 0) for r in execution_results)
        
        return {
            "total_duration": round(total_duration, 2),
            "success_rate": len(completed_steps) / len(execution_results) if execution_results else 0,
            "key_findings": key_findings,
            "statistics": {
                "papers_collected": papers_collected,
                "trends_analyzed": trends_analyzed,
                "blue_ocean_topics": blue_ocean_topics,
                "visualizations_created": visualizations_created
            },
            "recommendations": self._generate_recommendations(execution_results)
        }
    
    def _generate_recommendations(self, execution_results: List[Dict[str, Any]]) -> List[str]:
        """生成改进建议"""
        recommendations = []
        
        failed_steps = [r for r in execution_results if r["status"] == "failed"]
        if failed_steps:
            recommendations.append("检查失败步骤的错误原因并修复")
        
        # 检查执行时间
        total_duration = sum(r.get("duration", 0) for r in execution_results)
        if total_duration > 300:  # 5分钟
            recommendations.append("考虑优化执行时间或增加并行处理")
        
        # 检查数据采集情况
        data_collection_step = next((r for r in execution_results if r["agent"] == "data_collector"), None)
        if data_collection_step and data_collection_step["status"] == "completed":
            try:
                data = json.loads(data_collection_step["result"])
                collection_stats = data.get("collection_stats", {})
                if collection_stats.get("total_saved", 0) < 10:
                    recommendations.append("建议扩大搜索范围或调整查询条件")
            except:
                pass
        
        if not recommendations:
            recommendations.append("系统运行良好，建议定期监控和优化")
        
        return recommendations


def collect_academic_data_tool(queries: str, max_results: int = 100, ctx=None) -> str:
    """采集学术数据的工具函数"""
    return collect_academic_data(ctx, queries.split(',') if queries else [], max_results)

def analyze_research_trends_tool(topics: str = "", ctx=None) -> str:
    """分析研究趋势的工具函数"""
    return analyze_research_trends(ctx, topics.split(',') if topics else [])

def identify_blue_ocean_topics_tool(ctx=None) -> str:
    """识别蓝海研究主题的工具函数"""
    return identify_blue_ocean_topics(ctx)

def align_cross_language_concepts_tool(ctx=None) -> str:
    """对齐中英文概念的工具函数"""
    return align_cross_language_concepts(ctx)

def create_knowledge_graph_tool(topic: str = "", ctx=None) -> str:
    """创建知识图谱的工具函数"""
    return create_knowledge_graph(ctx, topic)

def generate_research_report_tool(ctx=None) -> str:
    """生成研究报告的工具函数"""
    return generate_research_report(ctx)

def perform_system_reflection_tool(ctx=None) -> str:
    """执行系统反思的工具函数"""
    return perform_system_reflection(ctx)

def search_academic_content_tool(topic: str, max_results: int = 10, ctx=None) -> str:
    """搜索学术内容的工具函数"""
    return search_academic_content(ctx, topic, max_results)

def search_chinese_content_tool(topic: str, max_results: int = 10, ctx=None) -> str:
    """搜索中文学术内容的工具函数"""
    return search_chinese_content(ctx, topic, max_results)


def build_agent(ctx=None):
    """构建学术侦探Agent"""
    workspace_path = os.getenv("COZE_WORKSPACE_PATH", "/workspace/projects")
    config_path = os.path.join(workspace_path, LLM_CONFIG)
    
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    
    api_key = os.getenv("COZE_WORKLOAD_IDENTITY_API_KEY")
    base_url = os.getenv("COZE_INTEGRATION_MODEL_BASE_URL")
    
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
    
    # 定义工具
    tools = [
        Tool(
            name="collect_academic_data",
            description="采集学术数据，支持从多个数据源获取论文信息。参数：queries（查询关键词列表，用逗号分隔），max_results（最大结果数）",
            func=collect_academic_data_tool
        ),
        Tool(
            name="analyze_research_trends",
            description="分析研究趋势和热点。参数：topics（主题列表，用逗号分隔）",
            func=analyze_research_trends_tool
        ),
        Tool(
            name="identify_blue_ocean_topics",
            description="识别蓝海研究主题和机会",
            func=identify_blue_ocean_topics_tool
        ),
        Tool(
            name="align_cross_language_concepts",
            description="对齐中英文概念",
            func=align_cross_language_concepts_tool
        ),
        Tool(
            name="create_knowledge_graph",
            description="创建知识图谱。参数：topic（主题名称）",
            func=create_knowledge_graph_tool
        ),
        Tool(
            name="generate_research_report",
            description="生成研究报告",
            func=generate_research_report_tool
        ),
        Tool(
            name="perform_system_reflection",
            description="执行系统反思和优化",
            func=perform_system_reflection_tool
        ),
        Tool(
            name="search_academic_content",
            description="搜索学术内容。参数：topic（搜索主题），max_results（最大结果数）",
            func=search_academic_content_tool
        ),
        Tool(
            name="search_chinese_content",
            description="搜索中文学术内容。参数：topic（搜索主题），max_results（最大结果数）",
            func=search_chinese_content_tool
        )
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
        orchestrator = AcademicDetectiveOrchestrator(ctx)
        
        # 规划工作流程
        workflow_steps = orchestrator.plan_workflow(user_request)
        
        # 执行工作流程
        execution_result = orchestrator.execute_workflow(workflow_steps)
        
        # 生成最终响应
        response = {
            "request": user_request,
            "workflow_steps": workflow_steps,
            "execution_result": execution_result,
            "message": "学术侦探系统处理完成"
        }
        
        return json.dumps(response, ensure_ascii=False, indent=2)
        
    except Exception as e:
        error_response = {
            "error": str(e),
            "message": "学术侦探系统处理失败",
            "request": user_request
        }
        return json.dumps(error_response, ensure_ascii=False, indent=2)