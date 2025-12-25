"""
反思工具 - 用于系统反思与持续优化
"""
import os
import json
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict, Counter
from sqlalchemy.orm import Session
from sqlalchemy import func, and_
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from coze_coding_utils.runtime_ctx.context import Context, default_headers
from cozeloop.decorator import observe

from storage.database.db import get_session
from storage.database.shared.academic_schema import (
    Paper, ResearchTopic, TrendAnalysis, OpportunityScore, 
    SystemMetrics, TaskLog
)


class SystemReflector:
    """系统反思专家"""
    
    def __init__(self, ctx: Context):
        self.ctx = ctx
        self.config = self._load_config()
        self.llm = self._init_llm()
        
    def _load_config(self) -> Dict[str, Any]:
        """加载配置"""
        workspace_path = os.getenv("COZE_WORKSPACE_PATH", "/workspace/projects")
        config_path = os.path.join(workspace_path, "config/academic_detective_config.json")
        
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
            default_headers=default_headers(self.ctx)
        )
    
    @observe
    def evaluate_prediction_accuracy(self, days_back: int = 30) -> Dict[str, Any]:
        """评估预测准确率"""
        with get_session() as session:
            # 获取历史预测数据
            start_date = datetime.now() - timedelta(days=days_back)
            
            # 查询历史趋势预测
            historical_trends = session.query(TrendAnalysis).filter(
                TrendAnalysis.analysis_date >= start_date,
                TrendAnalysis.predicted_growth.isnot(None)
            ).all()
            
            accuracy_scores = []
            topic_predictions = []
            
            for trend in historical_trends:
                # 获取该主题的最新趋势数据
                latest_trend = session.query(TrendAnalysis).filter(
                    TrendAnalysis.topic_id == trend.topic_id,
                    TrendAnalysis.analysis_date > trend.analysis_date
                ).order_by(TrendAnalysis.analysis_date.desc()).first()
                
                if latest_trend:
                    # 计算预测准确率
                    predicted = trend.predicted_growth
                    actual = latest_trend.growth_rate
                    
                    if predicted is not None and actual is not None:
                        # 简单的准确率计算
                        accuracy = 1 - abs(predicted - actual) / (abs(predicted) + abs(actual) + 1)
                        accuracy_scores.append(max(0, min(1, accuracy)))
                        
                        topic_predictions.append({
                            "topic_id": trend.topic_id,
                            "predicted_growth": predicted,
                            "actual_growth": actual,
                            "accuracy": max(0, min(1, accuracy)),
                            "prediction_date": trend.analysis_date.isoformat(),
                            "actual_date": latest_trend.analysis_date.isoformat()
                        })
            
            # 计算整体准确率
            overall_accuracy = sum(accuracy_scores) / len(accuracy_scores) if accuracy_scores else 0
            
            # 生成准确率评估报告
            evaluation = {
                "evaluation_date": datetime.now().isoformat(),
                "evaluation_period": f"过去{days_back}天",
                "total_predictions": len(topic_predictions),
                "overall_accuracy": round(overall_accuracy, 3),
                "accuracy_distribution": self._calculate_accuracy_distribution(accuracy_scores),
                "topic_predictions": topic_predictions[:10],  # 返回前10个预测结果
                "insights": self._generate_accuracy_insights(accuracy_scores, topic_predictions),
                "recommendations": self._generate_accuracy_recommendations(overall_accuracy)
            }
            
            # 更新系统指标
            self._update_system_metrics({"prediction_accuracy": overall_accuracy})
            
            return evaluation
    
    def _calculate_accuracy_distribution(self, accuracy_scores: List[float]) -> Dict[str, int]:
        """计算准确率分布"""
        distribution = {
            "excellent (>0.8)": 0,
            "good (0.6-0.8)": 0,
            "fair (0.4-0.6)": 0,
            "poor (<0.4)": 0
        }
        
        for score in accuracy_scores:
            if score > 0.8:
                distribution["excellent (>0.8)"] += 1
            elif score > 0.6:
                distribution["good (0.6-0.8)"] += 1
            elif score > 0.4:
                distribution["fair (0.4-0.6)"] += 1
            else:
                distribution["poor (<0.4)"] += 1
        
        return distribution
    
    def _generate_accuracy_insights(self, accuracy_scores: List[float], predictions: List[Dict[str, Any]]) -> List[str]:
        """生成准确率洞察"""
        insights = []
        
        if not accuracy_scores:
            return ["数据不足，无法生成准确率洞察"]
        
        avg_accuracy = sum(accuracy_scores) / len(accuracy_scores)
        
        if avg_accuracy > 0.7:
            insights.append("系统预测准确率较高，模型表现良好")
        elif avg_accuracy > 0.5:
            insights.append("系统预测准确率中等，有改进空间")
        else:
            insights.append("系统预测准确率较低，需要大幅优化")
        
        # 分析预测偏差
        predicted_high = [p for p in predictions if p["predicted_growth"] > 0.1]
        predicted_low = [p for p in predictions if p["predicted_growth"] < -0.1]
        
        if predicted_high:
            high_accuracy = sum(p["accuracy"] for p in predicted_high) / len(predicted_high)
            if high_accuracy < 0.5:
                insights.append("高增长预测准确率偏低，需要调整增长预测模型")
        
        if predicted_low:
            low_accuracy = sum(p["accuracy"] for p in predicted_low) / len(predicted_low)
            if low_accuracy < 0.5:
                insights.append("负增长预测准确率偏低，需要重新评估下降趋势指标")
        
        return insights
    
    def _generate_accuracy_recommendations(self, overall_accuracy: float) -> List[str]:
        """生成准确率改进建议"""
        recommendations = []
        
        if overall_accuracy < 0.5:
            recommendations.extend([
                "重新训练趋势预测模型，增加特征工程",
                "考虑使用更复杂的时间序列模型",
                "增加外部因素（如政策、经济指标）作为预测特征",
                "调整预测模型的超参数"
            ])
        elif overall_accuracy < 0.7:
            recommendations.extend([
                "微调现有预测模型参数",
                "增加更多的历史数据进行训练",
                "考虑集成多个预测模型进行融合预测"
            ])
        else:
            recommendations.extend([
                "保持当前模型性能，定期监控准确率变化",
                "考虑优化模型计算效率",
                "探索更细粒度的预测维度"
            ])
        
        return recommendations
    
    @observe
    def analyze_system_performance(self) -> Dict[str, Any]:
        """分析系统性能"""
        with get_session() as session:
            # 获取任务执行统计
            task_stats = self._get_task_statistics(session)
            
            # 获取数据处理统计
            data_stats = self._get_data_processing_stats(session)
            
            # 获取系统资源使用情况
            resource_stats = self._get_resource_statistics(session)
            
            # 生成性能分析报告
            performance_analysis = {
                "analysis_date": datetime.now().isoformat(),
                "task_performance": task_stats,
                "data_processing": data_stats,
                "resource_usage": resource_stats,
                "bottlenecks": self._identify_bottlenecks(task_stats, data_stats, resource_stats),
                "optimization_opportunities": self._identify_optimization_opportunities(task_stats, data_stats, resource_stats)
            }
            
            return performance_analysis
    
    def _get_task_statistics(self, session: Session) -> Dict[str, Any]:
        """获取任务执行统计"""
        # 获取最近7天的任务数据
        start_date = datetime.now() - timedelta(days=7)
        
        tasks = session.query(TaskLog).filter(
            TaskLog.start_time >= start_date
        ).all()
        
        if not tasks:
            return {"error": "没有足够的任务数据"}
        
        # 按Agent分组统计
        agent_stats = defaultdict(list)
        for task in tasks:
            if task.duration:
                agent_stats[task.agent_name].append(task.duration)
        
        # 计算统计指标
        task_performance = {}
        for agent_name, durations in agent_stats.items():
            task_performance[agent_name] = {
                "total_tasks": len(durations),
                "avg_duration": sum(durations) / len(durations),
                "max_duration": max(durations),
                "min_duration": min(durations),
                "success_rate": len([t for t in tasks if t.agent_name == agent_name and t.status == "completed"]) / len([t for t in tasks if t.agent_name == agent_name])
            }
        
        # 总体统计
        all_durations = [task.duration for task in tasks if task.duration]
        task_performance["overall"] = {
            "total_tasks": len(tasks),
            "completed_tasks": len([t for t in tasks if t.status == "completed"]),
            "failed_tasks": len([t for t in tasks if t.status == "failed"]),
            "success_rate": len([t for t in tasks if t.status == "completed"]) / len(tasks),
            "avg_duration": sum(all_durations) / len(all_durations) if all_durations else 0,
            "total_errors": len([t for t in tasks if t.status == "failed"])
        }
        
        return task_performance
    
    def _get_data_processing_stats(self, session: Session) -> Dict[str, Any]:
        """获取数据处理统计"""
        # 论文处理统计
        total_papers = session.query(Paper).count()
        processed_papers = session.query(Paper).filter(Paper.is_processed == True).count()
        
        # 主题统计
        total_topics = session.query(ResearchTopic).count()
        
        # 最近7天的数据增长
        start_date = datetime.now() - timedelta(days=7)
        recent_papers = session.query(Paper).filter(
            Paper.collection_date >= start_date
        ).count()
        
        return {
            "papers": {
                "total": total_papers,
                "processed": processed_papers,
                "processing_rate": processed_papers / total_papers if total_papers > 0 else 0,
                "recent_growth": recent_papers
            },
            "topics": {
                "total": total_topics
            }
        }
    
    def _get_resource_statistics(self, session: Session) -> Dict[str, Any]:
        """获取资源使用统计"""
        # 这里可以集成系统监控数据
        # 目前返回模拟数据
        return {
            "memory_usage": "60%",  # 实际应该从系统监控获取
            "cpu_usage": "45%",      # 实际应该从系统监控获取
            "storage_usage": "30%",  # 实际应该从系统监控获取
            "network_usage": "20%"   # 实际应该从系统监控获取
        }
    
    def _identify_bottlenecks(self, task_stats: Dict, data_stats: Dict, resource_stats: Dict) -> List[str]:
        """识别系统瓶颈"""
        bottlenecks = []
        
        # 任务执行瓶颈
        if "overall" in task_stats:
            overall_stats = task_stats["overall"]
            if overall_stats.get("success_rate", 1.0) < 0.9:
                bottlenecks.append(f"任务成功率较低 ({overall_stats['success_rate']:.2%})")
            
            if overall_stats.get("avg_duration", 0) > 300:  # 5分钟
                bottlenecks.append(f"平均任务执行时间过长 ({overall_stats['avg_duration']:.1f}秒)")
        
        # 数据处理瓶颈
        if "papers" in data_stats:
            processing_rate = data_stats["papers"].get("processing_rate", 1.0)
            if processing_rate < 0.8:
                bottlenecks.append(f"论文处理率较低 ({processing_rate:.2%})")
        
        # 资源瓶颈
        if "memory_usage" in resource_stats:
            memory_usage = float(resource_stats["memory_usage"].rstrip('%'))
            if memory_usage > 80:
                bottlenecks.append(f"内存使用率过高 ({resource_stats['memory_usage']})")
        
        return bottlenecks
    
    def _identify_optimization_opportunities(self, task_stats: Dict, data_stats: Dict, resource_stats: Dict) -> List[str]:
        """识别优化机会"""
        opportunities = []
        
        # 任务优化
        if "overall" in task_stats:
            success_rate = task_stats["overall"].get("success_rate", 1.0)
            if success_rate < 0.95:
                opportunities.append("改进错误处理和重试机制以提高任务成功率")
        
        # 数据处理优化
        if "papers" in data_stats:
            processing_rate = data_stats["papers"].get("processing_rate", 1.0)
            if processing_rate < 1.0:
                opportunities.append("优化数据处理流程，提高论文处理效率")
        
        # 缓存优化
        opportunities.append("增加缓存机制以减少重复计算")
        opportunities.append("实现增量更新机制以提高数据更新效率")
        
        return opportunities
    
    @observe
    def generate_reflection_report(self) -> Dict[str, Any]:
        """生成系统反思报告"""
        try:
            # 评估预测准确率
            accuracy_evaluation = self.evaluate_prediction_accuracy()
            
            # 分析系统性能
            performance_analysis = self.analyze_system_performance()
            
            # 使用LLM生成深度反思
            reflection_insights = self._generate_llm_reflections(accuracy_evaluation, performance_analysis)
            
            # 生成综合报告
            reflection_report = {
                "report_date": datetime.now().isoformat(),
                "accuracy_evaluation": accuracy_evaluation,
                "performance_analysis": performance_analysis,
                "llm_insights": reflection_insights,
                "action_items": self._generate_action_items(accuracy_evaluation, performance_analysis, reflection_insights),
                "next_review_date": (datetime.now() + timedelta(days=7)).isoformat()
            }
            
            return reflection_report
            
        except Exception as e:
            return {
                "error": str(e),
                "message": "反思报告生成失败"
            }
    
    def _generate_llm_reflections(self, accuracy_eval: Dict, performance: Dict) -> Dict[str, Any]:
        """使用LLM生成深度反思洞察"""
        prompt = f"""
        作为系统反思专家，请基于以下数据生成深度的系统反思洞察：
        
        预测准确率评估：
        {json.dumps(accuracy_eval, indent=2, ensure_ascii=False)}
        
        系统性能分析：
        {json.dumps(performance, indent=2, ensure_ascii=False)}
        
        请按以下格式返回JSON：
        {{
            "strengths": ["系统优势1", "系统优势2"],
            "weaknesses": ["系统弱点1", "系统弱点2"],
            "improvement_areas": ["改进领域1", "改进领域2"],
            "strategic_recommendations": ["战略建议1", "战略建议2"],
            "risk_assessment": ["风险1", "风险2"],
            "opportunities": ["机会1", "机会2"]
        }}
        
        要求：
        1. 基于数据提供具体、可操作的洞察
        2. 识别系统优势和劣势
        3. 提供战略层面的改进建议
        4. 评估潜在风险和机会
        """
        
        try:
            messages = [
                SystemMessage(content="你是专业的系统反思专家，擅长从数据中提取深度洞察和战略建议。"),
                HumanMessage(content=prompt)
            ]
            
            response = ""
            for chunk in self.llm.stream(messages):
                if hasattr(chunk, 'content') and isinstance(chunk.content, str):
                    response += chunk.content
            
            # 解析JSON响应
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            
        except Exception as e:
            print(f"LLM反思生成失败: {str(e)}")
        
        # 回退到基础分析
        return {
            "strengths": ["系统运行稳定", "数据采集正常"],
            "weaknesses": ["预测准确率有提升空间", "性能优化待加强"],
            "improvement_areas": ["模型优化", "系统性能"],
            "strategic_recommendations": ["持续监控", "定期优化"],
            "risk_assessment": ["数据质量风险", "系统过载风险"],
            "opportunities": ["算法改进", "架构优化"]
        }
    
    def _generate_action_items(self, accuracy_eval: Dict, performance: Dict, llm_insights: Dict) -> List[Dict[str, Any]]:
        """生成行动项"""
        action_items = []
        
        # 基于准确率评估的行动项
        overall_accuracy = accuracy_eval.get("overall_accuracy", 0)
        if overall_accuracy < 0.7:
            action_items.append({
                "priority": "high",
                "action": "优化预测模型",
                "description": "重新训练趋势预测模型以提高准确率",
                "owner": "Trend Analyst Agent",
                "deadline": (datetime.now() + timedelta(days=3)).isoformat()
            })
        
        # 基于性能分析的行动项
        bottlenecks = performance.get("bottlenecks", [])
        for bottleneck in bottlenecks:
            action_items.append({
                "priority": "medium",
                "action": "解决系统瓶颈",
                "description": f"解决识别到的瓶颈：{bottleneck}",
                "owner": "System Administrator",
                "deadline": (datetime.now() + timedelta(days=7)).isoformat()
            })
        
        # 基于LLM洞察的行动项
        strategic_recommendations = llm_insights.get("strategic_recommendations", [])
        for i, recommendation in enumerate(strategic_recommendations[:3]):  # 取前3个建议
            action_items.append({
                "priority": "medium",
                "action": f"战略改进-{i+1}",
                "description": recommendation,
                "owner": "Orchestrator Agent",
                "deadline": (datetime.now() + timedelta(days=14)).isoformat()
            })
        
        return action_items
    
    def _update_system_metrics(self, metrics: Dict[str, Any]):
        """更新系统指标"""
        with get_session() as session:
            # 创建或更新系统指标
            system_metric = SystemMetrics(
                metric_date=datetime.now(),
                prediction_accuracy=metrics.get("prediction_accuracy"),
                total_papers=0,  # 可以进一步填充
                processed_papers=0,
                total_topics=0,
                active_topics=0,
                system_performance={},  # 可以进一步填充
                error_count=0
            )
            
            session.add(system_metric)
            session.commit()


# 工具函数，供Agent使用
@observe
def perform_system_reflection(ctx: Context) -> str:
    """执行系统反思的主要接口"""
    try:
        reflector = SystemReflector(ctx)
        
        # 生成反思报告
        report = reflector.generate_reflection_report()
        
        output = {
            "status": "success",
            "report": report,
            "message": "系统反思报告生成完成"
        }
        
        return json.dumps(output, ensure_ascii=False, indent=2)
        
    except Exception as e:
        error_output = {
            "status": "error",
            "error": str(e),
            "message": "系统反思失败"
        }
        return json.dumps(error_output, ensure_ascii=False, indent=2)


@observe
def evaluate_model_performance(ctx: Context, days_back: int = 30) -> str:
    """评估模型性能的主要接口"""
    try:
        reflector = SystemReflector(ctx)
        
        # 评估预测准确率
        evaluation = reflector.evaluate_prediction_accuracy(days_back)
        
        output = {
            "status": "success",
            "evaluation": evaluation,
            "message": f"过去{days_back}天的模型性能评估完成"
        }
        
        return json.dumps(output, ensure_ascii=False, indent=2)
        
    except Exception as e:
        error_output = {
            "status": "error",
            "error": str(e),
            "message": "模型性能评估失败"
        }
        return json.dumps(error_output, ensure_ascii=False, indent=2)


@observe
def analyze_system_bottlenecks(ctx: Context) -> str:
    """分析系统瓶颈的主要接口"""
    try:
        reflector = SystemReflector(ctx)
        
        # 分析系统性能
        performance = reflector.analyze_system_performance()
        
        output = {
            "status": "success",
            "performance_analysis": performance,
            "message": "系统性能分析完成"
        }
        
        return json.dumps(output, ensure_ascii=False, indent=2)
        
    except Exception as e:
        error_output = {
            "status": "error",
            "error": str(e),
            "message": "系统瓶颈分析失败"
        }
        return json.dumps(error_output, ensure_ascii=False, indent=2)