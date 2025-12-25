"""
可视化工具 - 用于生成知识图谱和研究报告
"""
import os
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict, Counter
import hashlib
from sqlalchemy.orm import Session
from coze_coding_utils.runtime_ctx.context import Context
from cozeloop.decorator import observe

from storage.database.db import get_session
from storage.database.shared.academic_schema import (
    Paper, ResearchTopic, TrendAnalysis, OpportunityScore, 
    ConceptMapping, VisualizationCache
)


class VisualizationExpert:
    """可视化专家"""
    
    def __init__(self, ctx: Context):
        self.ctx = ctx
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """加载配置"""
        workspace_path = os.getenv("COZE_WORKSPACE_PATH", "/workspace/projects")
        config_path = os.path.join(workspace_path, "config/academic_detective_config.json")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    @observe
    def generate_knowledge_graph(self, topic: str = None, max_nodes: int = 500) -> Dict[str, Any]:
        """生成知识图谱"""
        with get_session() as session:
            # 获取节点数据
            nodes = []
            edges = []
            
            if topic:
                # 基于特定主题生成图谱
                topic_data = self._generate_topic_graph(session, topic)
            else:
                # 生成整体知识图谱
                topic_data = self._generate_global_graph(session, max_nodes)
            
            nodes = topic_data["nodes"]
            edges = topic_data["edges"]
            
            # 生成图谱布局
            graph_data = {
                "nodes": nodes,
                "edges": edges,
                "layout": {
                    "type": "force_directed",
                    "iterations": 1000,
                    "node_spacing": 100,
                    "edge_length": 150
                },
                "metadata": {
                    "generated_date": datetime.now().isoformat(),
                    "total_nodes": len(nodes),
                    "total_edges": len(edges),
                    "topic_filter": topic
                }
            }
            
            # 缓存图谱数据
            self._cache_visualization_data(f"knowledge_graph_{topic or 'global'}", "knowledge_graph", graph_data)
            
            return graph_data
    
    def _generate_topic_graph(self, session: Session, topic: str) -> Dict[str, Any]:
        """生成特定主题的知识图谱"""
        nodes = []
        edges = []
        node_ids = set()
        
        # 查找主题
        db_topic = session.query(ResearchTopic).filter(
            (ResearchTopic.name_en.ilike(f"%{topic}%")) |
            (ResearchTopic.name_zh.ilike(f"%{topic}%"))
        ).first()
        
        if db_topic:
            # 添加主题节点
            topic_node = {
                "id": f"topic_{db_topic.id}",
                "label": db_topic.name_en or db_topic.name_zh,
                "type": "topic",
                "size": 30,
                "color": "#FF6B6B",
                "properties": {
                    "name_en": db_topic.name_en,
                    "name_zh": db_topic.name_zh,
                    "description": db_topic.description
                }
            }
            nodes.append(topic_node)
            node_ids.add(topic_node["id"])
            
            # 获取相关论文
            related_papers = session.query(Paper).filter(
                (Paper.title.contains(topic)) |
                (Paper.abstract.contains(topic)) |
                (Paper.keywords.contains(topic))
            ).limit(50).all()
            
            # 添加论文节点
            for paper in related_papers:
                paper_node = {
                    "id": f"paper_{paper.id}",
                    "label": paper.title[:50] + "..." if len(paper.title) > 50 else paper.title,
                    "type": "paper",
                    "size": 15,
                    "color": "#4ECDC4",
                    "properties": {
                        "authors": paper.authors,
                        "publish_date": paper.publish_date.isoformat() if paper.publish_date else None,
                        "citation_count": paper.citation_count,
                        "language": paper.language
                    }
                }
                nodes.append(paper_node)
                node_ids.add(paper_node["id"])
                
                # 添加边
                edge = {
                    "source": topic_node["id"],
                    "target": paper_node["id"],
                    "type": "topic_paper",
                    "weight": 2,
                    "color": "#95A5A6"
                }
                edges.append(edge)
        
        # 添加概念映射关系
        if db_topic:
            concept_mappings = session.query(ConceptMapping).filter(
                ConceptMapping.topic_id == db_topic.id
            ).all()
            
            for mapping in concept_mappings:
                # 创建概念节点
                concept_id = f"concept_{hashlib.md5(mapping.concept_en.encode()).hexdigest()[:8]}"
                if concept_id not in node_ids:
                    concept_node = {
                        "id": concept_id,
                        "label": f"{mapping.concept_en} / {mapping.concept_zh}",
                        "type": "concept",
                        "size": 20,
                        "color": "#FFD93D",
                        "properties": {
                            "concept_en": mapping.concept_en,
                            "concept_zh": mapping.concept_zh,
                            "similarity": mapping.similarity_score,
                            "confidence": mapping.confidence
                        }
                    }
                    nodes.append(concept_node)
                    node_ids.add(concept_id)
                
                # 添加主题-概念边
                edge = {
                    "source": f"topic_{db_topic.id}",
                    "target": concept_id,
                    "type": "topic_concept",
                    "weight": mapping.similarity_score * 5,
                    "color": "#E8B4F3"
                }
                edges.append(edge)
        
        return {"nodes": nodes, "edges": edges}
    
    def _generate_global_graph(self, session: Session, max_nodes: int) -> Dict[str, Any]:
        """生成全局知识图谱"""
        nodes = []
        edges = []
        node_ids = set()
        
        # 获取热门主题
        popular_topics = session.query(ResearchTopic).limit(30).all()
        
        # 添加主题节点
        for topic in popular_topics:
            if len(nodes) >= max_nodes:
                break
                
            topic_node = {
                "id": f"topic_{topic.id}",
                "label": topic.name_en or topic.name_zh,
                "type": "topic",
                "size": 25,
                "color": "#FF6B6B",
                "properties": {
                    "name_en": topic.name_en,
                    "name_zh": topic.name_zh,
                    "category": topic.category
                }
            }
            nodes.append(topic_node)
            node_ids.add(topic_node["id"])
        
        # 添加主题间关系（基于概念映射）
        concept_mappings = session.query(ConceptMapping).all()
        topic_connections = defaultdict(set)
        
        for mapping in concept_mappings:
            topic_connections[mapping.topic_id].add(mapping.topic_id)
        
        # 创建主题间边
        for i, topic1 in enumerate(popular_topics):
            for topic2 in popular_topics[i+1:]:
                if f"topic_{topic2.id}" in node_ids and f"topic_{topic1.id}" in node_ids:
                    # 简单的关系检测（可以进一步优化）
                    shared_concepts = session.query(ConceptMapping).filter(
                        (ConceptMapping.topic_id == topic1.id) |
                        (ConceptMapping.topic_id == topic2.id)
                    ).count()
                    
                    if shared_concepts > 0:
                        edge = {
                            "source": f"topic_{topic1.id}",
                            "target": f"topic_{topic2.id}",
                            "type": "topic_topic",
                            "weight": shared_concepts,
                            "color": "#95A5A6"
                        }
                        edges.append(edge)
        
        return {"nodes": nodes, "edges": edges}
    
    @observe
    def generate_trend_visualization(self, days_back: int = 365) -> Dict[str, Any]:
        """生成趋势可视化数据"""
        with get_session() as session:
            # 获取趋势数据
            start_date = datetime.now() - timedelta(days=days_back)
            
            # 获取热门主题的趋势
            trend_data = []
            topics = session.query(ResearchTopic).limit(15).all()
            
            for topic in topics:
                # 获取该主题的趋势分析
                latest_trend = session.query(TrendAnalysis).filter(
                    TrendAnalysis.topic_id == topic.id
                ).order_by(TrendAnalysis.analysis_date.desc()).first()
                
                if latest_trend:
                    trend_info = {
                        "topic_name": topic.name_en or topic.name_zh,
                        "paper_count": latest_trend.paper_count,
                        "growth_rate": latest_trend.growth_rate,
                        "hotness_score": latest_trend.hotness_score,
                        "trend_direction": latest_trend.trend_direction,
                        "predicted_growth": latest_trend.predicted_growth,
                        "opportunity_index": self._get_latest_opportunity_index(session, topic.id)
                    }
                    trend_data.append(trend_info)
            
            # 生成可视化数据
            viz_data = {
                "chart_type": "trend_analysis",
                "data": trend_data,
                "time_range": f"过去{days_back}天",
                "generated_date": datetime.now().isoformat(),
                "visualization_options": {
                    "chart_types": ["bar", "line", "scatter", "bubble"],
                    "metrics": ["paper_count", "growth_rate", "hotness_score", "opportunity_index"]
                }
            }
            
            # 缓存数据
            self._cache_visualization_data("trend_analysis", "chart", viz_data)
            
            return viz_data
    
    def _get_latest_opportunity_index(self, session: Session, topic_id: int) -> Optional[float]:
        """获取最新的机会指数"""
        latest_opportunity = session.query(OpportunityScore).filter(
            OpportunityScore.topic_id == topic_id
        ).order_by(OpportunityScore.score_date.desc()).first()
        
        return latest_opportunity.opportunity_index if latest_opportunity else None
    
    @observe
    def generate_opportunity_report(self, top_k: int = 10) -> Dict[str, Any]:
        """生成研究机会报告"""
        with get_session() as session:
            # 获取蓝海主题
            blue_ocean_topics = []
            
            # 查询高机会指数的主题
            opportunity_scores = session.query(
                OpportunityScore, ResearchTopic
            ).join(ResearchTopic).filter(
                OpportunityScore.opportunity_index > 0.6
            ).order_by(OpportunityScore.opportunity_index.desc()).limit(top_k).all()
            
            for score, topic in opportunity_scores:
                topic_info = {
                    "topic_name": topic.name_en or topic.name_zh,
                    "opportunity_index": score.opportunity_index,
                    "competition_level": score.competition_level,
                    "innovation_potential": score.innovation_potential,
                    "market_size": score.market_size,
                    "difficulty_level": score.difficulty_level,
                    "overall_score": score.overall_score,
                    "recommendation": self._generate_recommendation(score)
                }
                blue_ocean_topics.append(topic_info)
            
            # 生成报告
            report = {
                "report_type": "opportunity_analysis",
                "generation_date": datetime.now().isoformat(),
                "summary": {
                    "total_topics_analyzed": len(blue_ocean_topics),
                    "high_opportunity_count": len([t for t in blue_ocean_topics if t["opportunity_index"] > 0.8]),
                    "average_opportunity_index": sum(t["opportunity_index"] for t in blue_ocean_topics) / len(blue_ocean_topics) if blue_ocean_topics else 0
                },
                "blue_ocean_topics": blue_ocean_topics,
                "insights": self._generate_insights(blue_ocean_topics),
                "recommendations": self._generate_strategic_recommendations(blue_ocean_topics)
            }
            
            # 缓存报告
            self._cache_visualization_data("opportunity_report", "report", report)
            
            return report
    
    def _generate_recommendation(self, score: OpportunityScore) -> str:
        """生成主题建议"""
        if score.opportunity_index > 0.8:
            return "强烈推荐：这是高价值的蓝海研究领域"
        elif score.opportunity_index > 0.7:
            return "推荐：有较好的研究机会和发展前景"
        elif score.opportunity_index > 0.6:
            return "可以考虑：有一定潜力但需要谨慎评估"
        else:
            return "评估：需要进一步分析确定可行性"
    
    def _generate_insights(self, topics: List[Dict[str, Any]]) -> List[str]:
        """生成洞察"""
        insights = []
        
        if topics:
            # 分析竞争程度分布
            low_competition = len([t for t in topics if t["competition_level"] < 0.5])
            if low_competition > 0:
                insights.append(f"发现 {low_competition} 个低竞争度的高价值主题")
            
            # 分析创新潜力
            high_innovation = len([t for t in topics if t["innovation_potential"] > 0.8])
            if high_innovation > 0:
                insights.append(f"识别出 {high_innovation} 个高创新潜力的研究方向")
            
            # 分析难度等级
            manageable_difficulty = len([t for t in topics if t["difficulty_level"] < 0.6])
            if manageable_difficulty > 0:
                insights.append(f"有 {manageable_difficulty} 个主题难度适中，适合投入研究")
        
        return insights
    
    def _generate_strategic_recommendations(self, topics: List[Dict[str, Any]]) -> List[str]:
        """生成战略建议"""
        recommendations = []
        
        if not topics:
            return ["当前数据不足，建议扩大数据采集范围"]
        
        # 基于主题特征生成建议
        avg_opportunity = sum(t["opportunity_index"] for t in topics) / len(topics)
        
        if avg_opportunity > 0.75:
            recommendations.append("当前存在较多高质量研究机会，建议优先投入资源")
        elif avg_opportunity > 0.65:
            recommendations.append("研究机会较好，建议选择性投入重点方向")
        else:
            recommendations.append("高质量机会有限，建议加强基础研究或拓展研究领域")
        
        # 竞争分析
        low_competition_topics = [t for t in topics if t["competition_level"] < 0.5]
        if low_competition_topics:
            recommendations.append(f"重点关注低竞争主题：{', '.join([t['topic_name'] for t in low_competition_topics[:3]])}")
        
        # 创新潜力分析
        high_innovation_topics = [t for t in topics if t["innovation_potential"] > 0.8]
        if high_innovation_topics:
            recommendations.append(f"重点投入创新潜力高的主题：{', '.join([t['topic_name'] for t in high_innovation_topics[:3]])}")
        
        return recommendations
    
    def _cache_visualization_data(self, cache_key: str, viz_type: str, data: Dict[str, Any]):
        """缓存可视化数据"""
        with get_session() as session:
            # 检查是否已存在
            existing_cache = session.query(VisualizationCache).filter(
                VisualizationCache.cache_key == cache_key
            ).first()
            
            expiry_date = datetime.now() + timedelta(hours=24)  # 24小时过期
            
            if existing_cache:
                existing_cache.data = data
                existing_cache.created_date = datetime.now()
                existing_cache.expiry_date = expiry_date
                existing_cache.is_valid = True
            else:
                cache_entry = VisualizationCache(
                    cache_key=cache_key,
                    viz_type=viz_type,
                    data=data,
                    expiry_date=expiry_date
                )
                session.add(cache_entry)
            
            session.commit()
    
    @observe
    def get_cached_visualization(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """获取缓存的可视化数据"""
        with get_session() as session:
            cache_entry = session.query(VisualizationCache).filter(
                VisualizationCache.cache_key == cache_key,
                VisualizationCache.is_valid == True,
                VisualizationCache.expiry_date > datetime.now()
            ).first()
            
            if cache_entry:
                return cache_entry.data
            
            return None


# 工具函数，供Agent使用
@observe
def create_knowledge_graph(ctx: Context, topic: str = None) -> str:
    """创建知识图谱的主要接口"""
    try:
        expert = VisualizationExpert(ctx)
        
        # 检查缓存
        cache_key = f"knowledge_graph_{topic or 'global'}"
        cached_data = expert.get_cached_visualization(cache_key)
        
        if cached_data:
            output = {
                "status": "success",
                "source": "cache",
                "data": cached_data,
                "message": "从缓存获取知识图谱数据"
            }
        else:
            # 生成新的图谱
            graph_data = expert.generate_knowledge_graph(topic)
            
            output = {
                "status": "success",
                "source": "generated",
                "data": graph_data,
                "message": f"成功生成{topic if topic else '全局'}知识图谱，包含{len(graph_data['nodes'])}个节点"
            }
        
        return json.dumps(output, ensure_ascii=False, indent=2)
        
    except Exception as e:
        error_output = {
            "status": "error",
            "error": str(e),
            "message": "知识图谱生成失败"
        }
        return json.dumps(error_output, ensure_ascii=False, indent=2)


@observe
def generate_research_report(ctx: Context) -> str:
    """生成研究机会报告的主要接口"""
    try:
        expert = VisualizationExpert(ctx)
        
        # 检查缓存
        cached_report = expert.get_cached_visualization("opportunity_report")
        
        if cached_report:
            output = {
                "status": "success",
                "source": "cache",
                "data": cached_report,
                "message": "从缓存获取研究报告"
            }
        else:
            # 生成新报告
            report_data = expert.generate_opportunity_report()
            
            output = {
                "status": "success",
                "source": "generated",
                "data": report_data,
                "message": f"成功生成研究机会报告，识别{len(report_data['blue_ocean_topics'])}个蓝海主题"
            }
        
        return json.dumps(output, ensure_ascii=False, indent=2)
        
    except Exception as e:
        error_output = {
            "status": "error",
            "error": str(e),
            "message": "研究报告生成失败"
        }
        return json.dumps(error_output, ensure_ascii=False, indent=2)


@observe
def create_trend_visualization(ctx: Context, days_back: int = 365) -> str:
    """创建趋势可视化的主要接口"""
    try:
        expert = VisualizationExpert(ctx)
        
        # 生成趋势数据
        trend_data = expert.generate_trend_visualization(days_back)
        
        output = {
            "status": "success",
            "data": trend_data,
            "message": f"成功生成过去{days_back}天的趋势可视化数据"
        }
        
        return json.dumps(output, ensure_ascii=False, indent=2)
        
    except Exception as e:
        error_output = {
            "status": "error",
            "error": str(e),
            "message": "趋势可视化生成失败"
        }
        return json.dumps(error_output, ensure_ascii=False, indent=2)