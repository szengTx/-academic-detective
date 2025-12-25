"""
趋势分析工具 - 用于识别研究主题和评估机会指数
"""
import os
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict, Counter
import re
from sqlalchemy.orm import Session
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from coze_coding_utils.runtime_ctx.context import Context, default_headers
from cozeloop.decorator import observe

from storage.database.db import get_session
from storage.database.shared.academic_schema import Paper, ResearchTopic, TrendAnalysis, OpportunityScore


class TrendAnalyst:
    """趋势分析师"""
    
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
    def extract_research_topics(self, papers: List[Paper], min_papers: int = 5) -> List[str]:
        """从论文中提取研究主题"""
        # 合并所有标题和摘要
        all_text = ""
        for paper in papers:
            all_text += f"Title: {paper.title}\n"
            all_text += f"Abstract: {paper.abstract or ''}\n"
            all_text += f"Keywords: {paper.keywords or ''}\n\n"
        
        # 使用LLM提取主题
        prompt = f"""
        作为学术趋势分析专家，请从以下学术文本中提取主要的研究主题和关键词。
        
        文本内容：
        {all_text[:8000]}  # 限制文本长度
        
        请按以下格式返回JSON：
        {{
            "topics": [
                {{
                    "name": "主题名称（英文）",
                    "name_zh": "主题名称（中文）", 
                    "description": "主题描述",
                    "keywords": ["关键词1", "关键词2", ...],
                    "relevance_score": 0.9
                }}
            ]
        }}
        
        要求：
        1. 识别出5-10个主要研究主题
        2. 包含中英文对应关系
        3. 评估每个主题的相关性
        4. 确保主题具有学术价值和前沿性
        """
        
        try:
            messages = [
                SystemMessage(content="你是专业的学术趋势分析专家，擅长从学术文献中提取研究主题和趋势。"),
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
                result = json.loads(json_match.group())
                return [topic['name'] for topic in result.get('topics', [])]
            
        except Exception as e:
            print(f"LLM提取主题失败: {str(e)}")
        
        # 回退到简单的关键词提取
        return self._simple_topic_extraction(papers, min_papers)
    
    def _simple_topic_extraction(self, papers: List[Paper], min_papers: int) -> List[str]:
        """简单的主题提取方法"""
        # 统计关键词频率
        keyword_counter = Counter()
        title_words = Counter()
        
        for paper in papers:
            # 处理关键词
            if paper.keywords:
                keywords = [k.strip() for k in paper.keywords.split(',') if k.strip()]
                keyword_counter.update(keywords)
            
            # 处理标题中的词汇
            title_words.update(self._extract_title_words(paper.title))
        
        # 合并并排序
        combined_counts = defaultdict(int)
        for word, count in keyword_counter.items():
            if len(word) > 2:  # 过滤短词
                combined_counts[word.lower()] += count * 2  # 关键词权重更高
        
        for word, count in title_words.items():
            if count >= min_papers:  # 至少出现在min_papers篇论文中
                combined_counts[word] += count
        
        # 返回最常见的主题
        topics = [topic for topic, count in sorted(combined_counts.items(), key=lambda x: x[1], reverse=True)[:20]]
        return topics[:10]  # 返回前10个主题
    
    def _extract_title_words(self, title: str) -> List[str]:
        """从标题中提取重要词汇"""
        # 简单的词汇提取，可以进一步优化
        words = re.findall(r'\b[A-Za-z]{4,}\b', title)
        # 过滤常见词汇
        stop_words = {'that', 'with', 'from', 'this', 'they', 'have', 'been', 'than', 'will', 'would', 'could', 'should'}
        return [word.lower() for word in words if word.lower() not in stop_words]
    
    @observe
    def analyze_trend(self, topic_name: str, days_back: int = 365) -> Dict[str, Any]:
        """分析特定主题的趋势"""
        with get_session() as session:
            # 获取相关论文
            start_date = datetime.now() - timedelta(days=days_back)
            papers = session.query(Paper).filter(
                Paper.publish_date >= start_date,
                (Paper.title.contains(topic_name)) |
                (Paper.abstract.contains(topic_name)) |
                (Paper.keywords.contains(topic_name))
            ).all()
            
            if len(papers) < 5:
                return {"error": f"主题 '{topic_name}' 相关论文数量不足"}
            
            # 按月统计论文数量
            monthly_counts = defaultdict(int)
            monthly_citations = defaultdict(int)
            
            for paper in papers:
                month_key = paper.publish_date.strftime('%Y-%m') if paper.publish_date else 'unknown'
                monthly_counts[month_key] += 1
                monthly_citations[month_key] += paper.citation_count or 0
            
            # 计算趋势指标
            months = sorted(monthly_counts.keys())
            if len(months) >= 2:
                recent_growth = (monthly_counts[months[-1]] - monthly_counts[months[-2]]) / max(1, monthly_counts[months[-2]])
            else:
                recent_growth = 0
            
            avg_citation = sum(paper.citation_count or 0 for paper in papers) / len(papers)
            
            # 计算热度分数
            hotness_score = self._calculate_hotness_score(len(papers), recent_growth, avg_citation)
            
            # 趋势方向
            if recent_growth > 0.1:
                trend_direction = "up"
            elif recent_growth < -0.1:
                trend_direction = "down"
            else:
                trend_direction = "stable"
            
            # 预测增长率（简单模型）
            predicted_growth = recent_growth * 1.2  # 简单的预测模型
            
            return {
                "topic": topic_name,
                "total_papers": len(papers),
                "monthly_counts": dict(monthly_counts),
                "growth_rate": recent_growth,
                "average_citation": avg_citation,
                "hotness_score": hotness_score,
                "trend_direction": trend_direction,
                "predicted_growth": predicted_growth,
                "confidence": 0.8  # 置信度
            }
    
    def _calculate_hotness_score(self, paper_count: int, growth_rate: float, avg_citation: float) -> float:
        """计算热度分数"""
        # 标准化各个指标
        normalized_papers = min(paper_count / 100, 1.0)  # 论文数量（标准化到0-1）
        normalized_growth = min(max(growth_rate, -1), 1)  # 增长率（标准化到-1到1）
        normalized_citations = min(avg_citation / 50, 1.0)  # 平均引用数（标准化到0-1）
        
        # 加权计算热度分数
        hotness = (normalized_papers * 0.3 + 
                  (normalized_growth + 1) / 2 * 0.4 +  # 将-1到1的growth_rate转换为0到1
                  normalized_citations * 0.3)
        
        return round(hotness, 3)
    
    @observe
    def calculate_opportunity_index(self, topic_name: str, trend_data: Dict[str, Any]) -> Dict[str, Any]:
        """计算机会指数"""
        # 获取竞争程度（基于论文数量）
        paper_count = trend_data.get('total_papers', 0)
        competition_level = self._assess_competition_level(paper_count)
        
        # 评估创新潜力（基于增长率和热度）
        growth_rate = trend_data.get('growth_rate', 0)
        hotness_score = trend_data.get('hotness_score', 0)
        innovation_potential = self._assess_innovation_potential(growth_rate, hotness_score)
        
        # 评估市场规模（基于平均引用数）
        avg_citation = trend_data.get('average_citation', 0)
        market_size = self._assess_market_size(avg_citation)
        
        # 评估难度等级
        difficulty_level = self._assess_difficulty_level(topic_name, avg_citation)
        
        # 计算综合机会指数
        weights = self.config['analysis_config']['opportunity_scoring']
        opportunity_index = (
            (1 - competition_level) * weights['competition_weight'] +
            innovation_potential * weights['innovation_weight'] +
            market_size * weights['market_weight']
        )
        
        return {
            "topic": topic_name,
            "opportunity_index": round(opportunity_index, 3),
            "competition_level": round(competition_level, 3),
            "innovation_potential": round(innovation_potential, 3),
            "market_size": round(market_size, 3),
            "difficulty_level": round(difficulty_level, 3),
            "overall_score": round(opportunity_index * 10, 2),  # 转换为10分制
            "recommendation": self._get_recommendation(opportunity_index)
        }
    
    def _assess_competition_level(self, paper_count: int) -> float:
        """评估竞争程度"""
        # 论文数量越多，竞争越激烈
        if paper_count < 10:
            return 0.2  # 低竞争
        elif paper_count < 50:
            return 0.5  # 中等竞争
        elif paper_count < 200:
            return 0.7  # 较高竞争
        else:
            return 0.9  # 高竞争
    
    def _assess_innovation_potential(self, growth_rate: float, hotness_score: float) -> float:
        """评估创新潜力"""
        # 增长率和热度越高，创新潜力越大
        return min((max(growth_rate, 0) * 0.6 + hotness_score * 0.4), 1.0)
    
    def _assess_market_size(self, avg_citation: float) -> float:
        """评估市场规模"""
        # 平均引用数越高，市场规模越大
        return min(avg_citation / 30, 1.0)  # 30引用作为满分基准
    
    def _assess_difficulty_level(self, topic_name: str, avg_citation: float) -> float:
        """评估难度等级"""
        # 基于主题复杂度和平均引用数
        complexity_keywords = ['deep learning', 'quantum', 'neural', 'algorithm', 'optimization']
        base_difficulty = 0.5
        
        # 检查是否包含复杂关键词
        if any(keyword in topic_name.lower() for keyword in complexity_keywords):
            base_difficulty += 0.3
        
        # 基于引用数调整
        citation_adjustment = min(avg_citation / 50, 0.3)
        
        return min(base_difficulty + citation_adjustment, 1.0)
    
    def _get_recommendation(self, opportunity_index: float) -> str:
        """获取建议"""
        if opportunity_index > 0.8:
            return "强烈推荐：这是一个高价值的蓝海研究领域"
        elif opportunity_index > 0.6:
            return "推荐：有较好的研究机会和发展前景"
        elif opportunity_index > 0.4:
            return "可以考虑：有一定潜力但需要谨慎评估"
        else:
            return "不推荐：竞争激烈或机会有限"


# 工具函数，供Agent使用
@observe
def analyze_research_trends(ctx: Context, topics: List[str] = None) -> str:
    """分析研究趋势的主要接口"""
    try:
        analyst = TrendAnalyst(ctx)
        
        with get_session() as session:
            # 如果没有指定主题，从数据库中分析
            if not topics:
                # 获取最近一年的论文
                start_date = datetime.now() - timedelta(days=365)
                recent_papers = session.query(Paper).filter(
                    Paper.publish_date >= start_date
                ).limit(1000).all()
                
                if not recent_papers:
                    return json.dumps({"status": "error", "message": "没有足够的论文数据进行分析"})
                
                # 提取主题
                topics = analyst.extract_research_topics(recent_papers)
            
            if not topics:
                return json.dumps({"status": "error", "message": "未能识别到研究主题"})
            
            results = []
            for topic in topics[:5]:  # 分析前5个主题
                trend_data = analyst.analyze_trend(topic)
                if "error" not in trend_data:
                    opportunity_data = analyst.calculate_opportunity_index(topic, trend_data)
                    results.append({
                        "topic": topic,
                        "trend_analysis": trend_data,
                        "opportunity_analysis": opportunity_data
                    })
            
            output = {
                "status": "success",
                "analysis_date": datetime.now().isoformat(),
                "topics_analyzed": len(results),
                "results": results
            }
            
            return json.dumps(output, ensure_ascii=False, indent=2)
            
    except Exception as e:
        error_output = {
            "status": "error",
            "error": str(e),
            "message": "趋势分析失败"
        }
        return json.dumps(error_output, ensure_ascii=False, indent=2)


@observe
def identify_blue_ocean_topics(ctx: Context) -> str:
    """识别蓝海研究主题"""
    try:
        analyst = TrendAnalyst(ctx)
        
        # 分析趋势
        trend_result = analyze_research_trends(ctx)
        trend_data = json.loads(trend_result)
        
        if trend_data.get("status") != "success":
            return json.dumps({"status": "error", "message": "无法获取趋势数据"})
        
        # 筛选蓝海主题
        blue_ocean_topics = []
        for result in trend_data.get("results", []):
            opportunity = result.get("opportunity_analysis", {})
            opportunity_index = opportunity.get("opportunity_index", 0)
            competition_level = opportunity.get("competition_level", 1)
            
            # 蓝海标准：高机会指数 + 低竞争
            if opportunity_index > 0.6 and competition_level < 0.6:
                blue_ocean_topics.append(result)
        
        # 按机会指数排序
        blue_ocean_topics.sort(key=lambda x: x["opportunity_analysis"]["opportunity_index"], reverse=True)
        
        output = {
            "status": "success",
            "analysis_date": datetime.now().isoformat(),
            "blue_ocean_topics": blue_ocean_topics[:10],  # 返回前10个
            "summary": f"识别到 {len(blue_ocean_topics)} 个蓝海研究主题"
        }
        
        return json.dumps(output, ensure_ascii=False, indent=2)
        
    except Exception as e:
        error_output = {
            "status": "error",
            "error": str(e),
            "message": "蓝海主题识别失败"
        }
        return json.dumps(error_output, ensure_ascii=False, indent=2)