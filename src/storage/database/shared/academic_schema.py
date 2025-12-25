"""
学术侦探系统数据库Schema定义
"""
from sqlalchemy import Column, Integer, String, Text, DateTime, Float, Boolean, JSON, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()


class Paper(Base):
    """论文表"""
    __tablename__ = 'papers'
    
    id = Column(Integer, primary_key=True)
    title = Column(String(500), nullable=False, comment='论文标题')
    authors = Column(Text, comment='作者列表')
    abstract = Column(Text, comment='摘要')
    content = Column(Text, comment='全文内容')
    keywords = Column(Text, comment='关键词')
    language = Column(String(10), comment='语言')
    source = Column(String(50), comment='来源平台')
    source_url = Column(String(500), comment='源链接')
    publish_date = Column(DateTime, comment='发布日期')
    collection_date = Column(DateTime, default=datetime.utcnow, comment='采集日期')
    doi = Column(String(100), comment='DOI')
    citation_count = Column(Integer, default=0, comment='引用数')
    is_processed = Column(Boolean, default=False, comment='是否已处理')
    
    # 关联关系
    topics = relationship("PaperTopic", back_populates="paper")
    trend_data = relationship("TrendData", back_populates="paper")
    
    def __repr__(self):
        return f"<Paper(id={self.id}, title='{self.title[:50]}...')>"


class ResearchTopic(Base):
    """研究主题表"""
    __tablename__ = 'research_topics'
    
    id = Column(Integer, primary_key=True)
    name_en = Column(String(200), comment='英文名称')
    name_zh = Column(String(200), comment='中文名称')
    description = Column(Text, comment='描述')
    category = Column(String(100), comment='类别')
    created_date = Column(DateTime, default=datetime.utcnow, comment='创建日期')
    
    # 关联关系
    papers = relationship("PaperTopic", back_populates="topic")
    trend_analysis = relationship("TrendAnalysis", back_populates="topic")
    opportunity_scores = relationship("OpportunityScore", back_populates="topic")
    concept_mappings = relationship("ConceptMapping", back_populates="topic")
    
    def __repr__(self):
        return f"<ResearchTopic(id={self.id}, name_en='{self.name_en}')>"


class PaperTopic(Base):
    """论文-主题关联表"""
    __tablename__ = 'paper_topics'
    
    id = Column(Integer, primary_key=True)
    paper_id = Column(Integer, ForeignKey('papers.id'), nullable=False)
    topic_id = Column(Integer, ForeignKey('research_topics.id'), nullable=False)
    relevance_score = Column(Float, comment='相关性评分')
    extracted_date = Column(DateTime, default=datetime.utcnow, comment='提取日期')
    
    # 关联关系
    paper = relationship("Paper", back_populates="topics")
    topic = relationship("ResearchTopic", back_populates="papers")


class TrendAnalysis(Base):
    """趋势分析表"""
    __tablename__ = 'trend_analysis'
    
    id = Column(Integer, primary_key=True)
    topic_id = Column(Integer, ForeignKey('research_topics.id'), nullable=False)
    analysis_date = Column(DateTime, default=datetime.utcnow, comment='分析日期')
    paper_count = Column(Integer, comment='相关论文数量')
    growth_rate = Column(Float, comment='增长率')
    average_citation = Column(Float, comment='平均引用数')
    hotness_score = Column(Float, comment='热度分数')
    trend_direction = Column(String(20), comment='趋势方向(up/down/stable)')
    predicted_growth = Column(Float, comment='预测增长率')
    confidence = Column(Float, comment='置信度')
    
    # 关联关系
    topic = relationship("ResearchTopic", back_populates="trend_analysis")


class OpportunityScore(Base):
    """机会指数表"""
    __tablename__ = 'opportunity_scores'
    
    id = Column(Integer, primary_key=True)
    topic_id = Column(Integer, ForeignKey('research_topics.id'), nullable=False)
    score_date = Column(DateTime, default=datetime.utcnow, comment='评分日期')
    opportunity_index = Column(Float, comment='机会指数')
    competition_level = Column(Float, comment='竞争程度')
    market_size = Column(Float, comment='市场规模')
    innovation_potential = Column(Float, comment='创新潜力')
    difficulty_level = Column(Float, comment='难度等级')
    overall_score = Column(Float, comment='综合评分')
    
    # 关联关系
    topic = relationship("ResearchTopic", back_populates="opportunity_scores")


class ConceptMapping(Base):
    """概念映射表（中英文对齐）"""
    __tablename__ = 'concept_mappings'
    
    id = Column(Integer, primary_key=True)
    topic_id = Column(Integer, ForeignKey('research_topics.id'), nullable=False)
    concept_en = Column(String(200), comment='英文概念')
    concept_zh = Column(String(200), comment='中文概念')
    similarity_score = Column(Float, comment='相似度分数')
    mapping_type = Column(String(50), comment='映射类型')
    confidence = Column(Float, comment='置信度')
    created_date = Column(DateTime, default=datetime.utcnow, comment='创建日期')
    
    # 关联关系
    topic = relationship("ResearchTopic", back_populates="concept_mappings")


class TrendData(Base):
    """趋势数据表"""
    __tablename__ = 'trend_data'
    
    id = Column(Integer, primary_key=True)
    paper_id = Column(Integer, ForeignKey('papers.id'), nullable=False)
    date = Column(DateTime, comment='日期')
    citation_count = Column(Integer, comment='引用数')
    download_count = Column(Integer, comment='下载次数')
    view_count = Column(Integer, comment='浏览次数')
    social_media_mentions = Column(Integer, comment='社交媒体提及次数')
    
    # 关联关系
    paper = relationship("Paper", back_populates="trend_data")


class SystemMetrics(Base):
    """系统指标表"""
    __tablename__ = 'system_metrics'
    
    id = Column(Integer, primary_key=True)
    metric_date = Column(DateTime, default=datetime.utcnow, comment='指标日期')
    total_papers = Column(Integer, comment='总论文数')
    processed_papers = Column(Integer, comment='已处理论文数')
    total_topics = Column(Integer, comment='总主题数')
    active_topics = Column(Integer, comment='活跃主题数')
    prediction_accuracy = Column(Float, comment='预测准确率')
    system_performance = Column(JSON, comment='系统性能数据')
    error_count = Column(Integer, comment='错误次数')


class TaskLog(Base):
    """任务日志表"""
    __tablename__ = 'task_logs'
    
    id = Column(Integer, primary_key=True)
    task_id = Column(String(100), comment='任务ID')
    agent_name = Column(String(100), comment='Agent名称')
    task_type = Column(String(100), comment='任务类型')
    status = Column(String(20), comment='状态(started/completed/failed)')
    start_time = Column(DateTime, comment='开始时间')
    end_time = Column(DateTime, comment='结束时间')
    duration = Column(Float, comment='持续时间(秒)')
    input_data = Column(JSON, comment='输入数据')
    output_data = Column(JSON, comment='输出数据')
    error_message = Column(Text, comment='错误信息')
    extra_metadata = Column(JSON, comment='元数据')


class VisualizationCache(Base):
    """可视化缓存表"""
    __tablename__ = 'visualization_cache'
    
    id = Column(Integer, primary_key=True)
    cache_key = Column(String(200), unique=True, comment='缓存键')
    viz_type = Column(String(50), comment='可视化类型')
    data = Column(JSON, comment='可视化数据')
    created_date = Column(DateTime, default=datetime.utcnow, comment='创建日期')
    expiry_date = Column(DateTime, comment='过期日期')
    is_valid = Column(Boolean, default=True, comment='是否有效')