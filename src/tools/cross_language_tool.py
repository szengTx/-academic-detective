"""
跨语言工具 - 用于中英文概念对齐和知识整合
"""
import os
import json
import re
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
from sqlalchemy.orm import Session
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from coze_coding_utils.runtime_ctx.context import Context, default_headers
from cozeloop.decorator import observe

from storage.database.db import get_session
from storage.database.shared.academic_schema import Paper, ResearchTopic, ConceptMapping


class CrossLanguageExpert:
    """跨语言学术专家"""
    
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
    def translate_academic_concept(self, concept: str, source_lang: str = "en", target_lang: str = "zh") -> Dict[str, Any]:
        """翻译学术概念"""
        lang_map = {"en": "英文", "zh": "中文"}
        
        prompt = f"""
        作为专业的学术翻译专家，请翻译以下学术概念，并提供相关的学术术语对照。
        
        源语言：{lang_map.get(source_lang, source_lang)}
        目标语言：{lang_map.get(target_lang, target_lang)}
        概念：{concept}
        
        请按以下格式返回JSON：
        {{
            "original_concept": "原始概念",
            "translated_concept": "翻译后的概念",
            "alternatives": ["备选翻译1", "备选翻译2"],
            "context": "使用语境说明",
            "related_terms": [
                {{
                    "en": "英文相关术语",
                    "zh": "中文对应术语",
                    "relationship": "关系描述"
                }}
            ],
            "confidence": 0.95
        }}
        
        要求：
        1. 翻译要准确反映学术含义
        2. 提供多个备选翻译方案
        3. 列出相关的学术术语
        4. 评估翻译的置信度
        """
        
        try:
            messages = [
                SystemMessage(content="你是专业的学术翻译专家，精通中英文学术术语的翻译和概念对应。"),
                HumanMessage(content=prompt)
            ]
            
            response = ""
            for chunk in self.llm.stream(messages):
                if hasattr(chunk, 'content') and isinstance(chunk.content, str):
                    response += chunk.content
            
            # 解析JSON响应
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return {
                    "status": "success",
                    "concept": concept,
                    "translation": result
                }
            
        except Exception as e:
            print(f"LLM翻译失败: {str(e)}")
        
        # 回退到简单翻译
        return self._simple_translation(concept, source_lang, target_lang)
    
    def _simple_translation(self, concept: str, source_lang: str, target_lang: str) -> Dict[str, Any]:
        """简单翻译回退方案"""
        # 这里可以集成其他翻译API，目前返回基础映射
        common_mappings = {
            "machine learning": "机器学习",
            "deep learning": "深度学习", 
            "neural network": "神经网络",
            "artificial intelligence": "人工智能",
            "natural language processing": "自然语言处理",
            "computer vision": "计算机视觉",
            "reinforcement learning": "强化学习",
            "large language model": "大语言模型",
            "transformer": "变换器",
            "attention mechanism": "注意力机制",
            "generative ai": "生成式人工智能"
        }
        
        translated = common_mappings.get(concept.lower(), concept)
        
        return {
            "status": "success",
            "concept": concept,
            "translation": {
                "original_concept": concept,
                "translated_concept": translated,
                "alternatives": [translated],
                "context": "基于常见学术术语映射",
                "related_terms": [],
                "confidence": 0.6
            }
        }
    
    @observe
    def align_concepts(self, english_topics: List[str], chinese_topics: List[str]) -> List[Dict[str, Any]]:
        """对齐中英文概念"""
        alignments = []
        
        for en_topic in english_topics:
            best_match = None
            best_score = 0
            
            for zh_topic in chinese_topics:
                # 使用LLM计算语义相似度
                similarity_score = self._calculate_semantic_similarity(en_topic, zh_topic)
                
                if similarity_score > best_score and similarity_score > 0.7:  # 相似度阈值
                    best_score = similarity_score
                    best_match = zh_topic
            
            if best_match:
                alignments.append({
                    "english": en_topic,
                    "chinese": best_match,
                    "similarity_score": best_score,
                    "confidence": min(best_score * 1.1, 1.0),  # 调整置信度
                    "mapping_type": "semantic_alignment"
                })
        
        return alignments
    
    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """计算语义相似度"""
        prompt = f"""
        请评估以下两个学术概念的语义相似度，返回0-1之间的数值：
        
        概念1（英文）：{text1}
        概念2（中文）：{text2}
        
        请只返回一个数字（0-1之间），表示相似度分数。
        1.0表示完全相同，0.0表示完全不同。
        """
        
        try:
            messages = [
                SystemMessage(content="你是专业的语言专家，擅长评估跨语言概念的语义相似度。"),
                HumanMessage(content=prompt)
            ]
            
            response = ""
            for chunk in self.llm.stream(messages):
                if hasattr(chunk, 'content') and isinstance(chunk.content, str):
                    response += chunk.content
            
            # 提取数字
            numbers = re.findall(r'0\.\d+|1\.0|0\.0|1|0', response)
            if numbers:
                return float(numbers[0])
                
        except Exception as e:
            print(f"相似度计算失败: {str(e)}")
        
        # 回退到简单匹配
        return self._simple_similarity(text1, text2)
    
    def _simple_similarity(self, text1: str, text2: str) -> float:
        """简单相似度计算"""
        # 检查是否包含关键词
        en_lower = text1.lower()
        zh_lower = text2.lower()
        
        # 关键词映射
        keyword_mappings = {
            "ai": ["人工智能", "ai"],
            "machine learning": ["机器学习", "ml"],
            "deep learning": ["深度学习", "dl"],
            "neural": ["神经", "neural"],
            "vision": ["视觉", "vision"],
            "language": ["语言", "language"],
            "model": ["模型", "model"],
            "algorithm": ["算法", "algorithm"],
            "optimization": ["优化", "optimization"],
            "classification": ["分类", "classification"],
            "detection": ["检测", "detection"],
            "segmentation": ["分割", "segmentation"]
        }
        
        match_score = 0
        total_keywords = 0
        
        for en_keyword, zh_keywords in keyword_mappings.items():
            total_keywords += 1
            if en_keyword in en_lower:
                if any(zh_kw in zh_lower for zh_kw in zh_keywords):
                    match_score += 1
        
        if total_keywords == 0:
            return 0.0
        
        return match_score / total_keywords
    
    @observe
    def create_concept_mapping(self, topic: str, concept_en: str, concept_zh: str, similarity: float) -> Dict[str, Any]:
        """创建概念映射"""
        with get_session() as session:
            # 查找或创建主题
            db_topic = session.query(ResearchTopic).filter(
                (ResearchTopic.name_en == topic) | (ResearchTopic.name_zh == topic)
            ).first()
            
            if not db_topic:
                db_topic = ResearchTopic(
                    name_en=topic if concept_en == topic else topic,
                    name_zh=topic if concept_zh == topic else topic,
                    category="cross_language_mapped"
                )
                session.add(db_topic)
                session.flush()
            
            # 创建概念映射
            existing_mapping = session.query(ConceptMapping).filter(
                ConceptMapping.topic_id == db_topic.id,
                ConceptMapping.concept_en == concept_en,
                ConceptMapping.concept_zh == concept_zh
            ).first()
            
            if not existing_mapping:
                mapping = ConceptMapping(
                    topic_id=db_topic.id,
                    concept_en=concept_en,
                    concept_zh=concept_zh,
                    similarity_score=similarity,
                    mapping_type="llm_alignment",
                    confidence=min(similarity * 1.1, 1.0)
                )
                session.add(mapping)
            
            session.commit()
            
            return {
                "status": "success",
                "topic_id": db_topic.id,
                "concept_en": concept_en,
                "concept_zh": concept_zh,
                "similarity_score": similarity
            }
    
    @observe
    def analyze_language_gaps(self) -> Dict[str, Any]:
        """分析语言隔阂和机会"""
        with get_session() as session:
            # 获取英文和中文论文统计
            english_papers = session.query(Paper).filter(Paper.language == 'en').count()
            chinese_papers = session.query(Paper).filter(Paper.language == 'zh').count()
            
            # 获取主题统计
            english_topics = session.query(ResearchTopic).filter(
                ResearchTopic.name_en.isnot(None)
            ).all()
            
            chinese_topics = session.query(ResearchTopic).filter(
                ResearchTopic.name_zh.isnot(None)
            ).all()
            
            # 获取映射统计
            total_mappings = session.query(ConceptMapping).count()
            
            # 分析未被映射的英文主题
            unmapped_english = []
            for topic in english_topics:
                mapping_count = session.query(ConceptMapping).filter(
                    ConceptMapping.topic_id == topic.id
                ).count()
                
                if mapping_count == 0 and topic.name_en:
                    unmapped_english.append(topic.name_en)
            
            # 分析未被映射的中文主题
            unmapped_chinese = []
            for topic in chinese_topics:
                mapping_count = session.query(ConceptMapping).filter(
                    ConceptMapping.topic_id == topic.id
                ).count()
                
                if mapping_count == 0 and topic.name_zh:
                    unmapped_chinese.append(topic.name_zh)
            
            return {
                "paper_statistics": {
                    "english_papers": english_papers,
                    "chinese_papers": chinese_papers,
                    "total_papers": english_papers + chinese_papers,
                    "language_balance": english_papers / max(1, english_papers + chinese_papers)
                },
                "topic_statistics": {
                    "english_topics": len(english_topics),
                    "chinese_topics": len(chinese_topics),
                    "total_mappings": total_mappings
                },
                "gap_analysis": {
                    "unmapped_english_topics": unmapped_english[:10],  # 前10个
                    "unmapped_chinese_topics": unmapped_chinese[:10],   # 前10个
                    "mapping_coverage": total_mappings / max(1, min(len(english_topics), len(chinese_topics)))
                },
                "recommendations": self._generate_gap_recommendations(unmapped_english, unmapped_chinese)
            }
    
    def _generate_gap_recommendations(self, unmapped_english: List[str], unmapped_chinese: List[str]) -> List[str]:
        """生成隔阂分析建议"""
        recommendations = []
        
        if unmapped_english:
            recommendations.append(f"建议优先映射 {len(unmapped_english)} 个英文主题到中文，促进知识传播")
        
        if unmapped_chinese:
            recommendations.append(f"建议优先映射 {len(unmapped_chinese)} 个中文主题到英文，增强国际影响力")
        
        if len(unmapped_english) > len(unmapped_chinese):
            recommendations.append("英文研究主题较多，建议加强中文学术生态建设")
        else:
            recommendations.append("中文研究主题较多，建议加强国际化学术交流")
        
        recommendations.append("建立自动化的概念映射更新机制")
        recommendations.append("鼓励研究者进行跨语言合作研究")
        
        return recommendations


# 工具函数，供Agent使用
@observe
def align_cross_language_concepts(ctx: Context) -> str:
    """跨语言概念对齐的主要接口"""
    try:
        expert = CrossLanguageExpert(ctx)
        
        with get_session() as session:
            # 获取英文主题
            english_papers = session.query(Paper).filter(Paper.language == 'en').limit(500).all()
            english_keywords = []
            for paper in english_papers:
                if paper.keywords:
                    english_keywords.extend([k.strip() for k in paper.keywords.split(',') if k.strip()])
            
            english_topics = list(set([kw for kw in english_keywords if len(kw) > 3]))[:20]
            
            # 获取中文主题
            chinese_papers = session.query(Paper).filter(Paper.language == 'zh').limit(500).all()
            chinese_keywords = []
            for paper in chinese_papers:
                if paper.keywords:
                    chinese_keywords.extend([k.strip() for k in paper.keywords.split(',') if k.strip()])
            
            chinese_topics = list(set([kw for kw in chinese_keywords if len(kw) > 3]))[:20]
            
            # 进行概念对齐
            alignments = expert.align_concepts(english_topics, chinese_topics)
            
            # 保存映射到数据库
            saved_mappings = []
            for alignment in alignments:
                result = expert.create_concept_mapping(
                    alignment["english"],
                    alignment["english"],
                    alignment["chinese"],
                    alignment["similarity_score"]
                )
                if result["status"] == "success":
                    saved_mappings.append(alignment)
            
            # 分析语言隔阂
            gap_analysis = expert.analyze_language_gaps()
            
            output = {
                "status": "success",
                "alignment_date": datetime.now().isoformat(),
                "total_alignments": len(saved_mappings),
                "alignments": saved_mappings,
                "gap_analysis": gap_analysis,
                "message": f"成功对齐 {len(saved_mappings)} 个中英文概念"
            }
            
            return json.dumps(output, ensure_ascii=False, indent=2)
            
    except Exception as e:
        error_output = {
            "status": "error",
            "error": str(e),
            "message": "跨语言概念对齐失败"
        }
        return json.dumps(error_output, ensure_ascii=False, indent=2)


@observe
def translate_and_map_concept(ctx: Context, concept: str, target_lang: str = "zh") -> str:
    """翻译并映射单个概念"""
    try:
        expert = CrossLanguageExpert(ctx)
        
        # 翻译概念
        if target_lang == "zh":
            translation_result = expert.translate_academic_concept(concept, "en", "zh")
        else:
            translation_result = expert.translate_academic_concept(concept, "zh", "en")
        
        if translation_result["status"] == "success":
            translation = translation_result["translation"]
            
            # 创建映射
            if target_lang == "zh":
                mapping_result = expert.create_concept_mapping(
                    concept,
                    concept,
                    translation["translated_concept"],
                    translation["confidence"]
                )
            else:
                mapping_result = expert.create_concept_mapping(
                    concept,
                    translation["translated_concept"],
                    concept,
                    translation["confidence"]
                )
            
            output = {
                "status": "success",
                "concept": concept,
                "translation": translation,
                "mapping": mapping_result
            }
        else:
            output = translation_result
        
        return json.dumps(output, ensure_ascii=False, indent=2)
        
    except Exception as e:
        error_output = {
            "status": "error",
            "error": str(e),
            "message": "概念翻译失败"
        }
        return json.dumps(error_output, ensure_ascii=False, indent=2)