"""
数据采集工具 - 用于从多个学术数据源采集论文数据
"""
import os
import json
import time
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from urllib.parse import quote
import feedparser
from sqlalchemy.orm import Session
from coze_coding_utils.runtime_ctx.context import Context, default_headers
from cozeloop.decorator import observe

from storage.database.db import get_session
from storage.database.shared.academic_schema import Paper


class DataCollector:
    """学术数据采集器"""
    
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
    def collect_from_arxiv(self, query: str, max_results: int = 100, days_back: int = 7) -> List[Dict[str, Any]]:
        """从arXiv采集论文数据"""
        base_url = "http://export.arxiv.org/api/query"
        
        # 构建搜索查询
        search_query = f'all:"{query}"'
        start_date = datetime.now() - timedelta(days=days_back)
        
        # 构建请求参数
        params = {
            'search_query': search_query,
            'start': 0,
            'max_results': max_results,
            'sortBy': 'submittedDate',
            'sortOrder': 'descending'
        }
        
        papers = []
        try:
            response = requests.get(base_url, params=params, timeout=30)
            response.raise_for_status()
            
            # 解析XML响应
            feed = feedparser.parse(response.content)
            
            for entry in feed.entries:
                # 提取论文信息
                paper_data = {
                    'title': entry.title.strip(),
                    'authors': ', '.join([author.name for author in entry.authors]),
                    'abstract': entry.summary.strip(),
                    'source': 'arxiv',
                    'source_url': entry.id,
                    'publish_date': self._parse_arxiv_date(entry.published),
                    'collection_date': datetime.utcnow(),
                    'doi': entry.get('id', ''),
                    'language': 'en',
                    'keywords': self._extract_keywords(entry.summary)
                }
                
                # 过滤日期范围
                if paper_data['publish_date'] >= start_date:
                    papers.append(paper_data)
                    
        except Exception as e:
            raise Exception(f"arXiv数据采集失败: {str(e)}")
            
        return papers
    
    @observe 
    def collect_from_web_search(self, query: str, max_results: int = 50, days_back: int = 7) -> List[Dict[str, Any]]:
        """通过网络搜索采集学术数据"""
        from .web_search_tool import web_search
        
        papers = []
        try:
            # 构建学术搜索查询
            academic_query = f"{query} academic research paper"
            time_range = f"{(datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')}..{datetime.now().strftime('%Y-%m-%d')}"
            
            # 调用网络搜索
            web_items, content, image_items, result = web_search(
                ctx=self.ctx,
                query=academic_query,
                search_type="web_summary",
                count=max_results,
                need_content=True,
                need_url=True,
                time_range=time_range
            )
            
            for item in web_items:
                if item.Url and self._is_academic_source(item.Url):
                    paper_data = {
                        'title': item.Title,
                        'abstract': item.Summary or item.Snippet,
                        'source': 'web_search',
                        'source_url': item.Url,
                        'publish_date': self._parse_publish_date(item.PublishTime),
                        'collection_date': datetime.utcnow(),
                        'language': 'en',
                        'keywords': self._extract_keywords(item.Title + ' ' + (item.Summary or item.Snippet))
                    }
                    papers.append(paper_data)
                    
        except Exception as e:
            raise Exception(f"网络搜索数据采集失败: {str(e)}")
            
        return papers
    
    @observe
    def collect_chinese_papers(self, query: str, max_results: int = 50, days_back: int = 7) -> List[Dict[str, Any]]:
        """采集中文论文数据"""
        from .web_search_tool import web_search
        
        papers = []
        try:
            # 中文搜索查询
            chinese_query = f"{query} 学术论文 研究"
            time_range = f"{(datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')}..{datetime.now().strftime('%Y-%m-%d')}"
            
            # 限制在中文站点搜索
            sites = "cnki.net|wanfangdata.com.cn|cqvip.com"
            
            web_items, content, image_items, result = web_search(
                ctx=self.ctx,
                query=chinese_query,
                search_type="web_summary",
                count=max_results,
                sites=sites,
                time_range=time_range
            )
            
            for item in web_items:
                paper_data = {
                    'title': item.Title,
                    'abstract': item.Summary or item.Snippet,
                    'source': 'chinese_web',
                    'source_url': item.Url,
                    'publish_date': self._parse_publish_date(item.PublishTime),
                    'collection_date': datetime.utcnow(),
                    'language': 'zh',
                    'keywords': self._extract_keywords(item.Title + ' ' + (item.Summary or item.Snippet))
                }
                papers.append(paper_data)
                
        except Exception as e:
            raise Exception(f"中文论文采集失败: {str(e)}")
            
        return papers
    
    @observe
    def save_papers_to_db(self, papers: List[Dict[str, Any]]) -> int:
        """将论文数据保存到数据库"""
        saved_count = 0
        
        with get_session() as session:
            for paper_data in papers:
                # 检查是否已存在（基于URL或标题）
                existing = session.query(Paper).filter(
                    (Paper.source_url == paper_data.get('source_url')) |
                    (Paper.title == paper_data['title'])
                ).first()
                
                if not existing:
                    paper = Paper(
                        title=paper_data['title'],
                        authors=paper_data.get('authors', ''),
                        abstract=paper_data.get('abstract', ''),
                        keywords=paper_data.get('keywords', ''),
                        language=paper_data.get('language', 'en'),
                        source=paper_data.get('source', 'unknown'),
                        source_url=paper_data.get('source_url', ''),
                        publish_date=paper_data.get('publish_date'),
                        collection_date=paper_data.get('collection_date', datetime.utcnow()),
                        doi=paper_data.get('doi', ''),
                        is_processed=False
                    )
                    
                    session.add(paper)
                    saved_count += 1
            
            session.commit()
            
        return saved_count
    
    def _parse_arxiv_date(self, date_str: str) -> datetime:
        """解析arXiv日期格式"""
        try:
            return datetime.strptime(date_str.split('Z')[0], '%Y-%m-%dT%H:%M:%S')
        except:
            return datetime.utcnow()
    
    def _parse_publish_date(self, date_str: Optional[str]) -> Optional[datetime]:
        """解析发布日期"""
        if not date_str:
            return None
        try:
            return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        except:
            return None
    
    def _extract_keywords(self, text: str) -> str:
        """从文本中提取关键词"""
        # 简单的关键词提取，可以后续优化
        if not text:
            return ""
        
        # 这里可以使用更复杂的NLP技术
        words = text.lower().split()
        # 过滤停用词
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those'}
        
        keywords = [word for word in words if word not in stopwords and len(word) > 2]
        return ', '.join(keywords[:10])  # 返回前10个关键词
    
    def _is_academic_source(self, url: str) -> bool:
        """判断是否为学术来源"""
        academic_domains = ['arxiv.org', 'scholar.google.com', 'researchgate.net', 'acm.org', 'ieee.org', 'springer.com', 'sciencedirect.com', 'nature.com', 'pnas.org']
        return any(domain in url.lower() for domain in academic_domains)
    
    @observe
    def collect_all_data(self, queries: List[str], max_results_per_source: int = 100) -> Dict[str, int]:
        """从所有数据源采集数据"""
        collection_stats = {
            'arxiv': 0,
            'web_search': 0,
            'chinese': 0,
            'total_saved': 0
        }
        
        all_papers = []
        
        for query in queries:
            try:
                # 从arXiv采集
                arxiv_papers = self.collect_from_arxiv(query, max_results_per_source // 2)
                all_papers.extend(arxiv_papers)
                collection_stats['arxiv'] += len(arxiv_papers)
                
                # 从网络搜索采集
                web_papers = self.collect_from_web_search(query, max_results_per_source // 4)
                all_papers.extend(web_papers)
                collection_stats['web_search'] += len(web_papers)
                
                # 采集中文论文
                chinese_papers = self.collect_chinese_papers(query, max_results_per_source // 4)
                all_papers.extend(chinese_papers)
                collection_stats['chinese'] += len(chinese_papers)
                
                # 避免请求过于频繁
                time.sleep(2)
                
            except Exception as e:
                print(f"采集查询 '{query}' 时出错: {str(e)}")
        
        # 保存到数据库
        saved_count = self.save_papers_to_db(all_papers)
        collection_stats['total_saved'] = saved_count
        
        return collection_stats


# 工具函数，供Agent使用
@observe
def collect_academic_data(ctx: Context, queries: List[str], max_results: int = 100) -> str:
    """采集学术数据的主要接口"""
    try:
        collector = DataCollector(ctx)
        stats = collector.collect_all_data(queries, max_results)
        
        result = {
            "status": "success",
            "collection_stats": stats,
            "message": f"数据采集完成，共保存 {stats['total_saved']} 篇论文到数据库"
        }
        
        return json.dumps(result, ensure_ascii=False, indent=2)
        
    except Exception as e:
        error_result = {
            "status": "error", 
            "error": str(e),
            "message": "数据采集失败"
        }
        return json.dumps(error_result, ensure_ascii=False, indent=2)