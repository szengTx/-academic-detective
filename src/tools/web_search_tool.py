"""
网络搜索工具 - 基于集成搜索API的封装
"""
import os
import requests
from typing import Optional, List, Dict, Any, Tuple
from pydantic import BaseModel, Field
from cozeloop.decorator import observe
from coze_coding_utils.runtime_ctx.context import Context, default_headers


class WebItem(BaseModel):
    """Web搜索结果项模型"""
    Id: str = Field(..., description="结果Id")
    SortId: int = Field(..., description="排序Id")
    Title: str = Field(..., description="标题")
    SiteName: Optional[str] = Field(None, description="站点名")
    Url: Optional[str] = Field(None, description="落地页")
    Snippet: str = Field(..., description="普通摘要")
    Summary: Optional[str] = Field(None, description="精准摘要")
    Content: Optional[str] = Field(None, description="正文")
    PublishTime: Optional[str] = Field(None, description="发布时间")
    LogoUrl: Optional[str] = Field(None, description="落地页IconUrl链接")
    RankScore: Optional[float] = Field(None, description="得分")
    AuthInfoDes: str = Field(..., description="权威度描述")
    AuthInfoLevel: int = Field(..., description="权威度评级")


class ImageInfo(BaseModel):
    """图片结果项"""
    Url: str = Field(..., description="图片链接")
    Width: Optional[int] = Field(None, description="宽")
    Height: Optional[int] = Field(None, description="高")
    Shape: str = Field(..., description="图片形状")


class ImageItem(BaseModel):
    """ImageItem-搜索结果项"""
    Id: str = Field(..., description="结果Id")
    SortId: int = Field(..., description="排序Id")
    Title: Optional[str] = Field(None, description="标题")
    SiteName: Optional[str] = Field(None, description="站点名")
    Url: Optional[str] = Field(None, description="落地页")
    PublishTime: Optional[str] = Field(None, description="发布时间")
    Image: ImageInfo = Field(..., description="图片详情")


@observe
def web_search(
        ctx: Context,
        query: str,
        search_type: str = "web",
        count: Optional[int] = 10,
        need_content: Optional[bool] = False,
        need_url: Optional[bool] = False,
        sites: Optional[str] = None,
        block_hosts: Optional[str] = None,
        need_summary: Optional[bool] = True,
        time_range: Optional[str] = None,
) -> Tuple[List[WebItem], str, Optional[List[ImageItem]], Dict]:
    """
    融合信息搜索API，返回搜索结果项列表、搜索结果内容总结和原始响应数据。
    
    Args:
        ctx: 上下文对象
        query: 用户搜索query，1~100个字符
        search_type: 搜索类型 (web/web_summary/image)
        count: 返回条数，最多50条
        need_content: 是否仅返回有正文的结果
        need_url: 是否仅返回原文链接的结果
        sites: 指定搜索的Site范围，多个域名使用'|'分隔
        block_hosts: 指定屏蔽的搜索Site
        need_summary: 是否需要精准摘要
        time_range: 指定搜索的发文时间
    
    Returns:
        tuple: (WebItem列表, 搜索结果摘要, ImageItem列表, 原始响应数据)
    """
    api_key = os.getenv("COZE_WORKLOAD_IDENTITY_API_KEY")
    base_url = os.getenv("COZE_INTEGRATION_BASE_URL")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    headers.update(default_headers(ctx))
    
    request = {
        "Query": query,
        "SearchType": search_type,
        "Count": count,
        "Filter": {
            "NeedContent": need_content,
            "NeedUrl": need_url,
            "Sites": sites,
            "BlockHosts": block_hosts,
        },
        "NeedSummary": need_summary,
        "TimeRange": time_range,
    }
    
    try:
        response = requests.post(f'{base_url}/api/search_api/web_search', json=request, headers=headers)
        response.raise_for_status()
        data = response.json()

        response_metadata = data.get("ResponseMetadata", {})
        result = data.get("Result", {})
        
        if response_metadata.get("Error"):
            raise Exception(f"web_search 失败: {response_metadata.get('Error')}")

        web_items = []
        image_items = []
        
        if result.get("WebResults"):
            web_items = [WebItem(**item) for item in result.get("WebResults", [])]
        
        if result.get("ImageResults"):
            image_items = [ImageItem(**item) for item in result.get("ImageResults", [])]
        
        content = None
        if result.get("Choices"):
            content = result.get("Choices", [{}])[0].get("Message", {}).get("Content", "")
            
        return web_items, content, image_items, result
        
    except requests.RequestException as e:
        raise Exception(f"网络请求失败: {str(e)}")
    except Exception as e:
        raise Exception(f"web_search 失败: {str(e)}")
    finally:
        if 'response' in locals():
            response.close()


# 便捷搜索函数
@observe
def search_academic_content(ctx: Context, topic: str, max_results: int = 20) -> str:
    """搜索学术内容"""
    try:
        query = f"{topic} academic research paper study"
        web_items, content, image_items, result = web_search(
            ctx=ctx,
            query=query,
            search_type="web_summary",
            count=max_results,
            need_content=True,
            need_url=True
        )
        
        # 格式化结果
        results = []
        for item in web_items:
            results.append({
                "title": item.Title,
                "summary": item.Summary or item.Snippet,
                "url": item.Url,
                "site": item.SiteName,
                "publish_time": item.PublishTime
            })
        
        output = {
            "status": "success",
            "query": topic,
            "total_results": len(results),
            "summary": content,
            "results": results
        }
        
        import json
        return json.dumps(output, ensure_ascii=False, indent=2)
        
    except Exception as e:
        import json
        error_output = {
            "status": "error",
            "query": topic,
            "error": str(e)
        }
        return json.dumps(error_output, ensure_ascii=False, indent=2)


@observe
def search_chinese_content(ctx: Context, topic: str, max_results: int = 20) -> str:
    """搜索中文内容"""
    try:
        query = f"{topic} 学术研究 论文"
        # 限制在中文站点
        sites = "cnki.net|wanfangdata.com.cn|cqvip.com|xueshu.baidu.com"
        
        web_items, content, image_items, result = web_search(
            ctx=ctx,
            query=query,
            search_type="web_summary",
            count=max_results,
            sites=sites
        )
        
        # 格式化结果
        results = []
        for item in web_items:
            results.append({
                "title": item.Title,
                "summary": item.Summary or item.Snippet,
                "url": item.Url,
                "site": item.SiteName,
                "publish_time": item.PublishTime
            })
        
        output = {
            "status": "success",
            "query": topic,
            "total_results": len(results),
            "summary": content,
            "results": results
        }
        
        import json
        return json.dumps(output, ensure_ascii=False, indent=2)
        
    except Exception as e:
        import json
        error_output = {
            "status": "error",
            "query": topic,
            "error": str(e)
        }
        return json.dumps(error_output, ensure_ascii=False, indent=2)