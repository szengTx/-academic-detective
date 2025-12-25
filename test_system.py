"""
学术侦探系统测试
"""
import sys
import os

# 添加项目路径到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_basic_imports():
    """测试基础导入"""
    print("测试基础导入...")
    
    try:
        from agents.agent_simple import build_agent, process_academic_detective_request
        print("✓ Agent导入成功")
    except Exception as e:
        print(f"✗ Agent导入失败: {e}")
        return False
    
    try:
        from tools.data_collection_tool import collect_academic_data
        print("✓ 数据采集工具导入成功")
    except Exception as e:
        print(f"✗ 数据采集工具导入失败: {e}")
        return False
    
    try:
        from tools.trend_analysis_tool import analyze_research_trends
        print("✓ 趋势分析工具导入成功")
    except Exception as e:
        print(f"✗ 趋势分析工具导入失败: {e}")
        return False
    
    try:
        from tools.visualization_tool import create_knowledge_graph
        print("✓ 可视化工具导入成功")
    except Exception as e:
        print(f"✗ 可视化工具导入失败: {e}")
        return False
    
    return True

def test_agent_creation():
    """测试Agent创建"""
    print("\n测试Agent创建...")
    
    try:
        from agents.agent_simple import build_agent, process_academic_detective_request
        agent = build_agent()
        print("✓ Agent创建成功")
        return True
    except Exception as e:
        print(f"✗ Agent创建失败: {e}")
        return False

def test_simple_request():
    """测试简单请求"""
    print("\n测试简单请求...")
    
    try:
        from agents.agent_simple import process_academic_detective_request
        
        # 简单的测试请求
        test_request = "分析AI研究趋势"
        result = process_academic_detective_request(test_request)
        
        print("✓ 请求处理完成")
        print(f"结果长度: {len(result)} 字符")
        
        return True
    except Exception as e:
        print(f"✗ 请求处理失败: {e}")
        return False

def main():
    """主测试函数"""
    print("=== 学术侦探系统测试 ===\n")
    
    # 运行测试
    tests = [
        test_basic_imports,
        test_agent_creation,
        test_simple_request
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"=== 测试结果 ===")
    print(f"通过: {passed}/{total}")
    print(f"成功率: {passed/total*100:.1f}%")
    
    if passed == total:
        print("🎉 所有测试通过！")
    else:
        print("❌ 部分测试失败，请检查错误信息")

if __name__ == "__main__":
    main()