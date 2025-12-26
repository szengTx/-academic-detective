# 学术侦探系统 - 快速部署指南

## 📦 快速下载和部署

### 选项1：完整项目结构（推荐）

#### 核心文件清单：
```
academic-detective/
├── README.md                           # 项目说明
├── requirements.txt                    # Python依赖
├── test_system.py                     # 系统测试
│
├── config/                            # 配置文件
│   └── academic_detective_config.json
│
├── src/                               # 源代码
│   ├── agents/                        # Agent实现
│   │   ├── agent_tools.py            # Agent工具函数
│   │   └── agent_simple.py           # 简化版Agent
│   │
│   ├── tools/                         # 工具模块
│   │   ├── cross_language_tool.py    # 跨语言对齐
│   │   ├── data_collection_tool.py    # 数据采集
│   │   ├── reflection_tool.py         # 系统反思
│   │   ├── trend_analysis_tool.py     # 趋势分析
│   │   ├── visualization_tool.py      # 可视化
│   │   └── web_search_tool.py         # 网络搜索
│   │
│   ├── storage/                       # 数据存储
│   │   └── database/
│   │       └── shared/
│   │           └── academic_schema.py # 数据库模型
│   │
│   └── utils/                         # 工具函数
│       └── agent_communication.py     # Agent通信
│
├── scripts/                           # 脚本
│   └── init_database.py              # 数据库初始化
│
└── docs/                             # 文档
    ├── ACADEMIC_DETECTIVE_README.md   # 详细文档
    └── architecture.md               # 架构说明
```

### 选项2：最小化部署包

只需要以下文件即可运行：

1. **requirements.txt**
2. **config/academic_detective_config.json**
3. **src/agents/agent_simple.py**
4. **src/agents/agent_tools.py**
5. **src/tools/** 目录下的所有文件

## 🚀 快速启动

### 步骤1：创建项目目录
```bash
mkdir academic-detective
cd academic-detective
```

### 步骤2：安装依赖
```bash
pip install -r requirements.txt
```

### 步骤3：初始化数据库
```bash
python scripts/init_database.py
```

### 步骤4：启动系统
```bash
cd src
python main.py -m http -p 5000
```

### 步骤5：访问系统
- **API文档**: http://localhost:5000/docs
- **健康检查**: http://localhost:5000/health

## 🔧 环境要求

- Python 3.8+
- PostgreSQL 12+
- 8GB+ RAM

## 📞 支持

如果遇到问题，请检查：
1. Python版本是否符合要求
2. 所有依赖是否正确安装
3. 数据库连接是否正常
4. 环境变量是否配置正确