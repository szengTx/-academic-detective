"""
数据库初始化脚本
"""
import os
import sys
from sqlalchemy import create_engine
from src.storage.database.shared.academic_schema import Base
from src.storage.database.db import get_db_url


def init_database():
    """初始化数据库"""
    print("正在初始化数据库...")
    
    # 创建数据库引擎
    db_url = get_db_url()
    engine = create_engine(db_url)
    
    # 创建所有表
    Base.metadata.create_all(engine)
    
    print("数据库初始化完成！")
    print(f"数据库连接: {db_url}")


if __name__ == "__main__":
    init_database()