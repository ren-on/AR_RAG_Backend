# RAG应收帐系统后端（FastAPI）

本项目为应收帐RAG系统的后端API，基于FastAPI开发。

## 主要功能
- 上传AR数据Excel文件
- 支持自然语言应收账款查询.

## 依赖安装
```bash
pip install -r requirements.txt
```

## 启动方法
本地开发：
```bash
python main.py
```
生产部署（推荐）：
```bash
gunicorn main:app --bind=0.0.0.0:8000 --timeout 120
```

## 环境变量
请参考 env_example.txt 配置Azure OpenAI相关环境变量。 