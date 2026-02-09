#!/bin/bash
# 启动推荐服务

echo "启动推荐服务..."
echo "访问 http://127.0.0.1:8000/docs 查看 API 文档"
uvicorn rec_demo.service:app --reload --port 8000
