#!/bin/bash

# 服务器部署脚本
# 在本地执行此脚本进行部署

SERVER="group28@121.14.82.109"
PROJECT_DIR="bigdatamodel"

# 密码不再硬编码：优先读取环境变量 DEPLOY_PASSWORD，否则交互式输入
if [ -z "$DEPLOY_PASSWORD" ]; then
    read -rsp "请输入服务器密码: " DEPLOY_PASSWORD
    echo
fi
PASSWORD="$DEPLOY_PASSWORD"

echo "=== 开始部署到服务器 ==="

# 1. 创建临时目录用于传输
echo "[1/5] 连接到服务器并创建项目目录..."
sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=no $SERVER << 'EOF'
mkdir -p ~/bigdatamodel
exit
EOF

# 2. 上传文件 (排除 .venv, .git, 调试文件等)
echo "[2/5] 上传项目文件..."
sshpass -p "$PASSWORD" scp -o StrictHostKeyChecking=no -r \
    main.py \
    requirements.txt \
    .env \
    prompts.py \
    scoring_engine.py \
    validator.py \
    db_init.py \
    kg_init.py \
    ingest.py \
    logic_adapter.py \
    batch_processor.py \
    kg_pipeline.py \
    init_kg.py \
    $SERVER:~/bigdatamodel/

# 3. 上传 pages 目录
echo "[3/5] 上传 pages 目录..."
sshpass -p "$PASSWORD" scp -o StrictHostKeyChecking=no -r \
    pages/ \
    $SERVER:~/bigdatamodel/

# 4. 安装依赖
echo "[4/5] 在服务器上安装依赖..."
sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=no $SERVER << 'EOF'
cd ~/bigdatamodel
pip install -r requirements.txt
exit
EOF

# 5. 启动应用
echo "[5/5] 启动 Streamlit 应用..."
sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=no $SERVER << 'EOF'
cd ~/bigdatamodel
nohup streamlit run main.py --server.port 8280 --server.address 0.0.0.0 > streamlit.log 2>&1 &
echo "应用已启动!"
exit
EOF

echo "=== 部署完成 ==="
echo "访问地址: http://121.14.82.109:8280"
