#!/bin/bash

# 获取脚本所在目录
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")

# 切换到项目根目录
cd "$SCRIPT_DIR/../.."

# 使用相对路径
python3 demo_2/src/evaluate.py $1 $2 $3 $4 $5 $6