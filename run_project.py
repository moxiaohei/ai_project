#!/usr/bin/env python3
"""
鸢尾花分类项目主运行脚本
"""

import os
import sys
import subprocess

def main():
    print("=== 鸢尾花分类项目 ===")
    
    # 检查依赖
    try:
        import pandas as pd
        import numpy as np
        from sklearn.datasets import load_iris
        import matplotlib.pyplot as plt
        print("✓ 所有依赖已安装")
    except ImportError as e:
        print(f"✗ 缺少依赖: {e}")
        print("请运行: pip3 install pandas numpy scikit-learn matplotlib seaborn")
        return
    
    # 运行数据预处理
    print("\n1. 运行数据预处理...")
    try:
        result = subprocess.run([sys.executable, "scripts/preprocess.py"], 
                               capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ 数据预处理完成")
            print(result.stdout)
        else:
            print(f"✗ 数据预处理失败: {result.stderr}")
            return
    except Exception as e:
        print(f"✗ 数据预处理失败: {e}")
        return
    
    # 运行模型训练
    print("\n2. 运行模型训练...")
    try:
        result = subprocess.run([sys.executable, "scripts/train_model.py"], 
                               capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ 模型训练完成")
            print(result.stdout)
        else:
            print(f"✗ 模型训练失败: {result.stderr}")
            return
    except Exception as e:
        print(f"✗ 模型训练失败: {e}")
        return
    
    print("\n🎉 项目运行完成！")
    print("查看生成的文件:")
    print("  - data/feature_distribution.png")
    print("  - data/correlation_heatmap.png")
    print("  - models/ 目录中的模型文件")

if __name__ == "__main__":
    main()
