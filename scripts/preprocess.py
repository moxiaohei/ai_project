import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_explore_data():
    """加载和探索鸢尾花数据集"""
    iris = load_iris()
    
    # 创建DataFrame
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    df['species'] = df['target'].apply(lambda x: iris.target_names[x])
    
    print("数据集基本信息:")
    print(f"数据形状: {df.shape}")
    print(f"特征名称: {iris.feature_names}")
    print(f"目标类别: {list(iris.target_names)}")
    
    return df, iris

def create_visualizations(df, iris):
    """创建数据可视化图表"""
    plt.figure(figsize=(15, 10))
    
    # 特征分布图
    plt.subplot(2, 2, 1)
    sns.boxplot(data=df, x='species', y=iris.feature_names[0])
    plt.title('花萼长度分布')
    
    plt.subplot(2, 2, 2)
    sns.boxplot(data=df, x='species', y=iris.feature_names[1])
    plt.title('花萼宽度分布')
    
    plt.subplot(2, 2, 3)
    sns.boxplot(data=df, x='species', y=iris.feature_names[2])
    plt.title('花瓣长度分布')
    
    plt.subplot(2, 2, 4)
    sns.boxplot(data=df, x='species', y=iris.feature_names[3])
    plt.title('花瓣宽度分布')
    
    plt.tight_layout()
    plt.savefig('data/feature_distribution.png', dpi=300, bbox_inches='tight')
    print("特征分布图已保存")
    
    # 特征相关性热力图
    plt.figure(figsize=(8, 6))
    correlation_matrix = df[iris.feature_names].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('特征相关性热力图')
    plt.savefig('data/correlation_heatmap.png', dpi=300, bbox_inches='tight')
    print("相关性热力图已保存")

if __name__ == "__main__":
    df, iris = load_and_explore_data()
    create_visualizations(df, iris)
