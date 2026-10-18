import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import time

class IrisClassifier:
    def __init__(self):
        self.models = {
            'decision_tree': DecisionTreeClassifier(max_depth=3, random_state=42),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'svm': SVC(kernel='rbf', random_state=42)
        }
        self.results = {}
        
    def load_data(self):
        """加载数据集"""
        iris = load_iris()
        self.X = iris.data
        self.y = iris.target
        self.feature_names = iris.feature_names
        self.target_names = iris.target_names
        
        # 分割数据集
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.3, random_state=42, stratify=self.y
        )
        
        print(f"训练集大小: {self.X_train.shape}")
        print(f"测试集大小: {self.X_test.shape}")
        
    def train_models(self):
        """训练所有模型"""
        for name, model in self.models.items():
            print(f"\n训练 {name}...")
            start_time = time.time()
            
            # 训练模型
            model.fit(self.X_train, self.y_train)
            
            # 预测
            y_pred = model.predict(self.X_test)
            accuracy = accuracy_score(self.y_test, y_pred)
            
            # 交叉验证
            cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5)
            
            training_time = time.time() - start_time
            
            self.results[name] = {
                'model': model,
                'accuracy': accuracy,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'training_time': training_time,
                'predictions': y_pred
            }
            
            print(f"{name} 准确率: {accuracy:.4f}")
            print(f"{name} 交叉验证平均分: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            print(f"{name} 训练时间: {training_time:.4f}秒")
    
    def evaluate_models(self):
        """评估模型性能"""
        best_model = None
        best_accuracy = 0
        
        print("\n" + "="*50)
        print("模型性能比较")
        print("="*50)
        
        for name, result in self.results.items():
            print(f"\n{name.upper()} 详细评估:")
            print(f"测试集准确率: {result['accuracy']:.4f}")
            print(f"5折交叉验证: {result['cv_mean']:.4f} (+/- {result['cv_std'] * 2:.4f})")
            
            # 分类报告
            print("\n分类报告:")
            print(classification_report(self.y_test, result['predictions'], 
                                      target_names=self.target_names))
            
            if result['accuracy'] > best_accuracy:
                best_accuracy = result['accuracy']
                best_model = name
        
        print(f"\n最佳模型: {best_model} (准确率: {best_accuracy:.4f})")
        return best_model
    
    def visualize_results(self, best_model_name):
        """可视化结果"""
        best_model = self.results[best_model_name]['model']
        
        # 混淆矩阵
        plt.figure(figsize=(12, 10))
        
        plt.subplot(2, 2, 1)
        y_pred_best = self.results[best_model_name]['predictions']
        cm = confusion_matrix(self.y_test, y_pred_best)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.target_names, yticklabels=self.target_names)
        plt.title(f'混淆矩阵 - {best_model_name}')
        plt.ylabel('真实标签')
        plt.xlabel('预测标签')
        
        # 决策树可视化（如果是决策树模型）
        if best_model_name == 'decision_tree':
            plt.subplot(2, 2, 2)
            plot_tree(best_model, 
                     feature_names=self.feature_names,
                     class_names=self.target_names,
                     filled=True, 
                     rounded=True)
            plt.title('决策树结构')
        
        # 模型比较图
        plt.subplot(2, 2, 3)
        model_names = list(self.results.keys())
        accuracies = [self.results[name]['accuracy'] for name in model_names]
        training_times = [self.results[name]['training_time'] for name in model_names]
        
        x = np.arange(len(model_names))
        width = 0.35
        
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        color = 'tab:blue'
        ax1.set_xlabel('模型')
        ax1.set_ylabel('准确率', color=color)
        bars1 = ax1.bar(x - width/2, accuracies, width, label='准确率', color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_ylim(0, 1)
        
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('训练时间(秒)', color=color)
        bars2 = ax2.bar(x + width/2, training_times, width, label='训练时间', color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        
        plt.title('模型性能比较')
        plt.xticks(x, model_names)
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')
        
        plt.tight_layout()
        plt.savefig('models/model_comparison.png', dpi=300, bbox_inches='tight')
        print("模型比较图已保存")
        
    def save_best_model(self, best_model_name):
        """保存最佳模型"""
        best_model = self.results[best_model_name]['model']
        joblib.dump(best_model, f'models/{best_model_name}_model.pkl')
        print(f"最佳模型已保存为 models/{best_model_name}_model.pkl")

def main():
    """主函数"""
    print("开始鸢尾花分类项目...")
    
    # 初始化分类器
    classifier = IrisClassifier()
    
    # 加载数据
    classifier.load_data()
    
    # 训练模型
    classifier.train_models()
    
    # 评估模型
    best_model = classifier.evaluate_models()
    
    # 可视化结果
    classifier.visualize_results(best_model)
    
    # 保存模型
    classifier.save_best_model(best_model)
    
    print("\n项目完成！")

if __name__ == "__main__":
    main()
