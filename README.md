# ai_project · 鸢尾花多模型分类对比

基于 scikit-learn 的鸢尾花（Iris）三分类实验项目：加载经典 Iris 数据集，对比 **决策树、随机森林、SVM** 三种模型的准确率与训练耗时，完成数据探索可视化、训练评估、结果对比与最佳模型保存。

## 项目结构

```
ai_project/
├── run_project.py              # 主入口：依次调用预处理与训练脚本
├── scripts/
│   ├── preprocess.py           # 数据加载、探索统计与可视化
│   └── train_model.py          # 三模型训练、评估、对比与保存
├── data/
│   ├── feature_distribution.png     # 四特征按类别的箱线分布图（运行生成）
│   └── correlation_heatmap.png      # 特征相关性热力图（运行生成）
├── models/
│   └── model_comparison.png         # 三模型准确率 / 训练时间对比图（运行生成）
└── .gitignore
```

> 训练产物 `*.pkl`（如 `decision_tree_model.pkl`）为二进制模型文件，已在 `.gitignore` 中排除，不纳入版本库，可由脚本重新生成。

## 环境依赖

Python 3，依赖如下：

```bash
pip3 install pandas numpy scikit-learn matplotlib seaborn joblib
```

## 快速开始

```bash
# 在项目根目录执行
python3 run_project.py
```

主脚本会自动：
1. 检查依赖是否齐全；
2. 运行 `scripts/preprocess.py` 完成数据探索与可视化；
3. 运行 `scripts/train_model.py` 完成训练、评估与结果输出。

也可单独运行：

```bash
python3 scripts/preprocess.py
python3 scripts/train_model.py
```

## 方法说明

- **数据集**：scikit-learn 内置 `load_iris()`，150 个样本、4 个特征、3 个类别（setosa / versicolor / virginica）。
- **划分**：`train_test_split` 按 7:3 分层划分（`stratify`），`random_state=42`。
- **对比模型**：
  | 模型 | 配置 |
  |---|---|
  | Decision Tree | `max_depth=3` |
  | Random Forest | `n_estimators=100` |
  | SVM | RBF 核 |
- **评估指标**：测试集准确率、5 折交叉验证均值与标准差、训练耗时，并输出分类报告与混淆矩阵。
- **产物**：特征分布箱线图、相关性热力图、模型性能对比图；最佳模型用 `joblib` 保存为 `.pkl`。

## 协作流程（Pull Request）

本仓库不直接向 `main` 推送，采用分支 + PR 跟踪改动：

```bash
# 1. 基于最新 main 新建功能分支
git checkout main && git pull origin main
git checkout -b feature/xxx

# 2. 提交改动
git add .
git commit -m '说明改动'

# 3. 推送分支并在 GitHub 发起 Pull Request 合并到 main
git push -u origin feature/xxx
```

## 环境

- 运行环境：WSL2 / openEuler 24.03，Python 3
- 版本控制：Git，远程 `git@github.com:moxiaohei/ai_project.git`
