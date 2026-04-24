from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_curve,
    auc,
    average_precision_score,
)
import matplotlib.pyplot as plt
import numpy as np

from .models.factory import ModelFactory


def train_model(X_train, y_train, X_valid, y_valid, param_dict, feature_names=None):
    """训练模型

    Args:
        X_train: 训练特征
        y_train: 训练标签
        X_valid: 验证特征
        y_valid: 验证标签
        param_dict: 参数字典，包含model_type等参数
        feature_names: 特征名称列表，如果为None则使用默认名称

    Returns:
        训练好的模型实例
    """
    # 获取模型类型，默认为xgboost以保持向后兼容
    model_type = param_dict.get("model_type", "xgboost")

    # 获取模型特定参数
    model_params = param_dict.get("model_params", {})

    # 处理可能的元组情况（Jupyter notebook中字典末尾的逗号会创建元组）
    if (
        isinstance(model_params, tuple)
        and len(model_params) == 1
        and isinstance(model_params[0], dict)
    ):
        model_params = model_params[0]

    print(f"训练 {model_type} 模型...")

    # 使用模型工厂创建模型
    model = ModelFactory.create_model(model_type, model_params)

    # 设置特征名称（如果提供了）
    if feature_names is not None and hasattr(model, "set_feature_names"):
        model.set_feature_names(feature_names)

    # 训练模型
    model.fit(X_train, y_train, X_valid, y_valid)

    # 打印特征重要性
    if hasattr(model, "get_feature_importance"):
        importance = model.get_feature_importance()
        if not importance.empty:
            print("\n特征重要性（前10个）:")
            print(importance.head(10))

    # 打印详细的XGBoost特征重要性（gain, weight, cover）
    if model_type == "xgboost" and hasattr(model, "get_xgboost_importance"):
        print("\nXGBoost特征重要性详情:")
        importance_df = model.get_xgboost_importance()
        if not importance_df.empty:
            print("\nGain重要性排名:")
            print(
                importance_df[["feature", "gain"]]
                .sort_values("gain", ascending=False)
                .head(20)
            )
            print("\nWeight重要性排名:")
            print(
                importance_df[["feature", "weight"]]
                .sort_values("weight", ascending=False)
                .head(20)
            )
            print("\nCover重要性排名:")
            print(
                importance_df[["feature", "cover"]]
                .sort_values("cover", ascending=False)
                .head(20)
            )

    return model


def evaluate_model(model, X_test, y_test, show_plots=False):
    """评估模型性能

    Args:
        model: 模型实例
        X_test: 测试特征
        y_test: 测试标签
        show_plots: 是否展示图表，默认为False

    Returns:
        准确率
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"测试集准确率: {accuracy:.4f}")
    print("\n分类报告:")
    print(classification_report(y_test, y_pred))

    # 添加PR曲线和概率分布图
    if hasattr(model, "predict_proba"):
        y_pred_proba_df = model.predict_proba(X_test)
        # 获取正类的概率（第二列）
        y_pred_proba = (
            y_pred_proba_df.iloc[:, 1].values
            if hasattr(y_pred_proba_df, "iloc")
            else y_pred_proba_df[:, 1]
        )
        precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
        pr_auc = auc(recall, precision)
        avg_precision = average_precision_score(y_test, y_pred_proba)

        print(f"\nPR曲线AUC: {pr_auc:.4f}")
        print(f"平均精度 (AP): {avg_precision:.4f}")

        # 计算预测概率的统计信息
        print(f"\n预测概率统计:")
        print(f"  均值: {y_pred_proba.mean():.4f}")
        print(f"  标准差: {y_pred_proba.std():.4f}")
        print(f"  最小值: {y_pred_proba.min():.4f}")
        print(f"  25%分位数: {np.percentile(y_pred_proba, 25):.4f}")
        print(f"  中位数: {np.median(y_pred_proba):.4f}")
        print(f"  75%分位数: {np.percentile(y_pred_proba, 75):.4f}")
        print(f"  最大值: {y_pred_proba.max():.4f}")

        # 按真实标签分组统计
        y_true_0 = y_pred_proba[y_test == 0]
        y_true_1 = y_pred_proba[y_test == 1]
        print(f"\n按真实标签分组的预测概率统计:")
        print(
            f"  标签0 (负类): 均值={y_true_0.mean():.4f}, 标准差={y_true_0.std():.4f}"
        )
        print(
            f"  标签1 (正类): 均值={y_true_1.mean():.4f}, 标准差={y_true_1.std():.4f}"
        )

        if show_plots:
            # 创建包含两个子图的图形
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))

            # 子图1: PR曲线
            ax1 = axes[0]
            ax1.plot(
                recall,
                precision,
                "b-",
                linewidth=2,
                label=f"PR curve (AUC = {pr_auc:.3f})",
            )
            ax1.plot([0, 1], [1, 0], "k--", alpha=0.5, label="Random")
            ax1.set_xlabel("Recall")
            ax1.set_ylabel("Precision")
            ax1.set_title("Precision-Recall Curve")
            ax1.grid(True, alpha=0.3)
            ax1.legend(loc="best")

            # 子图2: 预测概率分布图
            ax2 = axes[1]

            # 绘制直方图
            bins = np.linspace(0, 1, 21)
            ax2.hist(
                y_true_0,
                bins=bins,
                alpha=0.5,
                label="True Label 0",
                color="red",
                edgecolor="black",
            )
            ax2.hist(
                y_true_1,
                bins=bins,
                alpha=0.5,
                label="True Label 1",
                color="blue",
                edgecolor="black",
            )

            # 添加垂直线表示阈值0.5
            ax2.axvline(
                x=0.5,
                color="green",
                linestyle="--",
                linewidth=2,
                alpha=0.7,
                label="Threshold 0.5",
            )

            ax2.set_xlabel("Predicted Probability")
            ax2.set_ylabel("Frequency")
            ax2.set_title("Predicted Probability Distribution by True Label")
            ax2.legend(loc="best")
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show()

    return accuracy


def save_model(model, filename):
    """保存模型

    Args:
        model: 模型实例
        filename: 保存文件名
    """
    model.save(filename)
    print(f"模型已保存到: {filename}")


def load_model(filename, model_type="xgboost", model_params=None):
    """加载模型

    Args:
        filename: 模型文件名
        model_type: 模型类型（用于创建模型实例）
        model_params: 模型参数

    Returns:
        加载的模型实例
    """
    # 创建模型实例
    model = ModelFactory.create_model(model_type, model_params)

    # 加载模型权重
    model.load(filename)
    print(f"模型已从 {filename} 加载")

    return model


def get_trade_dates():
    trade_dates = ['20250901', '20250903', '20250905', '20250909', '20250911', '20250915', '20250917', '20250919', '20250923', '20250925', '20250929', '20251009', '20251013', '20251015', '20251017', '20251021', '20251023', '20251027', '20251029', '20251031', '20251103', '20251105', '20251107', '20251111', '20251113', '20251117', '20251119', '20251121', '20251125', '20251127', '20251201', '20251203', '20251205', '20251209', '20251211', '20251215', '20251217', '20251219', '20251223', '20251225', '20251229', '20251231', '20260105', '20260107', '20260109', '20260113', '20260115', '20260119', '20260121', '20260123', '20260127', '20260129', '20260203', '20260205', '20260209', '20260211', '20260213', '20260225', '20260227', '20260303', '20260305', '20260309', '20260311', '20260313', '20260317', '20260319', '20260323', '20260325', '20260327', '20260331', '20260401', '20260403', '20260407', '20260409']
    return trade_dates


def split_dates(trade_dates, train_days=35, valid_days=9, test_days=10):
    train_dates = trade_dates[:train_days]
    valid_dates = trade_dates[train_days : train_days + valid_days]
    test_dates = trade_dates[
        train_days + valid_days : train_days + valid_days + test_days
    ]

    print(f"训练集: {train_dates[0]} ~ {train_dates[-1]} ({len(train_dates)}天)")
    print(f"验证集: {valid_dates[0]} ~ {valid_dates[-1]} ({len(valid_dates)}天)")
    print(f"测试集: {test_dates[0]} ~ {test_dates[-1]} ({len(test_dates)}天)")

    return train_dates, valid_dates, test_dates


def split_dates_by_range(
    trade_dates,
    train_start=None,
    train_end=None,
    valid_start=None,
    valid_end=None,
    test_start=None,
    test_end=None,
):
    """按日期范围分割交易日

    Args:
        trade_dates: 完整的交易日列表
        train_start: 训练集开始日期 (包含)
        train_end: 训练集结束日期 (包含)
        valid_start: 验证集开始日期 (包含)
        valid_end: 验证集结束日期 (包含)
        test_start: 测试集开始日期 (包含)
        test_end: 测试集结束日期 (包含)

    Returns:
        train_dates, valid_dates, test_dates
    """

    def filter_dates_by_range(date_list, start_date, end_date):
        """根据开始和结束日期过滤日期列表"""
        if start_date is None and end_date is None:
            return []

        filtered = []
        for date in date_list:
            if start_date and date < start_date:
                continue
            if end_date and date > end_date:
                continue
            filtered.append(date)
        return filtered

    # 过滤出各个集合的日期
    train_dates = filter_dates_by_range(trade_dates, train_start, train_end)
    valid_dates = filter_dates_by_range(trade_dates, valid_start, valid_end)
    test_dates = filter_dates_by_range(trade_dates, test_start, test_end)

    # 打印结果
    if train_dates:
        print(f"训练集: {train_dates[0]} ~ {train_dates[-1]} ({len(train_dates)}天)")
    else:
        print("训练集: 无")

    if valid_dates:
        print(f"验证集: {valid_dates[0]} ~ {valid_dates[-1]} ({len(valid_dates)}天)")
    else:
        print("验证集: 无")

    if test_dates:
        print(f"测试集: {test_dates[0]} ~ {test_dates[-1]} ({len(test_dates)}天)")
    else:
        print("测试集: 无")

    return train_dates, valid_dates, test_dates


def split_dates_randomly(
    trade_dates, test_days_min=23, valid_days=0, random_seed=None, shuffle=True
):
    """随机划分日期为训练集和测试集（默认不要验证集）

    Args:
        trade_dates: 完整的交易日列表
        test_days_min: 测试集最少天数，默认为23天
        valid_days: 验证集天数，默认为0（不要验证集）
        random_seed: 随机种子，用于可重复性
        shuffle: 是否打乱日期顺序，默认为True

    Returns:
        train_dates, valid_dates, test_dates

    Raises:
        ValueError: 如果总天数不足或测试集天数不足
    """
    import random

    # 设置随机种子
    if random_seed is not None:
        random.seed(random_seed)

    # 检查总天数是否足够
    total_days = len(trade_dates)
    required_days = test_days_min + valid_days

    if total_days <= required_days:
        raise ValueError(
            f"总天数({total_days})不足，至少需要{required_days + 1}天 "
            f"(测试集最少{test_days_min}天 + 验证集{valid_days}天 + 训练集至少1天)"
        )

    # 如果需要打乱，创建日期的副本并打乱
    if shuffle:
        dates_to_shuffle = trade_dates.copy()
        random.shuffle(dates_to_shuffle)
    else:
        dates_to_shuffle = trade_dates

    # 划分日期
    # 首先分配测试集
    test_dates = dates_to_shuffle[:test_days_min]

    # 然后分配验证集（如果有）
    if valid_days > 0:
        valid_dates = dates_to_shuffle[test_days_min : test_days_min + valid_days]
        # 剩余为训练集
        train_dates = dates_to_shuffle[test_days_min + valid_days :]
    else:
        valid_dates = []
        # 剩余为训练集
        train_dates = dates_to_shuffle[test_days_min:]

    # 确保训练集不为空
    if not train_dates:
        raise ValueError("训练集为空，请减少测试集或验证集的天数")

    # 排序日期（可选，保持日期顺序）
    train_dates.sort()
    if valid_dates:
        valid_dates.sort()
    test_dates.sort()

    # 打印结果
    print(
        f"随机划分结果（随机种子: {random_seed if random_seed is not None else '未设置'}）:"
    )
    print(f"训练集: {train_dates[0]} ~ {train_dates[-1]} ({len(train_dates)}天)")

    if valid_dates:
        print(f"验证集: {valid_dates[0]} ~ {valid_dates[-1]} ({len(valid_dates)}天)")
    else:
        print("验证集: 无")

    print(f"测试集: {test_dates[0]} ~ {test_dates[-1]} ({len(test_dates)}天)")

    return train_dates, valid_dates, test_dates
