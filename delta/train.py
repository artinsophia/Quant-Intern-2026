from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_curve,
    auc,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import sys
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from .models.factory import ModelFactory

sys.path.append("/home/jovyan/base_demo")
import base_tool

RECOMMENDED_TRAINING_PRESETS = {
    "recent_6d": {
        "strategy": "recent",
        "train_days": 6,
    },
    "volatility_8d": {
        "strategy": "volatility_stratified",
        "train_days": 8,
        "n_bins": 5,
        "random_seed": 42,
    },
    "volatility_12d": {
        "strategy": "volatility_stratified",
        "train_days": 12,
        "n_bins": 5,
        "random_seed": 42,
    },
    "volatility_16d": {
        "strategy": "volatility_stratified",
        "train_days": 16,
        "n_bins": 5,
        "random_seed": 42,
    },
}


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


def load_daily_sample_cache(
    dates,
    instrument_id,
    param_dict,
    feature_func,
    y_func,
):
    """按天加载样本，便于做基于日期的抽样实验。"""
    from .data_processing import TrainValidTest

    sample_cache = {}

    for date in dates:
        try:
            snap_list = base_tool.snap_list_load(instrument_id, date)
            if len(snap_list) < param_dict["x_window"] + param_dict["y_window"]:
                print(f"{date}: 数据不足，跳过")
                continue

            tv = TrainValidTest(snap_list, param_dict, feature_func, y_func)
            X_day, y_day = tv.samples()
            if not X_day:
                print(f"{date}: 无触发样本，跳过")
                continue

            feature_names = list(X_day[0].keys())
            X_day_array = np.array(
                [[row[col] for col in feature_names] for row in X_day], dtype=float
            )
            y_day_array = np.array(y_day)
            sample_cache[date] = {
                "X": X_day_array,
                "y": y_day_array,
                "feature_names": feature_names,
                "sample_count": len(y_day_array),
                "positive_rate": float(y_day_array.mean()) if len(y_day_array) else 0.0,
            }
            print(f"{date}: 产生 {len(y_day_array)} 个样本")
        except Exception as e:
            print(f"{date}: 加载失败 - {e}")

    return sample_cache


def concat_sample_cache(sample_cache, dates):
    """按日期顺序拼接缓存样本。"""
    available_dates = [date for date in dates if date in sample_cache]
    if not available_dates:
        return np.array([]), np.array([]), []

    X_parts = [sample_cache[date]["X"] for date in available_dates]
    y_parts = [sample_cache[date]["y"] for date in available_dates]
    feature_names = sample_cache[available_dates[0]]["feature_names"]
    X_total = np.vstack(X_parts)
    y_total = np.concatenate(y_parts)
    return X_total, y_total, feature_names


def summarize_daily_volatility(
    instrument_id,
    trade_dates,
    price_field="price_last",
):
    """计算每个交易日的波动率与基础统计。"""
    records = []

    for trade_ymd in trade_dates:
        try:
            snap_list = base_tool.snap_list_load(instrument_id, trade_ymd)
            prices = np.array(
                [
                    snap.get(price_field)
                    for snap in snap_list
                    if snap.get(price_field) is not None
                ],
                dtype=float,
            )
            prices = prices[np.isfinite(prices)]

            if len(prices) < 2:
                print(f"{trade_ymd}: 价格点不足，跳过波动率统计")
                continue

            returns = np.diff(prices) / prices[:-1]
            returns = returns[np.isfinite(returns)]
            realized_volatility = float(np.sqrt(np.sum(np.square(returns))))
            normalized_range = float((prices.max() - prices.min()) / prices[0])
            close_to_close_return = float(prices[-1] / prices[0] - 1)
            intraday_std = float(np.std(prices) / np.mean(prices))

            records.append(
                {
                    "trade_ymd": trade_ymd,
                    "n_snapshots": int(len(prices)),
                    "open_price": float(prices[0]),
                    "close_price": float(prices[-1]),
                    "close_to_close_return": close_to_close_return,
                    "realized_volatility": realized_volatility,
                    "intraday_std": intraday_std,
                    "normalized_range": normalized_range,
                }
            )
        except Exception as e:
            print(f"{trade_ymd}: 波动率统计失败 - {e}")

    stats_df = pd.DataFrame(records).sort_values("trade_ymd").reset_index(drop=True)
    return stats_df


def assign_volatility_bins(
    stats_df,
    vol_col="realized_volatility",
    n_bins=5,
):
    """基于波动率分位数给交易日打桶。"""
    if stats_df.empty:
        return stats_df.copy()

    df = stats_df.copy()
    unique_count = df[vol_col].nunique(dropna=True)
    if unique_count == 0:
        df["vol_bucket"] = 0
        return df

    actual_bins = max(1, min(int(n_bins), int(unique_count)))
    if actual_bins == 1:
        df["vol_bucket"] = 0
        return df

    df["vol_bucket"] = pd.qcut(
        df[vol_col],
        q=actual_bins,
        labels=False,
        duplicates="drop",
    )
    df["vol_bucket"] = df["vol_bucket"].astype(int)
    return df


def sample_dates_by_volatility(
    stats_df,
    sample_size,
    random_seed=42,
    min_per_bucket=1,
    vol_col="realized_volatility",
    n_bins=5,
):
    """按波动率桶做近似分层抽样，优先保证每桶有代表。"""
    if sample_size <= 0:
        return []
    if stats_df.empty:
        return []

    df = assign_volatility_bins(stats_df, vol_col=vol_col, n_bins=n_bins)
    rng = random.Random(random_seed)
    bucket_to_dates = {}
    for bucket, bucket_df in df.groupby("vol_bucket"):
        dates = bucket_df["trade_ymd"].tolist()
        rng.shuffle(dates)
        bucket_to_dates[bucket] = dates

    buckets = sorted(bucket_to_dates.keys())
    available_total = sum(len(dates) for dates in bucket_to_dates.values())
    target_size = min(sample_size, available_total)

    selected = []
    remaining = {bucket: dates.copy() for bucket, dates in bucket_to_dates.items()}

    if target_size >= len(buckets) * min_per_bucket:
        for bucket in buckets:
            for _ in range(min_per_bucket):
                if remaining[bucket] and len(selected) < target_size:
                    selected.append(remaining[bucket].pop())

    while len(selected) < target_size:
        active_buckets = [bucket for bucket in buckets if remaining[bucket]]
        if not active_buckets:
            break

        counts = np.array([len(remaining[bucket]) for bucket in active_buckets], dtype=float)
        probs = counts / counts.sum()
        bucket = rng.choices(active_buckets, weights=probs, k=1)[0]
        selected.append(remaining[bucket].pop())

    selected.sort()
    return selected


def select_recent_train_dates(candidate_dates, sample_size):
    """选择最近的若干个训练日，作为时序基线。"""
    if sample_size <= 0:
        return []
    return sorted(candidate_dates)[-sample_size:]


def select_train_dates(
    candidate_dates,
    strategy="recent",
    train_days=None,
    instrument_id=None,
    stats_df=None,
    random_seed=42,
    n_bins=5,
    vol_col="realized_volatility",
):
    """统一的训练日选择入口。"""
    available_dates = sorted(candidate_dates)
    if not available_dates:
        return []

    if train_days is None:
        raise ValueError("train_days 不能为空")

    if train_days <= 0:
        raise ValueError("train_days 必须大于 0")

    train_days = min(int(train_days), len(available_dates))

    if strategy == "recent":
        return select_recent_train_dates(available_dates, train_days)

    if strategy == "random":
        rng = random.Random(random_seed)
        return sorted(rng.sample(available_dates, train_days))

    if strategy == "volatility_stratified":
        if stats_df is None:
            if instrument_id is None:
                raise ValueError(
                    "volatility_stratified 需要提供 instrument_id 或 stats_df"
                )
            stats_df = summarize_daily_volatility(instrument_id, available_dates)

        stats_df = stats_df[stats_df["trade_ymd"].isin(available_dates)].reset_index(
            drop=True
        )
        return sample_dates_by_volatility(
            stats_df,
            sample_size=train_days,
            random_seed=random_seed,
            vol_col=vol_col,
            n_bins=n_bins,
        )

    raise ValueError(f"未知采样策略: {strategy}")


def select_train_dates_by_preset(
    preset_name,
    candidate_dates,
    instrument_id=None,
    stats_df=None,
):
    """按预设方案选择训练日。"""
    if preset_name not in RECOMMENDED_TRAINING_PRESETS:
        raise ValueError(
            f"未知预设: {preset_name}, 可选: {sorted(RECOMMENDED_TRAINING_PRESETS)}"
        )

    config = RECOMMENDED_TRAINING_PRESETS[preset_name]
    return select_train_dates(
        candidate_dates=candidate_dates,
        strategy=config["strategy"],
        train_days=config["train_days"],
        instrument_id=instrument_id,
        stats_df=stats_df,
        random_seed=config.get("random_seed", 42),
        n_bins=config.get("n_bins", 5),
    )


def train_model_with_dates(
    instrument_id,
    train_dates,
    valid_dates,
    param_dict,
    feature_func,
    y_func,
):
    """给定训练日和验证日，直接训练模型。"""
    all_dates = sorted(set(train_dates) | set(valid_dates))
    sample_cache = load_daily_sample_cache(
        all_dates,
        instrument_id,
        param_dict,
        feature_func,
        y_func,
    )

    X_train, y_train, feature_names = concat_sample_cache(sample_cache, train_dates)
    X_valid, y_valid, _ = concat_sample_cache(sample_cache, valid_dates)

    if len(y_train) == 0:
        raise ValueError("训练集为空，无法训练模型")
    if len(y_valid) == 0:
        raise ValueError("验证集为空，无法训练模型")

    model = train_model(
        X_train,
        y_train,
        X_valid,
        y_valid,
        param_dict,
        feature_names,
    )
    return model, sample_cache, feature_names


def train_model_with_preset(
    preset_name,
    instrument_id,
    candidate_train_dates,
    valid_dates,
    param_dict,
    feature_func,
    y_func,
    stats_df=None,
):
    """按预设训练方案直接训练模型。"""
    train_dates = select_train_dates_by_preset(
        preset_name=preset_name,
        candidate_dates=candidate_train_dates,
        instrument_id=instrument_id,
        stats_df=stats_df,
    )
    model, sample_cache, feature_names = train_model_with_dates(
        instrument_id=instrument_id,
        train_dates=train_dates,
        valid_dates=valid_dates,
        param_dict=param_dict,
        feature_func=feature_func,
        y_func=y_func,
    )
    return model, train_dates, sample_cache, feature_names


def evaluate_binary_predictions(y_true, y_pred, y_prob):
    """统一输出分类指标。"""
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "average_precision": float(average_precision_score(y_true, y_prob)),
    }

    precision_arr, recall_arr, _ = precision_recall_curve(y_true, y_prob)
    metrics["pr_auc"] = float(auc(recall_arr, precision_arr))
    metrics["positive_rate"] = float(np.mean(y_true))
    metrics["predicted_positive_rate"] = float(np.mean(y_pred))
    return metrics


def run_training_subset_experiment(
    instrument_id,
    train_candidate_dates,
    valid_dates,
    test_dates,
    param_dict,
    feature_func,
    y_func,
    train_day_grid,
    strategies=("recent", "random", "volatility_stratified"),
    repeats=3,
    random_seed=42,
    n_bins=5,
    vol_col="realized_volatility",
):
    """探索最少训练天数与效果之间的关系。"""
    all_dates = sorted(set(train_candidate_dates) | set(valid_dates) | set(test_dates))
    print("加载按天样本缓存...")
    sample_cache = load_daily_sample_cache(
        all_dates,
        instrument_id,
        param_dict,
        feature_func,
        y_func,
    )

    print("统计训练候选区间的日波动率...")
    stats_df = summarize_daily_volatility(instrument_id, train_candidate_dates)
    stats_df = assign_volatility_bins(stats_df, vol_col=vol_col, n_bins=n_bins)

    X_valid, y_valid, _ = concat_sample_cache(sample_cache, valid_dates)
    X_test, y_test, _ = concat_sample_cache(sample_cache, test_dates)
    if len(y_valid) == 0 or len(y_test) == 0:
        raise ValueError("验证集或测试集为空，无法开展训练天数实验")

    records = []
    train_candidate_dates = sorted(
        [date for date in train_candidate_dates if date in sample_cache]
    )
    stats_df = stats_df[stats_df["trade_ymd"].isin(train_candidate_dates)].reset_index(
        drop=True
    )
    max_available_days = len(train_candidate_dates)

    for train_days in train_day_grid:
        if train_days <= 0:
            continue
        if train_days > max_available_days:
            print(f"train_days={train_days}: 超过可用训练日数量，跳过")
            continue

        for strategy_name in strategies:
            run_count = repeats if strategy_name in {"random", "volatility_stratified"} else 1

            for repeat_idx in range(run_count):
                seed = random_seed + repeat_idx

                if strategy_name == "recent":
                    selected_dates = select_recent_train_dates(
                        train_candidate_dates, train_days
                    )
                elif strategy_name == "random":
                    rng = random.Random(seed)
                    selected_dates = sorted(rng.sample(train_candidate_dates, train_days))
                elif strategy_name == "volatility_stratified":
                    selected_dates = sample_dates_by_volatility(
                        stats_df,
                        sample_size=train_days,
                        random_seed=seed,
                        vol_col=vol_col,
                        n_bins=n_bins,
                    )
                else:
                    raise ValueError(f"未知采样策略: {strategy_name}")

                X_train, y_train, feature_names = concat_sample_cache(
                    sample_cache, selected_dates
                )
                if len(y_train) == 0:
                    print(
                        f"{strategy_name} train_days={train_days} repeat={repeat_idx}: 训练集为空，跳过"
                    )
                    continue

                model = train_model(
                    X_train,
                    y_train,
                    X_valid,
                    y_valid,
                    param_dict,
                    feature_names,
                )

                y_test_pred = model.predict(X_test)
                y_test_prob = model.predict_proba(X_test)
                y_test_prob = (
                    y_test_prob.iloc[:, 1].values
                    if hasattr(y_test_prob, "iloc")
                    else y_test_prob[:, 1]
                )

                metrics = evaluate_binary_predictions(y_test, y_test_pred, y_test_prob)
                selected_stats = stats_df[stats_df["trade_ymd"].isin(selected_dates)]

                records.append(
                    {
                        "strategy": strategy_name,
                        "train_days": int(train_days),
                        "repeat": int(repeat_idx),
                        "selected_dates": ",".join(selected_dates),
                        "train_samples": int(len(y_train)),
                        "train_positive_rate": float(np.mean(y_train)),
                        "selected_vol_mean": float(selected_stats[vol_col].mean())
                        if not selected_stats.empty
                        else np.nan,
                        "selected_vol_std": float(selected_stats[vol_col].std())
                        if len(selected_stats) > 1
                        else 0.0,
                        **metrics,
                    }
                )

    result_df = pd.DataFrame(records)
    if result_df.empty:
        return result_df, pd.DataFrame(), stats_df

    summary_df = (
        result_df.groupby(["strategy", "train_days"], as_index=False)
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
            ap_mean=("average_precision", "mean"),
            ap_std=("average_precision", "std"),
            pr_auc_mean=("pr_auc", "mean"),
            pr_auc_std=("pr_auc", "std"),
            train_samples_mean=("train_samples", "mean"),
        )
        .sort_values(["train_days", "strategy"])
        .reset_index(drop=True)
    )
    summary_df = summary_df.fillna(0.0)
    return result_df, summary_df, stats_df
