import os
import json
import pickle
import pandas as pd
from datetime import datetime
import sys


def save_backtest_results(
    param_dict,
    summary,
    result_df=None,
    model=None,
    base_dir="/home/jovyan/work/tactics_demo/delta/backtest_result",
):
    """
    保存回测结果到结构化文件夹中

    参数:
    - param_dict: 策略参数字典
    - summary: 回测摘要统计字典
    - result_df: 每日回测结果的DataFrame (可选)
    - model: 训练好的模型 (可选)
    - base_dir: 基础保存目录

    返回:
    - 保存的文件夹路径
    """

    instrument_id = param_dict.get("instrument_id", "unknown")
    strategy_name = param_dict.get("name", "strategy")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    folder_name = f"{instrument_id}_{strategy_name}_{timestamp}"
    result_dir = os.path.join(base_dir, folder_name)
    os.makedirs(result_dir, exist_ok=True)

    print(f"保存回测结果到: {result_dir}")

    param_file = os.path.join(result_dir, "parameters.json")
    with open(param_file, "w", encoding="utf-8") as f:
        serializable_params = {}
        for key, value in param_dict.items():
            try:
                json.dumps({key: value})
                serializable_params[key] = value
            except (TypeError, ValueError):
                serializable_params[key] = str(value)

        json.dump(serializable_params, f, indent=2, ensure_ascii=False)

    # 4. 保存summary为JSON
    summary_file = os.path.join(result_dir, "summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # 5. 保存result_df为CSV和pickle (如果提供)
    if result_df is not None:
        # CSV格式
        csv_file = os.path.join(result_dir, "daily_results.csv")
        result_df.to_csv(csv_file, index=False, encoding="utf-8")

        # Pickle格式
        pkl_file = os.path.join(result_dir, "daily_results.pkl")
        with open(pkl_file, "wb") as f:
            pickle.dump(result_df, f)

    # 6. 保存模型 (如果提供)
    if model is not None:
        model_file = os.path.join(result_dir, "model.joblib")
        try:
            import joblib

            joblib.dump(model, model_file)
        except ImportError:
            # 如果joblib不可用，使用pickle
            model_file = os.path.join(result_dir, "model.pkl")
            with open(model_file, "wb") as f:
                pickle.dump(model, f)

    # 7. 创建README文件
    readme_file = os.path.join(result_dir, "README.md")
    with open(readme_file, "w", encoding="utf-8") as f:
        f.write(f"# 回测结果: {strategy_name}\n\n")
        f.write(f"**标的**: {instrument_id}\n")
        f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## 策略参数\n")
        f.write("```json\n")
        f.write(json.dumps(serializable_params, indent=2, ensure_ascii=False))
        f.write("\n```\n\n")

        f.write("## 回测摘要\n")
        f.write("| 指标 | 值 |\n")
        f.write("|------|-----|\n")
        for key, value in summary.items():
            f.write(f"| {key} | {value} |\n")

        f.write("\n## 文件说明\n")
        f.write("- `parameters.json`: 策略参数\n")
        f.write("- `summary.json`: 回测摘要统计\n")
        if result_df is not None:
            f.write("- `daily_results.csv`: 每日回测结果(CSV格式)\n")
            f.write("- `daily_results.pkl`: 每日回测结果(Pickle格式)\n")
        if model is not None:
            f.write("- `model.joblib` 或 `model.pkl`: 训练好的模型\n")

    # 8. 返回保存的文件夹路径
    return result_dir


def load_backtest_results(result_dir):
    """
    从保存的文件夹中加载回测结果

    参数:
    - result_dir: 结果文件夹路径

    返回:
    - dict: 包含加载的数据
    """
    results = {}

    # 1. 加载参数
    param_file = os.path.join(result_dir, "parameters.json")
    if os.path.exists(param_file):
        with open(param_file, "r", encoding="utf-8") as f:
            results["param_dict"] = json.load(f)

    # 2. 加载摘要
    summary_file = os.path.join(result_dir, "summary.json")
    if os.path.exists(summary_file):
        with open(summary_file, "r", encoding="utf-8") as f:
            results["summary"] = json.load(f)

    # 3. 加载每日结果
    csv_file = os.path.join(result_dir, "daily_results.csv")
    pkl_file = os.path.join(result_dir, "daily_results.pkl")

    if os.path.exists(csv_file):
        results["result_df"] = pd.read_csv(csv_file)
    elif os.path.exists(pkl_file):
        with open(pkl_file, "rb") as f:
            results["result_df"] = pickle.load(f)

    # 4. 加载模型
    model_joblib = os.path.join(result_dir, "model.joblib")
    model_pkl = os.path.join(result_dir, "model.pkl")

    if os.path.exists(model_joblib):
        try:
            import joblib

            results["model"] = joblib.load(model_joblib)
        except ImportError:
            pass
    elif os.path.exists(model_pkl):
        with open(model_pkl, "rb") as f:
            results["model"] = pickle.load(f)

    return results


def compare_results(result_dirs, metrics=["累计总盈亏", "胜率(天)%", "盈亏比(日均)"]):
    """
    比较多个回测结果

    参数:
    - result_dirs: 结果文件夹路径列表
    - metrics: 要比较的指标列表

    返回:
    - DataFrame: 比较结果
    """
    comparison_data = []

    for result_dir in result_dirs:
        try:
            results = load_backtest_results(result_dir)
            if "summary" in results:
                row = {"策略文件夹": os.path.basename(result_dir)}
                row.update(results["summary"])
                comparison_data.append(row)
        except Exception as e:
            print(f"加载 {result_dir} 失败: {e}")

    if comparison_data:
        df = pd.DataFrame(comparison_data)
        # 只保留指定的指标
        if metrics:
            columns_to_keep = ["策略文件夹"] + [m for m in metrics if m in df.columns]
            df = df[columns_to_keep]
        return df
    else:
        return pd.DataFrame()


def list_backtest_results(
    base_dir="/home/jovyan/work/tactics_demo/delta/backtest_result",
    instrument_id=None,
    strategy_name=None,
):
    """
    列出所有回测结果文件夹

    参数:
    - base_dir: 基础目录
    - instrument_id: 过滤标的ID
    - strategy_name: 过滤策略名称

    返回:
    - list: 结果文件夹路径列表
    """
    if not os.path.exists(base_dir):
        return []

    result_dirs = []
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path):
            # 检查是否包含必要的文件
            param_file = os.path.join(item_path, "parameters.json")
            summary_file = os.path.join(item_path, "summary.json")

            if os.path.exists(param_file) and os.path.exists(summary_file):
                # 应用过滤
                if instrument_id or strategy_name:
                    try:
                        with open(param_file, "r", encoding="utf-8") as f:
                            params = json.load(f)

                        match = True
                        if (
                            instrument_id
                            and params.get("instrument_id") != instrument_id
                        ):
                            match = False
                        if strategy_name and params.get("name") != strategy_name:
                            match = False

                        if match:
                            result_dirs.append(item_path)
                    except:
                        continue
                else:
                    result_dirs.append(item_path)

    # 按修改时间排序 (最新的在前)
    result_dirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return result_dirs


def delete_backtest_results_by_date(
    date_str,
    base_dir="/home/jovyan/work/tactics_demo/delta/backtest_result",
    dry_run=True,
):
    """
    删除指定日期的回测结果文件夹

    参数:
    - date_str: 日期字符串，格式为 "YYYYMMDD" 或 "YYYYMMDD_HHMMSS"
    - base_dir: 基础目录
    - dry_run: 如果为True，只显示将要删除的文件而不实际删除

    返回:
    - list: 被删除的文件夹路径列表
    """
    if not os.path.exists(base_dir):
        print(f"目录不存在: {base_dir}")
        return []

    deleted_dirs = []

    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path):
            # 检查文件夹名称是否包含指定日期
            # 文件夹命名模式: {instrument_id}_{strategy_name}_{YYYYMMDD_HHMMSS}
            if date_str in item:
                # 检查是否包含必要的回测结果文件
                param_file = os.path.join(item_path, "parameters.json")
                summary_file = os.path.join(item_path, "summary.json")

                if os.path.exists(param_file) and os.path.exists(summary_file):
                    if dry_run:
                        print(f"[DRY RUN] 将删除: {item_path}")
                    else:
                        try:
                            import shutil

                            shutil.rmtree(item_path)
                            print(f"已删除: {item_path}")
                            deleted_dirs.append(item_path)
                        except Exception as e:
                            print(f"删除失败 {item_path}: {e}")

    if dry_run:
        print(f"\n[DRY RUN] 总共将删除 {len(deleted_dirs)} 个文件夹")
    else:
        print(f"\n总共删除了 {len(deleted_dirs)} 个文件夹")

    return deleted_dirs


def delete_backtest_results_by_instrument_date(
    instrument_id,
    date_str,
    base_dir="/home/jovyan/work/tactics_demo/delta/backtest_result",
    dry_run=True,
):
    """
    删除指定标的和日期的回测结果文件夹

    参数:
    - instrument_id: 标的ID
    - date_str: 日期字符串，格式为 "YYYYMMDD" 或 "YYYYMMDD_HHMMSS"
    - base_dir: 基础目录
    - dry_run: 如果为True，只显示将要删除的文件而不实际删除

    返回:
    - list: 被删除的文件夹路径列表
    """
    if not os.path.exists(base_dir):
        print(f"目录不存在: {base_dir}")
        return []

    deleted_dirs = []

    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path):
            # 检查文件夹名称是否以instrument_id开头且包含指定日期
            # 文件夹命名模式: {instrument_id}_{strategy_name}_{YYYYMMDD_HHMMSS}
            if item.startswith(f"{instrument_id}_") and date_str in item:
                # 检查是否包含必要的回测结果文件
                param_file = os.path.join(item_path, "parameters.json")
                summary_file = os.path.join(item_path, "summary.json")

                if os.path.exists(param_file) and os.path.exists(summary_file):
                    if dry_run:
                        print(f"[DRY RUN] 将删除: {item_path}")
                    else:
                        try:
                            import shutil

                            shutil.rmtree(item_path)
                            print(f"已删除: {item_path}")
                            deleted_dirs.append(item_path)
                        except Exception as e:
                            print(f"删除失败 {item_path}: {e}")

    if dry_run:
        print(f"\n[DRY RUN] 总共将删除 {len(deleted_dirs)} 个文件夹")
    else:
        print(f"\n总共删除了 {len(deleted_dirs)} 个文件夹")

    return deleted_dirs
