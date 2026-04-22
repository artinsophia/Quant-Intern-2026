import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
import sys

sys.path.append("/home/jovyan/work/tactics_demo/tools")
from backtest_quick import backtest_quick


class RollingValidationConfig:
    """滚动验证配置类"""

    def __init__(
        self,
        step_size: int = 60,
        signal_window: int = 300,
        label_window: int = 60,
        min_samples: int = 100,
    ):
        """
        初始化滚动验证配置

        Args:
            step_size: 滚动步长（秒）
            signal_window: 信号计算窗口（秒）
            label_window: 标签计算窗口（秒）
            min_samples: 最小样本数要求
        """
        self.step_size = step_size
        self.signal_window = signal_window
        self.label_window = label_window
        self.min_samples = min_samples

    def validate(self) -> bool:
        """验证配置参数是否合理"""
        if self.step_size <= 0:
            raise ValueError("步长必须大于0")
        if self.signal_window <= 0:
            raise ValueError("信号计算窗口必须大于0")
        if self.label_window <= 0:
            raise ValueError("标签计算窗口必须大于0")
        if self.min_samples <= 0:
            raise ValueError("最小样本数必须大于0")
        return True


class BaseAlphaFactor:
    """Alpha因子基类"""

    def __init__(
        self,
        name: str,
        config: RollingValidationConfig,
        param_dict: Dict[str, Any] = None,
    ):
        """
        初始化Alpha因子

        Args:
            name: 因子名称
            config: 滚动验证配置
            param_dict: 参数字典，用于传递因子特定参数
        """
        self.name = name
        self.config = config
        self.signal_cache = {}

        # 设置默认参数
        if param_dict:
            self.__dict__.update(param_dict)

    def calculate_signal(
        self, snap_data: Dict[str, Any], lookback_window: List[Dict[str, Any]]
    ) -> float:
        """
        计算因子信号（需子类实现）

        Args:
            snap_data: 当前快照数据
            lookback_window: 回看窗口内的历史快照数据

        Returns:
            float: 因子信号值
        """
        raise NotImplementedError("子类必须实现calculate_signal方法")

    def calculate_label(
        self, current_snap: Dict[str, Any], future_window: List[Dict[str, Any]]
    ) -> float:
        """
        计算标签（需子类实现）

        Args:
            current_snap: 当前快照数据
            future_window: 未来窗口内的快照数据

        Returns:
            float: 标签值
        """
        raise NotImplementedError("子类必须实现calculate_label方法")

    def validate_data(self, snap_data: Dict[str, Any]) -> bool:
        """
        验证数据是否有效

        Args:
            snap_data: 快照数据

        Returns:
            bool: 数据是否有效
        """
        required_fields = ["time_mark", "price_last", "bid_book", "ask_book"]
        for field in required_fields:
            if field not in snap_data:
                return False
        return True


class RollingValidationEngine:
    """滚动验证引擎"""

    def __init__(self, config: RollingValidationConfig):
        """
        初始化滚动验证引擎

        Args:
            config: 滚动验证配置
        """
        self.config = config
        self.results = {}

    def single_day_validation(
        self,
        factor: BaseAlphaFactor,
        snap_list: List[Dict[str, Any]],
        instrument_id: str,
        trade_ymd: str,
    ) -> Dict[str, Any]:
        """
        单日因子检验

        Args:
            factor: Alpha因子实例
            snap_list: 单日快照数据列表
            instrument_id: 标的ID
            trade_ymd: 交易日

        Returns:
            Dict: 检验结果
        """
        if len(snap_list) < self.config.min_samples:
            return {
                "error": f"数据量不足: {len(snap_list)} < {self.config.min_samples}"
            }

        signals = []
        labels = []
        timestamps = []

        total_steps = (
            len(snap_list) - self.config.signal_window - self.config.label_window
        )

        for i in range(0, total_steps, self.config.step_size):
            signal_idx = i + self.config.signal_window
            label_idx = signal_idx + self.config.label_window

            if label_idx >= len(snap_list):
                break

            current_snap = snap_list[signal_idx]

            if not factor.validate_data(current_snap):
                continue

            lookback_window = snap_list[i:signal_idx]
            future_window = snap_list[signal_idx:label_idx]

            try:
                signal = factor.calculate_signal(current_snap, lookback_window)
                label = factor.calculate_label(current_snap, future_window, lookback_window)

                signals.append(signal)
                labels.append(label)
                timestamps.append(current_snap["time_mark"])
            except Exception as e:
                continue

        if len(signals) < self.config.min_samples:
            return {
                "error": f"有效样本不足: {len(signals)} < {self.config.min_samples}"
            }

        signals_array = np.array(signals)
        labels_array = np.array(labels)

        ic = np.corrcoef(signals_array, labels_array)[0, 1] if len(signals) > 1 else 0
        rank_ic = pd.Series(signals_array).corr(
            pd.Series(labels_array), method="spearman"
        )

        result = {
            "instrument_id": instrument_id,
            "trade_ymd": trade_ymd,
            "factor_name": factor.name,
            "total_samples": len(snap_list),
            "valid_samples": len(signals),
            "ic": float(ic),
            "rank_ic": float(rank_ic),
            "signal_mean": float(np.mean(signals_array)),
            "signal_std": float(np.std(signals_array)),
            "label_mean": float(np.mean(labels_array)),
            "label_std": float(np.std(labels_array)),
            "timestamps": timestamps,
            "signals": signals,
            "labels": labels,
        }

        self.results[f"{instrument_id}_{trade_ymd}"] = result
        return result

    def multi_day_validation(
        self,
        factor: BaseAlphaFactor,
        instrument_id: str,
        start_ymd: str,
        end_ymd: str,
        snap_loader: Callable[[str, str], List[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        """
        多日批量检测

        Args:
            factor: Alpha因子实例
            instrument_id: 标的ID
            start_ymd: 开始日期
            end_ymd: 结束日期
            snap_loader: 快照数据加载函数

        Returns:
            Dict: 多日检验结果汇总
        """
        import datetime

        start_date = datetime.datetime.strptime(start_ymd, "%Y%m%d")
        end_date = datetime.datetime.strptime(end_ymd, "%Y%m%d")

        daily_results = []
        current_date = start_date

        while current_date <= end_date:
            trade_ymd = current_date.strftime("%Y%m%d")

            try:
                snap_list = snap_loader(instrument_id, trade_ymd)
                if snap_list:
                    result = self.single_day_validation(
                        factor, snap_list, instrument_id, trade_ymd
                    )
                    if "error" not in result:
                        daily_results.append(result)
            except (Exception,SystemExit) as e:
                pass

            current_date += datetime.timedelta(days=1)

        ics = [r["ic"] for r in daily_results]
        rank_ics = [r["rank_ic"] for r in daily_results]

        summary = {
            "instrument_id": instrument_id,
            "start_date": start_ymd,
            "end_date": end_ymd,
            "factor_name": factor.name,
            "total_days": len(daily_results),
            "ic_mean": float(np.mean(ics)),
            "ic_std": float(np.std(ics)),
            "ic_ir": float(np.mean(ics) / np.std(ics)) if np.std(ics) > 0 else 0,
            "rank_ic_mean": float(np.mean(rank_ics)),
            "rank_ic_std": float(np.std(rank_ics)),
            "rank_ic_ir": float(np.mean(rank_ics) / np.std(rank_ics))
            if np.std(rank_ics) > 0
            else 0,
            "ic_positive_ratio": float(np.sum(np.array(ics) > 0) / len(ics)),
            "daily_results": daily_results,
        }

        return summary

    def backtest_validation(
        self,
        factor: BaseAlphaFactor,
        instrument_id: str,
        trade_ymd: str,
        snap_list: List[Dict[str, Any]],
        position_limit: int = 1,
        transaction_cost: float = 0.0001,
    ) -> Dict[str, Any]:
        """
        回测验证

        Args:
            factor: Alpha因子实例
            instrument_id: 标的ID
            trade_ymd: 交易日
            snap_list: 快照数据列表
            position_limit: 仓位限制
            transaction_cost: 交易成本

        Returns:
            Dict: 回测结果
        """

        class FactorStrategy:
            def __init__(self, factor, config):
                self.factor = factor
                self.config = config
                self.position_last = 0
                self.signals = []
                self.timestamps = []

            def on_snap(self, snap):
                if not self.factor.validate_data(snap):
                    return

                idx = len(self.signals)
                if idx < self.config.signal_window:
                    return

                lookback_window = self.signals[
                    max(0, idx - self.config.signal_window) : idx
                ]
                signal = self.factor.calculate_signal(snap, lookback_window)

                if signal > 0.5 and self.position_last < position_limit:
                    self.position_last = 1
                elif signal < -0.5 and self.position_last > -position_limit:
                    self.position_last = -1
                elif abs(signal) < 0.2:
                    self.position_last = 0

                self.signals.append(signal)
                self.timestamps.append(snap["time_mark"])

        strategy = FactorStrategy(factor, self.config)

        try:
            result = backtest_quick(snap_list, strategy)

            backtest_summary = {
                "instrument_id": instrument_id,
                "trade_ymd": trade_ymd,
                "factor_name": factor.name,
                "total_pnl": result.get("total_pnl", 0),
                "total_return": result.get("total_return", 0),
                "sharpe_ratio": result.get("sharpe_ratio", 0),
                "max_drawdown": result.get("max_drawdown", 0),
                "win_rate": result.get("win_rate", 0),
                "total_trades": result.get("total_trades", 0),
            }

            return backtest_summary

        except Exception as e:
            return {"error": f"回测失败: {e}"}

