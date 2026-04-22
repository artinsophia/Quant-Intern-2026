#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
标签构造方法实现
1. 收益率标签构造方法
2. 三重屏障法标签构造方法（参考delta/data_processing.py中的create_y方法）
"""

import numpy as np
import math
from typing import Dict, List, Any
from rolling_validation import BaseAlphaFactor, RollingValidationConfig


class ReturnLabelFactor(BaseAlphaFactor):
    """收益率标签构造方法"""

    def __init__(
        self, config: RollingValidationConfig, param_dict: Dict[str, Any] = None
    ):
        """
        初始化收益率标签因子

        Args:
            config: 滚动验证配置
            param_dict: 参数字典，可包含以下参数：
                - return_type: 收益率类型，可选 'simple'（简单收益率）或 'log'（对数收益率）
                - normalize: 是否标准化收益率，默认False
                - clip_threshold: 收益率截断阈值，默认None（不截断）
        """
        # 设置默认参数
        default_params = {
            "return_type": "simple",
            "normalize": False,
            "clip_threshold": None,
        }

        # 合并参数
        merged_params = default_params.copy()
        if param_dict:
            merged_params.update(param_dict)

        super().__init__("return_label", config, merged_params)

        # 设置实例变量
        self.return_type = merged_params["return_type"]
        self.normalize = merged_params["normalize"]
        self.clip_threshold = merged_params["clip_threshold"]

    def calculate_signal(
        self, snap_data: Dict[str, Any], lookback_window: List[Dict[str, Any]]
    ) -> float:
        """
        计算信号（收益率标签因子不产生信号，返回0）

        Args:
            snap_data: 当前快照数据
            lookback_window: 回看窗口内的历史快照数据

        Returns:
            float: 信号值（始终返回0）
        """
        return 0.0

    def calculate_label(
        self, current_snap: Dict[str, Any], future_window: List[Dict[str, Any]], lookback_window=None
    ) -> float:
        """
        计算收益率标签

        Args:
            current_snap: 当前快照数据
            future_window: 未来窗口内的快照数据

        Returns:
            float: 收益率标签值
        """
        if not future_window:
            return 0.0

        # 获取起始价格
        start_price = current_snap.get("price_last", 0)
        if (
            start_price is None
            or start_price == 0
            or (isinstance(start_price, float) and math.isnan(start_price))
        ):
            return 0.0

        # 获取结束价格（未来窗口最后一个有效价格）
        end_price = 0
        for snap in reversed(future_window):
            price = snap.get("price_last", 0)
            if (
                price is not None
                and price != 0
                and not (isinstance(price, float) and math.isnan(price))
            ):
                end_price = price
                break

        if end_price == 0:
            return 0.0

        # 计算收益率
        if self.return_type == "log":
            # 对数收益率
            return_value = np.log(end_price / start_price)
        else:
            # 简单收益率
            return_value = (end_price - start_price) / start_price

        # 截断处理
        if self.clip_threshold is not None:
            return_value = np.clip(
                return_value, -self.clip_threshold, self.clip_threshold
            )

        # 标准化处理
        if self.normalize and len(future_window) > 1:
            # 计算未来窗口内的价格波动
            prices = []
            for snap in future_window:
                price = snap.get("price_last", 0)
                if (
                    price is not None
                    and price != 0
                    and not (isinstance(price, float) and math.isnan(price))
                ):
                    prices.append(price)

            if len(prices) > 1:
                returns = np.diff(prices) / prices[:-1]
                if len(returns) > 0:
                    std_return = np.std(returns)
                    if std_return > 0:
                        return_value = return_value / std_return

        return float(return_value)

    def validate_data(self, snap_data: Dict[str, Any]) -> bool:
        """
        验证数据是否有效

        Args:
            snap_data: 快照数据

        Returns:
            bool: 数据是否有效
        """
        if "price_last" not in snap_data:
            return False

        price = snap_data["price_last"]
        if (
            price is None
            or price == 0
            or (isinstance(price, float) and math.isnan(price))
        ):
            return False

        return True


import numpy as np
import math
from typing import Dict, List, Any, Optional
from rolling_validation import BaseAlphaFactor, RollingValidationConfig


class TripleBarrierLabelFactor(BaseAlphaFactor):
    """三重屏障法标签构造方法（基于过去窗口波动率）"""

    def __init__(
        self, config: RollingValidationConfig, param_dict: Dict[str, Any] = None
    ):
        """
        初始化三重屏障法标签因子

        Args:
            config: 滚动验证配置
            param_dict: 参数字典，可包含以下参数：
                - k_up: 上屏障倍数，默认3.0
                - k_down: 下屏障倍数，默认3.0
                - past_vol_window: 过去波动率计算窗口，默认900
                - category: 交易方向（1=多头，-1=空头，其他=返回原始三分类标签）
                - vol_type: 波动率计算方式，可选 'return_std' 或 'price_std'
                - return_type: 收益率类型，可选 'simple' 或 'log'
                - use_fixed_horizon_when_no_barrier: 若未来未触碰上下屏障，是否返回0（默认True）
        """
        default_params = {
            "k_up": 3.0,
            "k_down": 3.0,
            "past_vol_window": 900,
            "category": 0,   # 0 表示返回原始标签 -1/0/1
            "vol_type": "return_std",
            "return_type": "simple",
            "use_fixed_horizon_when_no_barrier": True,
        }

        merged_params = default_params.copy()
        if param_dict:
            merged_params.update(param_dict)

        super().__init__("triple_barrier_label", config, merged_params)

        self.k_up = merged_params["k_up"]
        self.k_down = merged_params["k_down"]
        self.past_vol_window = merged_params["past_vol_window"]
        self.category = merged_params["category"]
        self.vol_type = merged_params["vol_type"]
        self.return_type = merged_params["return_type"]
        self.use_fixed_horizon_when_no_barrier = merged_params[
            "use_fixed_horizon_when_no_barrier"
        ]

    def calculate_signal(
        self, snap_data: Dict[str, Any], lookback_window: List[Dict[str, Any]]
    ) -> float:
        """
        标签因子不产生信号
        """
        return 0.0

    def _extract_valid_prices(self, snap_window: List[Dict[str, Any]]) -> List[float]:
        prices = []
        for snap in snap_window:
            price = snap.get("price_last", 0)
            if (
                price is not None
                and price != 0
                and not (isinstance(price, float) and math.isnan(price))
            ):
                prices.append(float(price))
        return prices

    def _calc_volatility_from_lookback(
        self, lookback_window: Optional[List[Dict[str, Any]]]
    ) -> float:
        """
        用过去窗口计算波动率
        """
        if not lookback_window:
            return 0.0

        if len(lookback_window) < self.past_vol_window:
            lookback_slice = lookback_window
        else:
            lookback_slice = lookback_window[-self.past_vol_window :]

        prices = self._extract_valid_prices(lookback_slice)
        if len(prices) < 2:
            return 0.0

        prices = np.array(prices, dtype=float)

        if self.vol_type == "price_std":
            mean_price = np.mean(prices)
            if mean_price == 0:
                return 0.0
            return float(np.std(prices) / mean_price)

        # 默认：收益率标准差
        if self.return_type == "log":
            rets = np.diff(np.log(prices))
        else:
            rets = np.diff(prices) / prices[:-1]

        if len(rets) == 0:
            return 0.0

        vol = np.std(rets)
        if not np.isfinite(vol):
            return 0.0
        return float(vol)

    def calculate_label(
        self,
        current_snap: Dict[str, Any],
        future_window: List[Dict[str, Any]],
        lookback_window: Optional[List[Dict[str, Any]]] = None,
    ) -> float:
        """
        计算三重屏障法标签

        逻辑：
        1. 用过去 lookback_window 中最近 past_vol_window 个样本计算波动率
        2. 上下屏障 = start_price * (1 ± k * vol)
        3. 在 future_window 中检查先碰到哪个屏障
        4. 返回 -1 / 0 / 1，或者根据 category 映射成成功/失败标签
        """
        if not future_window:
            return 0.0

        start_price = current_snap.get("price_last", 0)
        if (
            start_price is None
            or start_price == 0
            or (isinstance(start_price, float) and math.isnan(start_price))
        ):
            return 0.0

        volatility = self._calc_volatility_from_lookback(lookback_window)
        if volatility <= 0:
            return 0.0

        up_barrier = start_price * (1 + self.k_up * volatility)
        down_barrier = start_price * (1 - self.k_down * volatility)

        touched_up = False
        touched_down = False
        up_touch_idx = None
        down_touch_idx = None

        for i, snap in enumerate(future_window):
            price = snap.get("price_last", 0)
            if (
                price is None
                or price == 0
                or (isinstance(price, float) and math.isnan(price))
            ):
                continue

            if not touched_up and price >= up_barrier:
                touched_up = True
                up_touch_idx = i

            if not touched_down and price <= down_barrier:
                touched_down = True
                down_touch_idx = i

            if touched_up and touched_down:
                break

        if touched_up and touched_down:
            raw_label = 1 if up_touch_idx < down_touch_idx else -1
        elif touched_up:
            raw_label = 1
        elif touched_down:
            raw_label = -1
        else:
            raw_label = 0 if self.use_fixed_horizon_when_no_barrier else 0

        # category 映射
        if self.category == 1:
            return 1.0 if raw_label == 1 else 0.0
        elif self.category == -1:
            return 1.0 if raw_label == -1 else 0.0
        else:
            return float(raw_label)

    def validate_data(self, snap_data: Dict[str, Any]) -> bool:
        if "price_last" not in snap_data:
            return False

        price = snap_data["price_last"]
        if (
            price is None
            or price == 0
            or (isinstance(price, float) and math.isnan(price))
        ):
            return False

        return True


class LabelFactorFactory:
    """标签因子工厂类"""

    @staticmethod
    def create_label_factor(
        label_type: str,
        config: RollingValidationConfig,
        param_dict: Dict[str, Any] = None,
    ) -> BaseAlphaFactor:
        """
        创建标签因子实例

        Args:
            label_type: 标签类型，可选 'return' 或 'triple_barrier'
            config: 滚动验证配置
            param_dict: 参数字典

        Returns:
            BaseAlphaFactor: 标签因子实例
        """
        label_map = {
            "return": ReturnLabelFactor,
            "triple_barrier": TripleBarrierLabelFactor,
        }

        if label_type not in label_map:
            raise ValueError(f"未知标签类型: {label_type}")

        return label_map[label_type](config, param_dict)

    @staticmethod
    def get_available_label_factors() -> List[str]:
        """获取可用标签因子列表"""
        return ["return", "triple_barrier"]
