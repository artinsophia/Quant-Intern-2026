"""
数据加载工具模块
用于加载和处理市场数据
"""

import sys
import os
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np


class DataLoader:
    """数据加载器"""

    def __init__(self, base_tool_path: str = "/home/jovyan/base_demo"):
        """
        初始化数据加载器

        Args:
            base_tool_path: base_tool模块路径
        """
        self.base_tool_path = base_tool_path
        self._add_base_tool_to_path()

    def _add_base_tool_to_path(self):
        """添加base_tool到系统路径"""
        if self.base_tool_path not in sys.path:
            sys.path.append(self.base_tool_path)

    def load_snapshot_data(
        self, instrument_id: str, trade_ymd: str
    ) -> List[Dict[str, Any]]:
        """
        加载快照数据

        Args:
            instrument_id: 标的ID
            trade_ymd: 交易日期 (YYYYMMDD)

        Returns:
            快照数据列表
        """
        try:
            from base_tool import snap_list_load

            snap_list = snap_list_load(instrument_id, trade_ymd)
            print(f"加载数据: {instrument_id} {trade_ymd}, 共{len(snap_list)}条记录")
            return snap_list

        except ImportError as e:
            print(f"无法导入base_tool: {e}")
            print("请确保base_tool模块在正确路径")
            return []
        except Exception as e:
            print(f"加载数据失败: {e}")
            return []

    def create_mock_data(self, n_samples: int = 1000) -> List[Dict[str, Any]]:
        """
        创建模拟数据用于测试

        Args:
            n_samples: 样本数量

        Returns:
            模拟快照数据
        """
        snap_slice = []
        price = 100.0

        for i in range(n_samples):
            # 生成随机价格变动
            price_change = np.random.normal(0, 0.01)
            price = max(0.1, price + price_change)

            # 生成随机成交量
            bid_volume = np.random.randint(100, 1000)
            ask_volume = np.random.randint(100, 1000)

            snap = {
                "time_mark": i * 1000,
                "time_hms": f"{i // 3600:02d}:{(i % 3600) // 60:02d}:{i % 60:02d}",
                "price_open": price - np.random.random() * 0.01,
                "price_low": price - np.random.random() * 0.02,
                "price_high": price + np.random.random() * 0.02,
                "price_last": price,
                "price_vwap": price + np.random.normal(0, 0.005),
                "num_trades": i * 10,
                "bid_book": [
                    [price - 0.01, bid_volume],
                    [price - 0.02, bid_volume * 2],
                ],
                "ask_book": [
                    [price + 0.01, ask_volume],
                    [price + 0.02, ask_volume * 2],
                ],
                "buy_trade": [
                    [price - 0.005, np.random.randint(10, 100)],
                    [price - 0.01, np.random.randint(10, 100)],
                ],
                "sell_trade": [
                    [price + 0.005, np.random.randint(10, 100)],
                    [price + 0.01, np.random.randint(10, 100)],
                ],
            }
            snap_slice.append(snap)

        return snap_slice

    def snapshots_to_dataframe(self, snap_slice: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        将快照数据转换为DataFrame

        Args:
            snap_slice: 快照数据列表

        Returns:
            DataFrame
        """
        records = []

        for snap in snap_slice:
            record = {
                "timestamp": snap.get("time_mark"),
                "time_hms": snap.get("time_hms"),
                "price_open": snap.get("price_open"),
                "price_high": snap.get("price_high"),
                "price_low": snap.get("price_low"),
                "price_last": snap.get("price_last"),
                "price_vwap": snap.get("price_vwap"),
                "num_trades": snap.get("num_trades"),
            }

            # 提取买卖盘信息
            if snap.get("bid_book") and len(snap["bid_book"]) > 0:
                record["best_bid"] = snap["bid_book"][0][0]
                record["best_bid_volume"] = snap["bid_book"][0][1]

            if snap.get("ask_book") and len(snap["ask_book"]) > 0:
                record["best_ask"] = snap["ask_book"][0][0]
                record["best_ask_volume"] = snap["ask_book"][0][1]

            # 计算买卖成交量
            buy_volume = sum(vol for _, vol in snap.get("buy_trade", []))
            sell_volume = sum(vol for _, vol in snap.get("sell_trade", []))
            record["buy_volume"] = buy_volume
            record["sell_volume"] = sell_volume
            record["net_volume"] = buy_volume - sell_volume

            records.append(record)

        df = pd.DataFrame(records)

        # 设置时间索引
        if "timestamp" in df.columns:
            df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", errors="coerce")
            df.set_index("datetime", inplace=True)

        return df

    def extract_features_from_snapshots(
        self,
        snap_slice: List[Dict[str, Any]],
        window_size: int = 60,
        step_size: int = 1,
    ) -> pd.DataFrame:
        """
        从快照数据中提取特征

        Args:
            snap_slice: 快照数据列表
            window_size: 特征计算窗口大小
            step_size: 滑动步长

        Returns:
            特征DataFrame
        """
        try:
            # 导入特征提取模块
            sys.path.append("/home/jovyan/work/tactics_demo/delta")
            from features import create_feature

            features_list = []

            # 滑动窗口提取特征
            for i in range(window_size, len(snap_slice), step_size):
                window_slice = snap_slice[i - window_size : i]

                try:
                    features = create_feature(
                        window_slice, short_window=min(30, window_size)
                    )

                    # 添加时间戳
                    if window_slice:
                        features["timestamp"] = window_slice[-1].get("time_mark")
                        features["price"] = window_slice[-1].get("price_last")

                    features_list.append(features)

                except Exception as e:
                    print(f"提取特征失败 (窗口 {i}): {e}")
                    continue

            if not features_list:
                return pd.DataFrame()

            df = pd.DataFrame(features_list)

            # 设置时间索引
            if "timestamp" in df.columns:
                df["datetime"] = pd.to_datetime(
                    df["timestamp"], unit="ms", errors="coerce"
                )
                df.set_index("datetime", inplace=True)

            return df

        except ImportError as e:
            print(f"无法导入特征提取模块: {e}")
            return pd.DataFrame()
        except Exception as e:
            print(f"提取特征失败: {e}")
            return pd.DataFrame()

    def load_multiple_days_data(
        self, instrument_id: str, start_ymd: str, end_ymd: str
    ) -> List[Dict[str, Any]]:
        """
        加载多日数据

        Args:
            instrument_id: 标的ID
            start_ymd: 开始日期
            end_ymd: 结束日期

        Returns:
            合并的快照数据
        """
        all_snapshots = []

        # 生成日期范围
        start_date = pd.to_datetime(start_ymd, format="%Y%m%d")
        end_date = pd.to_datetime(end_ymd, format="%Y%m%d")
        date_range = pd.date_range(start=start_date, end=end_date, freq="D")

        for date in date_range:
            trade_ymd = date.strftime("%Y%m%d")
            print(f"加载日期: {trade_ymd}")

            daily_snapshots = self.load_snapshot_data(instrument_id, trade_ymd)
            all_snapshots.extend(daily_snapshots)

            if daily_snapshots:
                print(f"  加载了 {len(daily_snapshots)} 条记录")
            else:
                print(f"  无数据或加载失败")

        print(f"总共加载了 {len(all_snapshots)} 条记录")
        return all_snapshots
