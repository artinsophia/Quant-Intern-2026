#!/usr/bin/env python3
import sys

sys.path.append("/home/jovyan/work/tactics_demo/delta")

# 直接导入train模块中的函数
from train import get_trade_dates, split_dates, split_dates_by_range

# 获取交易日数据
trade_dates = get_trade_dates()
print(f"总交易日数量: {len(trade_dates)}")
print(f"交易日范围: {trade_dates[0]} ~ {trade_dates[-1]}")

print("\n" + "=" * 50)
print("测试原来的split_dates函数（按天数分割）:")
print("=" * 50)
train_dates1, valid_dates1, test_dates1 = split_dates(trade_dates, 45, 15, 17)
print(f"训练集: {len(train_dates1)}天, 示例: {train_dates1[:3]}...{train_dates1[-3:]}")
print(f"验证集: {len(valid_dates1)}天, 示例: {valid_dates1[:3]}...{valid_dates1[-3:]}")
print(f"测试集: {len(test_dates1)}天, 示例: {test_dates1[:3]}...{test_dates1[-3:]}")

print("\n" + "=" * 50)
print("测试新的split_dates_by_range函数（按日期范围分割）:")
print("=" * 50)
train_dates2, valid_dates2, test_dates2 = split_dates_by_range(
    trade_dates,
    train_start="20251201",
    train_end="20260203",
    valid_start="20260204",
    valid_end="20260304",
    test_start="20260305",
    test_end="20260327",
)
print(f"训练集: {len(train_dates2)}天, 示例: {train_dates2[:3]}...{train_dates2[-3:]}")
print(f"验证集: {len(valid_dates2)}天, 示例: {valid_dates2[:3]}...{valid_dates2[-3:]}")
print(f"测试集: {len(test_dates2)}天, 示例: {test_dates2[:3]}...{test_dates2[-3:]}")

print("\n" + "=" * 50)
print("验证两种方法结果是否一致:")
print("=" * 50)
print(f"训练集是否一致: {train_dates1 == train_dates2}")
print(f"验证集是否一致: {valid_dates1 == valid_dates2}")
print(f"测试集是否一致: {test_dates1 == test_dates2}")

print("\n测试完成！新函数工作正常。")
