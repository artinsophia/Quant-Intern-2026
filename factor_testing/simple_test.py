#!/usr/bin/env python3
"""
简单测试因子测试框架
"""

import sys
import os

# 添加项目路径
sys.path.insert(0, "/home/jovyan/work/tactics_demo/factor_testing")

print("测试因子测试框架...")
print("=" * 60)

# 测试1: 检查目录结构
print("1. 检查目录结构...")
required_dirs = ["utils", "metrics", "visualization"]
required_files = [
    "utils/data_loader.py",
    "metrics/ic_analysis.py",
    "metrics/group_backtest.py",
    "visualization/plot_factors.py",
    "test_factor_performance.py",
    "example_usage.py",
    "README.md",
]

all_good = True

for dir_name in required_dirs:
    dir_path = os.path.join("/home/jovyan/work/tactics_demo/factor_testing", dir_name)
    if os.path.exists(dir_path):
        print(f"  ✓ 目录存在: {dir_name}")
    else:
        print(f"  ✗ 目录不存在: {dir_name}")
        all_good = False

for file_name in required_files:
    file_path = os.path.join("/home/jovyan/work/tactics_demo/factor_testing", file_name)
    if os.path.exists(file_path):
        print(f"  ✓ 文件存在: {file_name}")
    else:
        print(f"  ✗ 文件不存在: {file_name}")
        all_good = False

# 测试2: 尝试导入主要模块
print("\n2. 尝试导入模块...")

try:
    from utils.data_loader import DataLoader

    print("  ✓ 导入 DataLoader 成功")

    # 测试创建实例
    loader = DataLoader()
    print("  ✓ 创建 DataLoader 实例成功")
except Exception as e:
    print(f"  ✗ 导入 DataLoader 失败: {e}")
    all_good = False

try:
    from metrics.ic_analysis import ICAnalyzer

    print("  ✓ 导入 ICAnalyzer 成功")

    # 测试创建实例
    ic_analyzer = ICAnalyzer()
    print("  ✓ 创建 ICAnalyzer 实例成功")
except Exception as e:
    print(f"  ✗ 导入 ICAnalyzer 失败: {e}")
    all_good = False

try:
    from metrics.group_backtest import GroupBacktester

    print("  ✓ 导入 GroupBacktester 成功")

    # 测试创建实例
    group_tester = GroupBacktester()
    print("  ✓ 创建 GroupBacktester 实例成功")
except Exception as e:
    print(f"  ✗ 导入 GroupBacktester 失败: {e}")
    all_good = False

# 测试3: 检查优化后的特征提取器
print("\n3. 检查优化后的特征提取器...")

try:
    # 添加delta模块路径
    sys.path.append("/home/jovyan/work/tactics_demo/delta")
    from features import FeatureExtractor, create_feature, calculate_hurst_exponent

    print("  ✓ 导入优化后的特征提取器成功")
    print("  ✓ 包含 calculate_hurst_exponent 函数")

    # 测试Hurst指数计算
    import numpy as np

    test_prices = 100 + np.cumsum(np.random.randn(100)) * 0.1
    hurst = calculate_hurst_exponent(test_prices.tolist())
    print(f"  ✓ 计算Hurst指数成功: {hurst:.4f}")

except Exception as e:
    print(f"  ✗ 导入特征提取器失败: {e}")
    all_good = False

# 测试4: 检查主脚本
print("\n4. 检查主脚本...")

main_script = "/home/jovyan/work/tactics_demo/factor_testing/test_factor_performance.py"
if os.path.exists(main_script):
    print(f"  ✓ 主脚本存在: {main_script}")

    # 检查脚本内容
    with open(main_script, "r", encoding="utf-8") as f:
        content = f.read()

    required_classes = [
        "FactorPerformanceTester",
        "DataLoader",
        "ICAnalyzer",
        "GroupBacktester",
        "FactorVisualizer",
    ]

    for class_name in required_classes:
        if class_name in content:
            print(f"  ✓ 包含类: {class_name}")
        else:
            print(f"  ✗ 缺少类: {class_name}")
            all_good = False
else:
    print(f"  ✗ 主脚本不存在")
    all_good = False

# 总结
print("\n" + "=" * 60)
if all_good:
    print("🎉 所有测试通过!")
    print("\n下一步:")
    print("1. 查看文档: cat README.md")
    print("2. 运行示例: python example_usage.py")
    print("3. 快速测试: python test_factor_performance.py --quick")
else:
    print("⚠ 部分测试失败，请检查上述错误")
print("=" * 60)
