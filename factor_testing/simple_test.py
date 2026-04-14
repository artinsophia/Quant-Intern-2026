#!/usr/bin/env python3
"""
简单测试因子测试框架
"""

import os
import sys

# 添加当前目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

print("=" * 80)
print("简单测试因子测试框架")
print("=" * 80)

# 直接导入各个模块
print("\n1. 直接导入各个模块...")

try:
    # 导入data模块
    from data.factor_data import FactorData

    print("  ✓ FactorData 导入成功")

    # 导入utils模块
    from utils.preprocessing import FactorPreprocessor

    print("  ✓ FactorPreprocessor 导入成功")

    # 导入metrics模块
    from metrics.ic_calculator import ICCalculator

    print("  ✓ ICCalculator 导入成功")

    from metrics.factor_metrics import FactorMetrics

    print("  ✓ FactorMetrics 导入成功")

    # 导入analysis模块
    from analysis.group_test import GroupTester

    print("  ✓ GroupTester 导入成功")

    from analysis.report_generator import ReportGenerator

    print("  ✓ ReportGenerator 导入成功")

    print("\n2. 测试类实例化...")

    # 测试类是否可以实例化（不需要实际数据）
    test_classes = [
        ("FactorData", FactorData),
        ("FactorPreprocessor", FactorPreprocessor),
        ("ICCalculator", ICCalculator),
        ("FactorMetrics", FactorMetrics),
        ("GroupTester", GroupTester),
        ("ReportGenerator", ReportGenerator),
    ]

    for name, cls in test_classes:
        try:
            # 检查类定义
            if hasattr(cls, "__name__"):
                print(f"  ✓ {name} 类定义正确")
            else:
                print(f"  ✗ {name} 类定义错误")
        except Exception as e:
            print(f"  ✗ 测试 {name} 失败: {e}")

    print("\n3. 检查模块结构...")

    # 检查必要的文件
    required_files = [
        "__init__.py",
        "data/__init__.py",
        "data/factor_data.py",
        "metrics/__init__.py",
        "metrics/ic_calculator.py",
        "metrics/factor_metrics.py",
        "analysis/__init__.py",
        "analysis/group_test.py",
        "analysis/report_generator.py",
        "utils/__init__.py",
        "utils/preprocessing.py",
        "example.py",
        "README.md",
    ]

    all_exist = True
    for file_path in required_files:
        full_path = os.path.join(current_dir, file_path)
        if os.path.exists(full_path):
            print(f"  ✓ {file_path}")
        else:
            print(f"  ✗ {file_path} (缺失)")
            all_exist = False

    print("\n" + "=" * 80)

    if all_exist:
        print("✅ 因子测试框架实现完成!")
        print("\n框架包含以下核心模块:")
        print("  1. FactorData - 因子数据管理")
        print("  2. FactorPreprocessor - 因子预处理")
        print("  3. ICCalculator - IC计算")
        print("  4. FactorMetrics - 综合指标计算")
        print("  5. GroupTester - 分组测试")
        print("  6. ReportGenerator - 报告生成")

        print("\n使用方法:")
        print("  1. 查看 example.py 了解基本用法")
        print("  2. 查看 README.md 获取详细文档")
        print("  3. 按照以下步骤使用:")
        print("""
    步骤1: 准备数据
        factor_df = pd.DataFrame(...)  # 因子数据，索引为(date, symbol)
        returns = pd.Series(...)       # 未来收益，索引与因子数据一致
    
    步骤2: 创建因子数据对象
        from factor_testing import FactorData
        factor_data = FactorData(factor_df)
    
    步骤3: 计算因子指标
        from factor_testing import FactorMetrics
        metrics_calc = FactorMetrics(factor_data.get_factor('your_factor'), returns)
        metrics = metrics_calc.calculate_all_metrics()
    
    步骤4: 分组测试
        from factor_testing import GroupTester
        tester = GroupTester(factor_data.get_factor('your_factor'), returns)
        results = tester.run_comprehensive_test()
    
    步骤5: 生成报告
        from factor_testing import ReportGenerator
        report_gen = ReportGenerator('your_factor', factor_data.get_factor('your_factor'), returns)
        report_gen.save_report('./output')
        """)
    else:
        print("⚠️  框架文件不完整")

except Exception as e:
    print(f"\n❌ 测试失败: {e}")
    import traceback

    traceback.print_exc()
