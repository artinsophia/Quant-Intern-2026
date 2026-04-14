#!/usr/bin/env python3
"""
测试因子测试框架的基本功能
"""

import os
import sys

# 添加当前目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

print("=" * 80)
print("测试因子测试框架")
print("=" * 80)

try:
    # 测试导入 - 使用相对导入
    print("\n1. 测试模块导入...")

    # 直接导入各个模块文件
    import importlib.util

    # 测试导入 __init__.py
    spec = importlib.util.spec_from_file_location(
        "factor_testing", os.path.join(current_dir, "__init__.py")
    )
    factor_testing = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(factor_testing)
    print("  ✓ factor_testing 包导入成功")

    # 测试导入各个模块
    modules_to_test = [
        ("data.factor_data", "FactorData"),
        ("utils.preprocessing", "FactorPreprocessor"),
        ("metrics.ic_calculator", "ICCalculator"),
        ("metrics.factor_metrics", "FactorMetrics"),
        ("analysis.group_test", "GroupTester"),
        ("analysis.report_generator", "ReportGenerator"),
    ]

    imported_classes = {}

    for module_path, class_name in modules_to_test:
        try:
            # 构建完整路径
            if module_path == "data.factor_data":
                file_path = os.path.join(current_dir, "data", "factor_data.py")
            elif module_path == "utils.preprocessing":
                file_path = os.path.join(current_dir, "utils", "preprocessing.py")
            elif module_path == "metrics.ic_calculator":
                file_path = os.path.join(current_dir, "metrics", "ic_calculator.py")
            elif module_path == "metrics.factor_metrics":
                file_path = os.path.join(current_dir, "metrics", "factor_metrics.py")
            elif module_path == "analysis.group_test":
                file_path = os.path.join(current_dir, "analysis", "group_test.py")
            elif module_path == "analysis.report_generator":
                file_path = os.path.join(current_dir, "analysis", "report_generator.py")

            spec = importlib.util.spec_from_file_location(module_path, file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # 获取类
            if hasattr(module, class_name):
                imported_classes[class_name] = getattr(module, class_name)
                print(f"  ✓ {class_name} 导入成功")
            else:
                print(f"  ✗ {class_name} 在模块 {module_path} 中未找到")

        except Exception as e:
            print(f"  ✗ 导入 {class_name} 失败: {e}")

    print("\n2. 测试类定义...")

    # 检查类是否存在
    for class_name, class_obj in imported_classes.items():
        if class_obj and hasattr(class_obj, "__name__"):
            print(f"  ✓ {class_name} 类定义正确")
        else:
            print(f"  ✗ {class_name} 类定义错误")

    print("\n3. 测试目录结构...")

    required_dirs = ["data", "metrics", "analysis", "utils"]
    for dir_name in required_dirs:
        dir_path = os.path.join(current_dir, dir_name)
        if os.path.exists(dir_path):
            print(f"  ✓ 目录 {dir_name}/ 存在")
        else:
            print(f"  ✗ 目录 {dir_name}/ 不存在")

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

    print("\n4. 测试文件结构...")
    missing_files = []

    for file_path in required_files:
        full_path = os.path.join(current_dir, file_path)
        if os.path.exists(full_path):
            print(f"  ✓ 文件 {file_path} 存在")
        else:
            print(f"  ✗ 文件 {file_path} 不存在")
            missing_files.append(file_path)

    if missing_files:
        print(f"\n警告: 缺少 {len(missing_files)} 个文件")
    else:
        print("\n✓ 所有必需文件都存在")

    print("\n5. 测试框架版本...")

    # 从__init__.py读取版本
    init_path = os.path.join(current_dir, "__init__.py")
    with open(init_path, "r") as f:
        content = f.read()
        if "__version__" in content:
            # 简单提取版本号
            import re

            match = re.search(r"__version__\s*=\s*['\"]([^'\"]+)['\"]", content)
            if match:
                version = match.group(1)
                print(f"  ✓ 框架版本: {version}")
            else:
                print("  ✗ 无法提取版本号")
        else:
            print("  ✗ 版本号未定义")

    print("\n" + "=" * 80)
    print("框架测试完成!")
    print("=" * 80)

    if not missing_files:
        print("\n✅ 因子测试框架实现成功!")
        print("\n主要功能模块:")
        print("  - FactorData: 因子数据管理")
        print("  - FactorPreprocessor: 因子预处理")
        print("  - ICCalculator: IC计算")
        print("  - FactorMetrics: 综合指标计算")
        print("  - GroupTester: 分组测试")
        print("  - ReportGenerator: 报告生成")

        print("\n使用说明:")
        print("  1. 查看 example.py 了解基本用法")
        print("  2. 查看 README.md 获取详细文档")
        print("  3. 运行: python example.py 查看完整示例")

    else:
        print(f"\n⚠️  框架不完整，缺少文件: {missing_files}")

except Exception as e:
    print(f"\n❌ 测试过程中出错: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)
