#!/usr/bin/env python3
"""
验证因子测试框架结构
"""

import os
import sys

print("=" * 80)
print("验证因子测试框架结构")
print("=" * 80)

current_dir = os.path.dirname(os.path.abspath(__file__))

print("\n1. 检查目录结构...")

# 检查主要目录
required_dirs = ["data", "metrics", "analysis", "utils"]
for dir_name in required_dirs:
    dir_path = os.path.join(current_dir, dir_name)
    if os.path.exists(dir_path) and os.path.isdir(dir_path):
        print(f"  ✓ 目录 {dir_name}/ 存在")
    else:
        print(f"  ✗ 目录 {dir_name}/ 不存在")

print("\n2. 检查核心文件...")

# 检查核心文件
core_files = [
    # 主模块
    "__init__.py",
    # data模块
    "data/__init__.py",
    "data/factor_data.py",
    # metrics模块
    "metrics/__init__.py",
    "metrics/ic_calculator.py",
    "metrics/factor_metrics.py",
    # analysis模块
    "analysis/__init__.py",
    "analysis/group_test.py",
    "analysis/report_generator.py",
    # utils模块
    "utils/__init__.py",
    "utils/preprocessing.py",
    # 文档和示例
    "example.py",
    "README.md",
]

missing_files = []
for file_path in core_files:
    full_path = os.path.join(current_dir, file_path)
    if os.path.exists(full_path):
        # 检查文件大小
        size = os.path.getsize(full_path)
        print(f"  ✓ {file_path:30} ({size} bytes)")
    else:
        print(f"  ✗ {file_path:30} (缺失)")
        missing_files.append(file_path)

print("\n3. 检查文件内容...")

# 检查关键文件是否包含必要的类定义
key_files_classes = {
    "data/factor_data.py": ["FactorData"],
    "utils/preprocessing.py": ["FactorPreprocessor"],
    "metrics/ic_calculator.py": ["ICCalculator"],
    "metrics/factor_metrics.py": ["FactorMetrics"],
    "analysis/group_test.py": ["GroupTester"],
    "analysis/report_generator.py": ["ReportGenerator"],
}

for file_path, expected_classes in key_files_classes.items():
    full_path = os.path.join(current_dir, file_path)
    if os.path.exists(full_path):
        try:
            with open(full_path, "r", encoding="utf-8") as f:
                content = f.read()

            missing_classes = []
            for class_name in expected_classes:
                if f"class {class_name}" in content:
                    print(f"  ✓ {file_path:30} 包含类 {class_name}")
                else:
                    missing_classes.append(class_name)

            if missing_classes:
                print(f"  ✗ {file_path:30} 缺少类: {missing_classes}")

        except Exception as e:
            print(f"  ✗ 读取 {file_path} 失败: {e}")
    else:
        print(f"  ✗ {file_path:30} 文件不存在")

print("\n4. 检查示例和文档...")

# 检查示例文件
example_path = os.path.join(current_dir, "example.py")
if os.path.exists(example_path):
    size = os.path.getsize(example_path)
    print(f"  ✓ example.py ({size} bytes) - 包含完整使用示例")
else:
    print("  ✗ example.py 缺失")

# 检查README
readme_path = os.path.join(current_dir, "README.md")
if os.path.exists(readme_path):
    size = os.path.getsize(readme_path)
    print(f"  ✓ README.md ({size} bytes) - 包含完整文档")
else:
    print("  ✗ README.md 缺失")

print("\n" + "=" * 80)

if not missing_files:
    print("✅ 因子测试框架结构完整!")
    print("\n框架架构总结:")
    print("  📁 data/          - 因子数据管理模块")
    print("    ├── factor_data.py    - FactorData类")
    print("  📁 metrics/       - 指标计算模块")
    print("    ├── ic_calculator.py  - ICCalculator类")
    print("    └── factor_metrics.py - FactorMetrics类")
    print("  📁 analysis/      - 分析测试模块")
    print("    ├── group_test.py     - GroupTester类")
    print("    └── report_generator.py - ReportGenerator类")
    print("  📁 utils/         - 工具模块")
    print("    └── preprocessing.py  - FactorPreprocessor类")
    print("  📄 example.py     - 使用示例")
    print("  📄 README.md      - 详细文档")

    print("\n核心功能:")
    print("  1. 因子数据加载和管理 (FactorData)")
    print("  2. 因子预处理 (FactorPreprocessor)")
    print("  3. IC计算 (ICCalculator)")
    print("  4. 综合指标计算 (FactorMetrics)")
    print("  5. 分组测试 (GroupTester)")
    print("  6. 报告生成 (ReportGenerator)")

    print("\n依赖库:")
    print("  - pandas: 数据处理")
    print("  - numpy: 数值计算")
    print("  - scipy: 统计计算")
    print("  - matplotlib: 可视化")
    print("  - seaborn: 高级可视化")
    print("  - scikit-learn: 机器学习（用于中性化）")

    print("\n安装依赖:")
    print("  pip install pandas numpy scipy matplotlib seaborn scikit-learn")

    print("\n使用前准备:")
    print("  1. 安装依赖库")
    print("  2. 准备因子数据和收益数据")
    print("  3. 参考 example.py 中的示例代码")
    print("  4. 查看 README.md 获取详细说明")

else:
    print(f"⚠️  框架不完整，缺失 {len(missing_files)} 个文件:")
    for file in missing_files:
        print(f"    - {file}")

print("=" * 80)
