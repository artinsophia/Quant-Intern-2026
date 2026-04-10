#!/usr/bin/env python3
"""
验证项目结构
"""

import os
import sys
from pathlib import Path


def verify_project_structure():
    """验证项目结构"""
    print("验证市场分析工具项目结构...")
    print("=" * 60)

    base_dir = Path(__file__).parent
    required_files = [
        "daily_analysis.py",
        "market_analysis_demo.ipynb",
        "cli_tool.py",
        "advanced_visualization.py",
        "README.md",
    ]

    print("检查必需文件:")
    all_exist = True
    for file in required_files:
        file_path = base_dir / file
        if file_path.exists():
            print(f"  ✓ {file}")
        else:
            print(f"  ✗ {file} (缺失)")
            all_exist = False

    print(f"\n检查回测结果目录:")
    backtest_dir = Path("/home/jovyan/work/backtest_result")
    if backtest_dir.exists():
        pkl_files = list(backtest_dir.glob("*.pkl"))
        print(f"  ✓ 目录存在，包含 {len(pkl_files)} 个.pkl文件")

        if pkl_files:
            # 显示一些示例文件
            print(f"  示例文件:")
            for file in pkl_files[:3]:
                print(f"    - {file.name}")

            # 分析文件命名模式
            print(f"\n  文件命名模式分析:")
            patterns = {}
            for file in pkl_files[:10]:  # 只分析前10个文件
                parts = file.stem.split("_")
                if len(parts) >= 3:
                    instrument = parts[0]
                    date = parts[1]
                    strategy = "_".join(parts[2:])

                    if instrument not in patterns:
                        patterns[instrument] = set()
                    patterns[instrument].add(strategy[:20])  # 只取策略名前20个字符

            for instrument, strategies in patterns.items():
                print(f"    标的 {instrument}: {len(strategies)} 种策略")
                for strategy in list(strategies)[:3]:  # 显示前3种策略
                    print(f"      - {strategy}...")
    else:
        print(f"  ✗ 目录不存在: {backtest_dir}")
        all_exist = False

    print(f"\n检查Python模块结构:")
    try:
        # 尝试导入模块（不执行，只检查语法）
        with open(base_dir / "daily_analysis.py", "r") as f:
            content = f.read()
            if "class DailyMarketAnalyzer" in content:
                print("  ✓ daily_analysis.py 包含 DailyMarketAnalyzer 类")
            else:
                print("  ✗ daily_analysis.py 不包含 DailyMarketAnalyzer 类")

        with open(base_dir / "cli_tool.py", "r") as f:
            content = f.read()
            if "argparse" in content and "main()" in content:
                print("  ✓ cli_tool.py 包含命令行接口")
            else:
                print("  ✗ cli_tool.py 不包含完整的命令行接口")

        with open(base_dir / "advanced_visualization.py", "r") as f:
            content = f.read()
            if "class MarketVisualizer" in content:
                print("  ✓ advanced_visualization.py 包含 MarketVisualizer 类")
            else:
                print("  ✗ advanced_visualization.py 不包含 MarketVisualizer 类")

    except Exception as e:
        print(f"  读取文件时出错: {e}")

    print(f"\n检查Jupyter notebook:")
    notebook_path = base_dir / "market_analysis_demo.ipynb"
    if notebook_path.exists():
        file_size = notebook_path.stat().st_size
        print(f"  ✓ notebook存在，大小: {file_size} 字节")

        # 简单检查内容
        with open(notebook_path, "r") as f:
            content = f.read(1000)  # 只读取前1000个字符
            if "cells" in content and "markdown" in content:
                print("  ✓ notebook格式正确")
            else:
                print("  ⚠ notebook格式可能不正确")
    else:
        print("  ✗ notebook不存在")

    print(f"\n检查README文档:")
    readme_path = base_dir / "README.md"
    if readme_path.exists():
        with open(readme_path, "r") as f:
            lines = f.readlines()
            print(f"  ✓ README存在，{len(lines)} 行")

            # 检查关键内容
            content = "".join(lines)
            if "功能特性" in content and "使用方法" in content:
                print("  ✓ README包含必要章节")
            else:
                print("  ⚠ README可能不完整")
    else:
        print("  ✗ README不存在")

    print("\n" + "=" * 60)

    if all_exist:
        print("项目结构验证通过！")
        print("\n下一步:")
        print("1. 安装依赖: pip install pandas numpy matplotlib")
        print("2. 运行示例: python daily_analysis.py")
        print("3. 查看完整教程: jupyter notebook market_analysis_demo.ipynb")
    else:
        print("项目结构验证失败，请检查缺失的文件")

    return all_exist


def check_data_files():
    """检查数据文件"""
    print("\n检查数据文件示例:")
    print("-" * 40)

    backtest_dir = Path("/home/jovyan/work/backtest_result")
    if backtest_dir.exists():
        # 按标的分组统计
        instruments = {}
        for file in backtest_dir.glob("*.pkl"):
            parts = file.stem.split("_")
            if len(parts) >= 2:
                instrument = parts[0]
                if instrument not in instruments:
                    instruments[instrument] = []
                instruments[instrument].append(file)

        print(f"找到 {len(instruments)} 个标的:")
        for instrument, files in instruments.items():
            print(f"  {instrument}: {len(files)} 个文件")

            # 统计日期范围
            dates = []
            for file in files[:10]:  # 只检查前10个文件
                parts = file.stem.split("_")
                if len(parts) >= 2:
                    dates.append(parts[1])

            if dates:
                dates.sort()
                print(f"    日期范围: {dates[0]} 到 {dates[-1]}")

    return True


def main():
    """主函数"""
    print("市场分析工具项目验证")
    print("=" * 60)

    verify_project_structure()
    check_data_files()

    print("\n" + "=" * 60)
    print("验证完成！")


if __name__ == "__main__":
    main()
