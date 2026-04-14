"""
因子测试框架 - Factor Testing Framework
提供完整的因子IC、IR、分组测试等指标计算功能
"""

# 延迟导入，避免循环依赖
__version__ = "1.0.0"
__all__ = [
    "FactorData",
    "ICCalculator",
    "FactorMetrics",
    "GroupTester",
    "ReportGenerator",
    "FactorPreprocessor",
]


# 提供导入函数
def import_factor_data():
    """导入FactorData类"""
    from .data.factor_data import FactorData

    return FactorData


def import_factor_preprocessor():
    """导入FactorPreprocessor类"""
    from .utils.preprocessing import FactorPreprocessor

    return FactorPreprocessor


def import_ic_calculator():
    """导入ICCalculator类"""
    from .metrics.ic_calculator import ICCalculator

    return ICCalculator


def import_factor_metrics():
    """导入FactorMetrics类"""
    from .metrics.factor_metrics import FactorMetrics

    return FactorMetrics


def import_group_tester():
    """导入GroupTester类"""
    from .analysis.group_test import GroupTester

    return GroupTester


def import_report_generator():
    """导入ReportGenerator类"""
    from .analysis.report_generator import ReportGenerator

    return ReportGenerator


# 提供直接访问
FactorData = import_factor_data()
FactorPreprocessor = import_factor_preprocessor()
ICCalculator = import_ic_calculator()
FactorMetrics = import_factor_metrics()
GroupTester = import_group_tester()
ReportGenerator = import_report_generator()
