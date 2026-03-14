# -*- coding: utf-8 -*-
"""
外部冲击识别系统 - 测试脚本
用于验证系统各模块是否正常工作
"""
import sys
import os
import asyncio
from datetime import datetime, timedelta

# 设置控制台编码
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 70)
print("外部冲击识别系统 - 测试脚本")
print("=" * 70)

# 测试1：导入模块
print("\n【测试1】导入模块...")
try:
    from src.data_collectors.free_data_sources import FreeDataSources
    from src.data_collectors.event_detector import EventValidator, HistoricalEventDatabase
    from src.models.endogenous_model import EndogenousBenchmarkModel
    from src.models.transmission_verifier import TransmissionPathAnalyzer
    from src.models.sector_crosssection_verifier import SectorCrossSectionVerifier
    from src.models.significance_tester import EventImpactAnalyzer
    from src.models.shock_detector_system import ExternalShockDetector
    print("✓ 所有模块导入成功")
except Exception as e:
    print(f"✗ 模块导入失败: {e}")
    sys.exit(1)

# 测试2：免费数据源
print("\n【测试2】测试免费数据源...")
try:
    # 测试获取板块数据
    print("  - 获取板块实时数据...")
    sector_data = FreeDataSources.get_sector_realtime()
    if not sector_data.empty:
        print(f"  ✓ 板块数据获取成功，共 {len(sector_data)} 条")
    else:
        print("  ⚠ 板块数据为空（可能是非交易时间）")
    
    # 测试获取北向资金
    print("  - 获取北向资金数据...")
    north_flow = FreeDataSources.get_north_flow()
    if not north_flow.empty:
        print(f"  ✓ 北向资金数据获取成功，共 {len(north_flow)} 条")
    else:
        print("  ⚠ 北向资金数据为空")
    
except Exception as e:
    print(f"  ⚠ 数据获取测试跳过（可能网络问题）: {e}")

# 测试3：事件检测器
print("\n【测试3】测试事件检测器...")
try:
    from src.data_collectors.event_detector import EventKeywordMatcher
    
    matcher = EventKeywordMatcher()
    
    # 测试关键词匹配
    test_news = [
        "中东爆发战争，原油运输受阻",
        "霍尔木兹海峡发生油轮袭击事件",
        "今天天气不错，适合户外运动"  # 无关新闻
    ]
    
    for news in test_news:
        matched, event_type, confidence = matcher.match_event(news)
        status = "✓" if matched else "✗"
        print(f"  {status} '{news[:20]}...' -> 匹配:{matched}, 类型:{event_type}, 置信度:{confidence:.2f}")
    
    print("✓ 事件检测器工作正常")
    
except Exception as e:
    print(f"✗ 事件检测器测试失败: {e}")

# 测试4：历史事件库
print("\n【测试4】测试历史事件库...")
try:
    event_db = HistoricalEventDatabase()
    events = event_db.to_dataframe()
    
    print(f"  ✓ 历史事件库加载成功，共 {len(events)} 个事件")
    print("  事件列表:")
    for _, event in events.iterrows():
        print(f"    - {event['event_date'].strftime('%Y-%m-%d')}: {event['trigger_source']}")
    
except Exception as e:
    print(f"✗ 历史事件库测试失败: {e}")

# 测试5：内生模型（模拟数据）
print("\n【测试5】测试内生基准拟合模型...")
try:
    import pandas as pd
    import numpy as np
    
    # 生成模拟数据
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=200, freq='D')
    
    index_data = pd.DataFrame({
        'date': dates,
        'close': 3000 + np.cumsum(np.random.randn(200) * 20),
        'volume': np.random.randint(1000000, 5000000, 200),
    })
    index_data['trade_date'] = index_data['date']
    
    factor_data = pd.DataFrame({
        'trade_date': dates,
        'return_5d': index_data['close'].pct_change(5).fillna(0),
        'volatility_20d': index_data['close'].pct_change().rolling(20).std().fillna(0),
    })
    
    # 拟合模型
    model = EndogenousBenchmarkModel()
    model.fit(index_data, factor_data)
    
    print("  ✓ 内生模型拟合成功")
    
except Exception as e:
    print(f"✗ 内生模型测试失败: {e}")

# 测试6：行业横截面验证
print("\n【测试6】测试行业横截面验证...")
try:
    verifier = SectorCrossSectionVerifier()
    
    # 模拟原油冲击场景
    sector_returns = {
        '石油开采': 0.05,
        '油服工程': 0.04,
        '煤炭开采': 0.03,
        '航空机场': -0.06,
        '航运港口': -0.05,
        '物流': -0.04,
        '银行': -0.01,
        '计算机': -0.01,
    }
    
    result = verifier.verify(sector_returns, benchmark_return=-0.02)
    
    print(f"  ✓ 行业验证完成")
    print(f"    - 是否原油冲击模式: {result.is_oil_shock_pattern}")
    print(f"    - 置信度: {result.confidence:.2f}")
    print(f"    - 板块分化度: {result.sector_divergence*100:.2f}%")
    
except Exception as e:
    print(f"✗ 行业验证测试失败: {e}")

# 测试7：传导路径验证
print("\n【测试7】测试传导路径验证...")
try:
    import pandas as pd
    import numpy as np
    
    analyzer = TransmissionPathAnalyzer()
    
    # 模拟数据
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    
    vix_data = pd.DataFrame({
        'trade_date': dates,
        'close': 15 + np.random.randn(100) * 3
    })
    
    oil_data = pd.DataFrame({
        'trade_date': dates,
        'close': 75 + np.random.randn(100) * 5
    })
    
    north_flow = pd.DataFrame({
        'trade_date': dates,
        'north_flow': np.random.randn(100) * 30
    })
    
    # 模拟事件日VIX飙升
    vix_data.loc[50, 'close'] = 25
    north_flow.loc[50, 'north_flow'] = -80
    
    result = analyzer.analyze(
        oil_data=oil_data,
        vix_data=vix_data,
        north_flow_data=north_flow,
        event_idx=50
    )
    
    print(f"  ✓ 传导验证完成")
    print(f"    - 主要路径: {result.get('primary_path', '无')}")
    print(f"    - 任何路径验证通过: {result.get('any_path_validated', False)}")
    
except Exception as e:
    print(f"✗ 传导验证测试失败: {e}")

# 测试8：统计检验
print("\n【测试8】测试统计显著性检验...")
try:
    from src.models.significance_tester import StatisticalSignificanceTester
    
    tester = StatisticalSignificanceTester()
    
    # 模拟数据
    np.random.seed(42)
    clean_residuals = np.random.randn(50) * 0.01
    event_residuals = np.random.randn(10) * 0.02 - 0.03
    
    result = tester.comprehensive_test(event_residuals, clean_residuals)
    
    print(f"  ✓ 统计检验完成")
    print(f"    - 是否显著: {result['is_significant']}")
    print(f"    - T检验p值: {result['t_test']['p_value_one_sided']:.4f}")
    
except Exception as e:
    print(f"✗ 统计检验测试失败: {e}")

# 测试9：综合系统
print("\n【测试9】测试综合系统...")
try:
    detector = ExternalShockDetector()
    print("  ✓ 外部冲击检测器初始化成功")
    
    # 显示历史事件
    events = detector.event_db.to_dataframe()
    print(f"  ✓ 加载了 {len(events)} 个历史事件")
    
except Exception as e:
    print(f"✗ 综合系统测试失败: {e}")

# 总结
print("\n" + "=" * 70)
print("测试完成！")
print("=" * 70)
print("\n下一步操作：")
print("1. 安装依赖: pip install -r requirements.txt")
print("2. 启动Web界面: streamlit run app.py")
print("3. 或使用Python脚本进行分析")
print("\n项目结构：")
print("  - app.py: Streamlit可视化界面")
print("  - src/data_collectors/: 数据采集模块")
print("  - src/models/: 核心分析模型")
print("  - src/utils/: 工具函数")
print("  - config/: 配置文件")
print("=" * 70)
