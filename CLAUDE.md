# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

A股外部冲击识别系统 - 量化识别外围突发事件（如地缘冲突、原油供应链冲击等）对A股市场的影响，区分外部冲击下跌与市场内生悲观预期下跌。

## 核心命令

### 开发环境

```bash
# 安装依赖
pip install -r requirements.txt

# 运行 Web 界面
streamlit run app.py

# 运行测试
pytest

# 运行单个测试文件
pytest test_system.py -v
```

### 数据获取测试

```bash
# 测试数据源连接
python -c "from src.data_collectors.free_data_sources import FreeDataSources; print(FreeDataSources.get_index_history('sh000300', '20240101', '20240131'))"
```

## 系统架构

### 核心流程（8步）

系统通过 `ExternalShockDetector` 主控制器执行完整识别流程：

1. **数据收集** - 获取A股指数、板块、北向资金、原油、VIX等数据
2. **事件标准化验证** - 通过 `EventValidator` 验证事件有效性
3. **内生基准拟合** - 使用 `EndogenousBenchmarkModel` 构建60日滚动窗口多元线性回归
4. **残差分析** - 通过 `ResidualAnalyzer` 检测异常（Z-score阈值 ≤-2.0）
5. **传导路径验证** - 使用 `TransmissionPathAnalyzer` 验证成本输入型/避险情绪传导
6. **行业横截面验证** - 通过 `SectorCrossSectionVerifier` 识别板块分化模式
7. **统计显著性检验** - 使用 `EventImpactAnalyzer` 执行T检验、F检验、贡献度拆分
8. **综合判定** - 输出 `ShockDetectionResult` 包含置信度、冲击类型、详细验证结果

### 模块职责

- `src/data_collectors/` - 数据采集层
  - `free_data_sources.py` - 统一数据接口（AKShare、yfinance、EFinance）
  - `event_detector.py` - 事件检测与历史事件库

- `src/models/` - 核心算法层
  - `shock_detector_system.py` - 主控制器，协调所有模块
  - `endogenous_model.py` - 内生基准拟合与残差分析
  - `transmission_verifier.py` - 传导路径验证（油价→通胀→A股 / VIX→外资→A股）
  - `sector_crosssection_verifier.py` - 行业横截面验证与内生下跌检测
  - `significance_tester.py` - 统计检验与贡献度分析

- `app.py` - Streamlit可视化界面

### 关键阈值

| 维度 | 阈值 | 用途 |
|------|------|------|
| 残差Z-score | ≤-2.0 | 异常检测 |
| 原油单日涨幅 | ≥3% | 价格冲击验证 |
| VIX单日涨幅 | ≥20% | 避险情绪验证 |
| 北向资金流出 | ≥50亿 | 外资流出验证 |
| 板块分化度 | ≥3% | 行业分化阈值 |
| 外生贡献度 | ≥60% | 外部冲击主导判定 |

## 数据源说明

所有数据源均为免费：
- **AKShare** - A股指数、板块、资金流、融资融券、PMI、利率等
- **yfinance** - VIX指数(`^VIX`)、原油期货(`CL=F`/`BZ=F`)、标普500(`^GSPC`)
- **EFinance** - 东方财富数据接口

数据可能有15-30分钟延迟，需考虑A股交易时间限制。

## 编码规范

### 异步编程

系统使用 `async/await` 进行数据采集：

```python
# 正确：使用 asyncio.run()
result = asyncio.run(detector.detect(event_date, news_text))

# 错误：直接调用 async 函数
result = detector.detect(event_date, news_text)  # 返回 coroutine 对象
```

### 数据处理

- 所有日期格式统一为 `datetime` 对象
- 数据源日期参数格式为 `YYYYMMDD` 字符串
- 涨跌幅统一为小数形式（0.05 表示5%）
- 使用 `loguru` 进行日志记录

### 错误处理

数据采集失败不应中断整个流程，使用 try-except 包裹每个数据源调用，返回空 DataFrame 作为降级处理。

## 常见开发任务

### 添加新的数据源

在 `src/data_collectors/free_data_sources.py` 的 `FreeDataSources` 类中添加静态方法。

### 添加新的验证维度

1. 在对应的验证器类中添加方法
2. 在 `ExternalShockDetector._make_final_decision()` 中更新综合判定逻辑
3. 在 `ShockDetectionResult` 中添加对应字段

### 调整阈值

阈值定义在各验证器类的常量或配置中，搜索具体数值（如 `-2.0`、`0.03`）进行修改。

## 注意事项

- 免费数据源可能不稳定，需要容错处理
- 原油以美元计价，需考虑汇率因素
- 内生熊市与外生冲击共振时，需综合判断（贡献度30%-60%区间）
- 历史事件回测需确保数据完整性
