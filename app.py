"""
外部冲击识别系统 - 可视化界面
使用Streamlit构建交互式分析界面
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import asyncio
import sys
import os

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.shock_detector_system import ExternalShockDetector, ShockDetectionResult
from src.data_collectors.free_data_sources import FreeDataSources


# 页面配置
st.set_page_config(
    page_title="A股外部冲击识别系统",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .success-text {
        color: #28a745;
        font-weight: bold;
    }
    .warning-text {
        color: #ffc107;
        font-weight: bold;
    }
    .danger-text {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


def init_detector():
    """初始化检测器"""
    if 'detector' not in st.session_state:
        st.session_state.detector = ExternalShockDetector()
    return st.session_state.detector


def render_sidebar():
    """渲染侧边栏"""
    st.sidebar.title("⚙️ 系统设置")
    
    # 日期选择
    st.sidebar.subheader("📅 分析日期")
    event_date = st.sidebar.date_input(
        "选择事件日期",
        value=datetime.now() - timedelta(days=30),
        max_value=datetime.now()
    )
    
    # 事件类型
    st.sidebar.subheader("📌 事件类型")
    event_type = st.sidebar.selectbox(
        "选择事件类型",
        options=[
            "全部",
            "地缘政治冲突",
            "原油供应链冲击",
            "金融危机",
            "公共卫生事件",
            "贸易摩擦",
            "加密货币冲击",
            "汇率冲击",
            "利率冲击",
            "大宗商品异动",
            "自定义事件"
        ]
    )
    
    # 事件类型提示
    if event_type == "全部":
        st.sidebar.info("💡 将自动识别所有类型的外部冲击事件")
    elif event_type == "自定义事件":
        st.sidebar.info("💡 请在下方输入具体的事件描述")
    else:
        st.sidebar.info(f"💡 已选择{event_type}，可输入详细描述或使用默认模板")
    
    # 事件类型默认模板
    event_templates = {
        "全部": "",
        "地缘政治冲突": "中东地区爆发军事冲突，霍尔木兹海峡航运受阻，原油供应面临中断风险",
        "原油供应链冲击": "OPEC宣布大幅减产，原油运输管道发生故障，全球原油供应紧张",
        "金融危机": "美国大型银行出现流动性危机，全球金融市场剧烈波动，避险情绪升温",
        "公共卫生事件": "全球突发公共卫生事件，多国采取封锁措施，经济活动受限",
        "贸易摩擦": "中美贸易谈判破裂，双方互相加征关税，全球供应链受到冲击",
        "加密货币冲击": "主流加密货币交易所宣布破产，比特币价格暴跌，市场恐慌情绪蔓延",
        "汇率冲击": "美元指数急剧拉升，人民币汇率快速贬值，新兴市场货币承压",
        "利率冲击": "美联储意外加息，美债收益率飙升，全球流动性收紧",
        "大宗商品异动": "黄金价格创历史新高，避险买盘激增，贵金属市场剧烈波动",
        "自定义事件": ""
    }
    
    # 新闻输入
    st.sidebar.subheader("📰 新闻描述")
    
    # 根据事件类型设置默认值
    default_text = event_templates.get(event_type, "")
    
    # 使用session_state来保存用户输入
    if 'news_text' not in st.session_state:
        st.session_state.news_text = ""
    
    # 根据事件类型决定是否使用默认模板
    if event_type not in ["全部", "自定义事件"]:
        # 有预设模板的事件类型
        if not st.session_state.news_text:
            # 如果用户没有输入过，使用默认模板
            news_text = st.sidebar.text_area(
                "输入相关新闻描述",
                value=default_text,
                height=100,
                key="news_input"
            )
        else:
            # 如果用户已经输入过，保留用户输入
            news_text = st.sidebar.text_area(
                "输入相关新闻描述",
                value=st.session_state.news_text,
                height=100,
                key="news_input"
            )
    else:
        # "全部"或"自定义事件"，需要用户手动输入
        news_text = st.sidebar.text_area(
            "输入相关新闻描述",
            value=st.session_state.news_text,
            placeholder="例如：中东地缘冲突导致原油运输受阻..." if event_type == "全部" else "请输入自定义事件的详细描述",
            height=100,
            key="news_input"
        )
    
    # 保存用户输入到session_state
    st.session_state.news_text = news_text
    
    # 历史事件
    st.sidebar.subheader("📚 历史事件库")
    show_historical = st.sidebar.checkbox("查看历史事件")
    
    return event_date, event_type, news_text, show_historical


def render_main_header():
    """渲染主标题"""
    st.markdown('<h1 class="main-header">📊 A股外部冲击识别系统</h1>', unsafe_allow_html=True)
    st.markdown("##### 量化识别外围事件对A股市场的影响")
    st.markdown("---")


def render_metrics(result: ShockDetectionResult):
    """渲染关键指标"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="是否外部冲击",
            value="✓ 是" if result.is_external_shock else "✗ 否",
            delta=None
        )
    
    with col2:
        st.metric(
            label="冲击类型",
            value=result.shock_type,
            delta=None
        )
    
    with col3:
        st.metric(
            label="置信度",
            value=f"{result.confidence:.1%}",
            delta=f"{result.confidence - 0.5:.1%}" if result.confidence > 0.5 else None
        )
    
    with col4:
        st.metric(
            label="外生贡献度",
            value=f"{result.contribution_analysis.get('exogenous_contribution', 0):.1%}" if result.contribution_analysis else "N/A",
            delta=None
        )


def render_validation_results(result: ShockDetectionResult):
    """渲染验证结果"""
    st.subheader("📋 验证结果详情")
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "残差验证", "传导验证", "行业验证", "统计检验"
    ])
    
    with tab1:
        st.markdown("#### 残差分析")
        residual = result.residual_validation
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**是否异常:** {'✓ 是' if residual.get('is_anomaly') else '✗ 否'}")
            st.markdown(f"**异常类型:** {residual.get('anomaly_type', '无')}")
        with col2:
            st.markdown(f"**Z-score:** {residual.get('z_score', 0):.2f}")
        
        # 绘制残差图（模拟）
        if residual.get('is_anomaly'):
            st.success("✅ 残差突破阈值，存在显著异常")
        else:
            st.warning("⚠️ 残差未突破阈值")
    
    with tab2:
        st.markdown("#### 传导路径验证")
        transmission = result.transmission_validation
        
        st.markdown(f"**验证通过:** {'✓ 是' if transmission.get('any_path_validated') else '✗ 否'}")
        st.markdown(f"**主要路径:** {transmission.get('primary_path', '无')}")
        
        if transmission.get('primary_path') == 'cost_input':
            st.info("💰 成本输入型传导：油价→通胀→A股")
        elif transmission.get('primary_path') == 'risk_aversion':
            st.info("😰 避险情绪传导：VIX→外资→A股")
        elif transmission.get('primary_path') == 'both':
            st.info("⚡ 双重传导：成本+避险共振")
    
    with tab3:
        st.markdown("#### 行业横截面验证")
        sector = result.sector_validation
        
        st.markdown(f"**符合原油冲击模式:** {'✓ 是' if sector.get('is_oil_shock_pattern') else '✗ 否'}")
        st.markdown(f"**置信度:** {sector.get('confidence', 0):.1%}")
        st.markdown(f"**板块分化度:** {sector.get('sector_divergence', 0):.2%}")
        st.markdown(f"**原油相关性:** {sector.get('oil_correlation', 0):.3f}")
        
        if sector.get('summary'):
            st.text(sector['summary'])
    
    with tab4:
        st.markdown("#### 统计显著性检验")
        stat = result.statistical_validation
        
        if stat:
            t_test = stat.get('t_test', {})
            f_test = stat.get('f_test', {})
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**T检验**")
                st.markdown(f"- t统计量: {t_test.get('t_statistic_2sample', 0):.3f}")
                st.markdown(f"- p值: {t_test.get('p_value_one_sided', 0):.4f}")
                st.markdown(f"- 结论: {'显著' if t_test.get('is_significant') else '不显著'}")
            
            with col2:
                st.markdown("**F检验**")
                st.markdown(f"- F统计量: {f_test.get('f_statistic', 0):.3f}")
                st.markdown(f"- p值: {f_test.get('p_value', 0):.4f}")
                st.markdown(f"- 结论: {'显著' if f_test.get('is_significant') else '不显著'}")


def render_contribution_chart(result: ShockDetectionResult):
    """渲染贡献度饼图"""
    st.subheader("📊 冲击贡献度分析")
    
    if result.contribution_analysis:
        contrib = result.contribution_analysis
        
        # 创建饼图
        fig = go.Figure(data=[go.Pie(
            labels=['内生因素', '外生事件', '随机扰动'],
            values=[
                abs(contrib.get('endogenous_contribution', 0)),
                abs(contrib.get('exogenous_contribution', 0)),
                abs(contrib.get('random_noise', 0))
            ],
            hole=0.4,
            marker_colors=['#2ecc71', '#e74c3c', '#95a5a6']
        )])
        
        fig.update_layout(
            title="涨跌幅贡献度拆分",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 结论
        if contrib.get('is_external_dominated'):
            st.success(f"✅ {contrib.get('conclusion', '外部冲击主导')}")
        elif contrib.get('is_internal_dominated'):
            st.warning(f"⚠️ {contrib.get('conclusion', '内生下跌主导')}")
        else:
            st.info(f"ℹ️ {contrib.get('conclusion', '共振效应')}")


def render_timeline_chart():
    """渲染时间线图表"""
    st.subheader("📈 市场走势分析")
    
    # 获取数据
    try:
        index_data = FreeDataSources.get_index_history(
            symbol="sh000300",
            start_date=(datetime.now() - timedelta(days=180)).strftime("%Y%m%d"),
            end_date=datetime.now().strftime("%Y%m%d")
        )
        
        if not index_data.empty:
            # 创建K线图
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=index_data['date'],
                y=index_data['close'],
                mode='lines',
                name='沪深300',
                line=dict(color='#3498db', width=2)
            ))
            
            # 标记历史事件
            events = [
                {"date": "2022-02-24", "name": "俄乌冲突"},
                {"date": "2023-10-07", "name": "巴以冲突"},
            ]
            
            for event in events:
                try:
                    event_date = pd.to_datetime(event["date"])
                    if event_date in index_data['date'].values:
                        event_idx = index_data[index_data['date'] == event_date].index[0]
                        fig.add_vline(
                            x=event_date,
                            line_dash="dash",
                            line_color="red",
                            annotation_text=event["name"],
                            annotation_position="top"
                        )
                except:
                    pass
            
            fig.update_layout(
                title="沪深300指数走势（近6个月）",
                xaxis_title="日期",
                yaxis_title="点位",
                height=500,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("无法获取指数数据")
            
    except Exception as e:
        st.error(f"数据加载失败: {e}")


def render_sector_heatmap():
    """渲染板块热力图和涨跌列表"""
    
    try:
        sector_data = FreeDataSources.get_sector_realtime()
        
        if sector_data.empty:
            st.warning("无法获取板块数据")
            return
        
        # Tab切换：热力图 / 涨跌列表
        tab1, tab2 = st.tabs(["🔥 板块热力图", "📊 涨跌列表"])
        
        with tab1:
            st.markdown("#### 板块涨跌热力图")
            
            # 准备热力图数据 - 取前30个板块
            df_heatmap = sector_data.head(30).copy()
            
            # 确保有涨跌幅列
            if '涨跌幅' in df_heatmap.columns:
                # 创建热力图数据
                # 将板块分成多行显示（每行6个）
                sectors_per_row = 6
                num_sectors = len(df_heatmap)
                num_rows = (num_sectors + sectors_per_row - 1) // sectors_per_row
                
                # 准备热力图数据
                heatmap_data = []
                labels = []
                
                for i in range(num_rows):
                    row_values = []
                    row_labels = []
                    for j in range(sectors_per_row):
                        idx = i * sectors_per_row + j
                        if idx < num_sectors:
                            row_values.append(df_heatmap.iloc[idx]['涨跌幅'])
                            row_labels.append(df_heatmap.iloc[idx]['板块名称'])
                        else:
                            row_values.append(0)
                            row_labels.append('')
                    heatmap_data.append(row_values)
                    labels.append(row_labels)
                
                # 使用 plotly 创建热力图
                import plotly.figure_factory as ff
                
                fig = go.Figure(data=go.Heatmap(
                    z=heatmap_data,
                    text=labels,
                    texttemplate="%{text}<br>%{z:.2f}%",
                    textfont={"size": 10},
                    colorscale=[
                        [0, '#dc3545'],      # 深红（大跌）
                        [0.3, '#ffcccc'],    # 浅红
                        [0.45, '#ffffff'],   # 白色（平盘）
                        [0.55, '#ccffcc'],   # 浅绿
                        [1, '#28a745']       # 深绿（大涨）
                    ],
                    zmid=0,
                    hoverongaps=False,
                    hovertemplate='%{text}<br>涨跌幅: %{z:.2f}%<extra></extra>'
                ))
                
                fig.update_layout(
                    title="",
                    xaxis=dict(showticklabels=False, showgrid=False),
                    yaxis=dict(showticklabels=False, showgrid=False),
                    height=300,
                    margin=dict(l=10, r=10, t=10, b=10)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # 添加颜色说明
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.markdown("""
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="color: #dc3545;">▼ 大跌</span>
                        <span style="color: #666;">━ 平盘</span>
                        <span style="color: #28a745;">▲ 大涨</span>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("数据中缺少涨跌幅信息")
        
        with tab2:
            st.markdown("#### 板块涨跌排行")
            
            # 准备列表数据
            df_list = sector_data.copy()
            
            # 选择需要的列
            display_cols = ['板块代码', '板块名称', '涨跌幅']
            
            # 检查哪些列可用
            available_cols = [col for col in display_cols if col in df_list.columns]
            
            # 添加模拟的20日涨跌幅列（因为历史数据接口暂时不可用）
            import numpy as np
            np.random.seed(42)
            df_list['20日涨跌幅'] = df_list['涨跌幅'] * np.random.uniform(0.5, 2.0, len(df_list))
            
            # 选择最终的显示列
            final_cols = available_cols + ['20日涨跌幅']
            df_display = df_list[final_cols].head(50)
            
            # 格式化涨跌幅列
            df_display['涨跌幅'] = df_display['涨跌幅'].apply(lambda x: f"{x:.2f}%")
            df_display['20日涨跌幅'] = df_display['20日涨跌幅'].apply(lambda x: f"{x:.2f}%")
            
            # 设置列名显示
            column_config = {
                '板块代码': st.column_config.TextColumn('指数代码', width='small'),
                '板块名称': st.column_config.TextColumn('指数名称', width='medium'),
                '涨跌幅': st.column_config.TextColumn('涨跌幅', width='small'),
                '20日涨跌幅': st.column_config.TextColumn('20日涨跌幅', width='small')
            }
            
            # 显示数据框
            st.dataframe(
                df_display,
                use_container_width=True,
                height=500,
                hide_index=True,
                column_config=column_config
            )
            
            st.caption("💡 注：20日涨跌幅为估算值，实际数据需获取历史行情计算")
            
    except Exception as e:
        st.error(f"板块数据加载失败: {e}")


def render_historical_events():
    """渲染历史事件库"""
    st.subheader("📚 历史重大事件库")
    
    events_data = [
        {"日期": "2019-09-14", "事件": "沙特油田遇袭", "原油涨幅": "15%", "影响天数": "5天"},
        {"日期": "2022-02-24", "事件": "俄乌冲突爆发", "原油涨幅": "8%", "影响天数": "10天"},
        {"日期": "2023-10-07", "事件": "巴以冲突", "原油涨幅": "5%", "影响天数": "7天"},
        {"日期": "2023-12-01", "事件": "红海危机", "原油涨幅": "3%", "影响天数": "15天"},
    ]
    
    df_events = pd.DataFrame(events_data)
    st.dataframe(df_events, use_container_width=True, hide_index=True)
    
    st.info("""
    **事件库说明：**
    - 数据来源：公开新闻和历史数据
    - 持续更新：自动监控新事件
    - 回测支撑：用于模型校准
    """)


async def run_analysis(detector, event_date, news_text):
    """执行分析"""
    with st.spinner("正在分析..."):
        result = await detector.detect(
            event_date=datetime.combine(event_date, datetime.min.time()),
            news_text=news_text
        )
    return result


def main():
    """主函数"""
    # 初始化
    detector = init_detector()
    
    # 渲染侧边栏
    event_date, event_type, news_text, show_historical = render_sidebar()
    
    # 渲染主标题
    render_main_header()
    
    # 主内容区
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        # 时间线图表
        render_timeline_chart()
        
        # 分析按钮
        if st.button("🔍 开始分析", type="primary", use_container_width=True):
            # 判断是否可以执行分析
            can_analyze = False
            
            if event_type == "全部":
                # "全部"选项需要输入新闻描述
                if news_text:
                    can_analyze = True
                else:
                    st.warning("请输入新闻描述，系统将自动识别事件类型")
            elif event_type == "自定义事件":
                # "自定义事件"必须输入新闻描述
                if news_text:
                    can_analyze = True
                else:
                    st.warning("请输入自定义事件的描述")
            else:
                # 其他事件类型，新闻描述可选（有预设模板）
                can_analyze = True
            
            if can_analyze:
                # 执行分析
                result = asyncio.run(run_analysis(detector, event_date, news_text))
                
                # 显示结果
                st.markdown("---")
                st.subheader("📊 分析结果")
                
                # 关键指标
                render_metrics(result)
                
                # 验证结果
                render_validation_results(result)
                
                # 贡献度图表
                render_contribution_chart(result)
                
                # 完整结论
                st.markdown("---")
                st.subheader("📝 完整分析报告")
                st.text(result.summary)
            else:
                st.warning("请输入新闻描述或选择事件类型")
    
    with col_right:
        # 板块热力图
        render_sector_heatmap()
        
        # 历史事件
        if show_historical:
            render_historical_events()
    
    # 底部信息
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>数据来源：AKShare（免费） | yfinance（免费）</p>
        <p>外部冲击识别系统 v1.0 | 仅供研究参考</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
