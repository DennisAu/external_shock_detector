@echo off
chcp 65001
echo ========================================
echo 安装外部冲击识别系统依赖
echo ========================================
echo.

echo [1/3] 安装核心依赖...
pip install numpy pandas scipy scikit-learn -q
echo 完成

echo.
echo [2/3] 安装数据源依赖...
pip install akshare yfinance -q
echo 完成

echo.
echo [3/3] 安装可视化和其他依赖...
pip install streamlit plotly loguru statsmodels -q
echo 完成

echo.
echo ========================================
echo 安装完成！
echo ========================================
echo.
echo 运行测试: python test_system.py
echo 启动界面: streamlit run app.py
echo.
pause
