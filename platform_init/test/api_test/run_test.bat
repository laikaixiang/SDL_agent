@echo off
REM ========================================
REM 自适应流式处理器测试脚本 (Windows批处理)
REM ========================================
REM
REM 用途：
REM   自动激活SDL_agent conda环境并运行自适应流式处理器测试
REM
REM 功能：
REM   1. 激活SDL_agent conda环境
REM   2. 运行test_adaptive_stream.py测试脚本
REM   3. 测试API是否支持流式响应
REM   4. 显示测试结果
REM
REM 使用方法：
REM   双击运行此文件，或在命令行中执行：
REM   cd test/api_test
REM   run_test.bat
REM
REM 测试内容：
REM   - 检测API流式响应支持
REM   - 测试非流式响应
REM   - 测试流式响应（如果支持）
REM   - 显示状态信息
REM
REM ========================================

echo ========================================
echo 自适应流式处理器测试
echo ========================================
echo.
echo 正在激活 SDL_agent 环境...
call conda activate SDL_agent
echo.
echo 运行测试...
python test/api_test/test_adaptive_stream.py
echo.
pause
