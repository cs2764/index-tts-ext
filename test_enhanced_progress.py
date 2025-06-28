#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IndexTTS 增强进度监控功能测试脚本

此脚本用于测试新增的实时系统监控、时间预测和详细进度显示功能。
"""

import time
import sys
import os

# 添加项目路径到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_system_info():
    """测试系统信息获取功能"""
    print("=== 测试系统信息获取功能 ===")
    
    try:
        # 模拟TTS对象
        class MockTTS:
            def __init__(self):
                import torch
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 模拟系统信息获取
        from webui import get_system_status
        
        # 临时设置全局TTS对象
        import webui
        webui.tts = MockTTS()
        
        system_info = get_system_status()
        print("系统信息获取成功:")
        print(system_info)
        
    except Exception as e:
        print(f"系统信息获取失败: {e}")
        import traceback
        traceback.print_exc()

def test_progress_tracking():
    """测试进度跟踪功能"""
    print("\n=== 测试进度跟踪功能 ===")
    
    try:
        from indextts.infer import IndexTTS
        
        # 创建模拟的TTS实例
        tts = IndexTTS()
        
        # 测试系统信息获取
        system_info = tts.get_system_info()
        print("TTS系统信息:")
        for key, value in system_info.items():
            print(f"  {key}: {value}")
        
        # 测试进度更新
        print("\n测试进度更新功能:")
        start_time = time.perf_counter()
        batch_times = []
        
        for i in range(5):
            batch_start = time.perf_counter()
            
            tts._set_gr_progress(
                value=i/4, 
                desc=f"处理步骤 {i+1}/5", 
                start_time=start_time,
                total_items=5,
                current_item=i,
                batch_times=batch_times
            )
            
            # 模拟不同的处理时间
            processing_time = 1.0 + (i % 3) * 0.5  # 1.0-2.0秒的变化
            time.sleep(processing_time)
            
            # 记录批次时间
            batch_time = time.perf_counter() - batch_start
            batch_times.append(batch_time)
        
        print("进度跟踪测试完成")
        print(f"批次时间记录: {[tts.format_time(t) for t in batch_times]}")
        
    except Exception as e:
        print(f"进度跟踪测试失败: {e}")
        import traceback
        traceback.print_exc()

def test_time_formatting():
    """测试时间格式化功能"""
    print("\n=== 测试时间格式化功能 ===")
    
    try:
        # 创建模拟的TTS实例用于时间格式化
        from indextts.infer import IndexTTS
        tts = IndexTTS()
        
        test_times = [5.5, 30.2, 65.8, 150.5, 3665.2, 7200.0]
        print("时间格式化测试:")
        for seconds in test_times:
            formatted = tts.format_time(seconds)
            print(f"  {seconds:>8.1f}秒 -> {formatted}")
        
    except Exception as e:
        print(f"时间格式化测试失败: {e}")

def test_time_estimation():
    """测试时间估算功能"""
    print("\n=== 测试时间估算功能 ===")
    
    try:
        start_time = time.time()
        total_items = 6
        batch_times = []
        
        print("模拟批次处理中的时间估算:")
        for i in range(1, total_items + 1):
            batch_start = time.time()
            
            elapsed = time.time() - start_time
            
            # 模拟批次处理时间
            processing_time = 0.8 + (i % 3) * 0.3  # 0.8-1.4秒的变化
            time.sleep(processing_time)
            
            batch_time = time.time() - batch_start
            batch_times.append(batch_time)
            
            if len(batch_times) > 0:
                # 使用最近几个批次的平均时间进行预测
                recent_batches = batch_times[-min(3, len(batch_times)):]
                avg_batch_time = sum(recent_batches) / len(recent_batches)
                remaining_batches = total_items - i
                estimated_remaining = avg_batch_time * remaining_batches
                
                print(f"批次 {i}/{total_items}:")
                print(f"  当前批次时间: {batch_time:.1f}秒")
                print(f"  平均批次时间: {avg_batch_time:.1f}秒")
                print(f"  已用时: {elapsed:.1f}秒")
                print(f"  预计剩余: {estimated_remaining:.1f}秒")
                print(f"  预计总时长: {elapsed + estimated_remaining:.1f}秒")
                print()
        
        total_time = time.time() - start_time
        print(f"实际总时长: {total_time:.1f}秒")
        print(f"批次时间记录: {[f'{t:.1f}s' for t in batch_times]}")
        
    except Exception as e:
        print(f"时间估算测试失败: {e}")

def test_memory_monitoring():
    """测试内存监控功能"""
    print("\n=== 测试内存监控功能 ===")
    
    try:
        import psutil
        import torch
        
        print("系统内存信息:")
        memory = psutil.virtual_memory()
        print(f"  总内存: {memory.total / 1024**3:.2f}GB")
        print(f"  已使用: {memory.used / 1024**3:.2f}GB")
        print(f"  使用率: {memory.percent:.1f}%")
        
        print("\nCPU信息:")
        cpu_percent = psutil.cpu_percent(interval=1.0)
        print(f"  使用率: {cpu_percent:.1f}%")
        
        if torch.cuda.is_available():
            print("\nGPU信息:")
            device = torch.device("cuda")
            total_memory = torch.cuda.get_device_properties(device).total_memory / 1024**3
            allocated_memory = torch.cuda.memory_allocated(device) / 1024**3
            reserved_memory = torch.cuda.memory_reserved(device) / 1024**3
            
            print(f"  设备: {torch.cuda.get_device_name(device)}")
            print(f"  总显存: {total_memory:.2f}GB")
            print(f"  已分配: {allocated_memory:.2f}GB")
            print(f"  已缓存: {reserved_memory:.2f}GB")
            print(f"  使用率: {(allocated_memory / total_memory) * 100:.1f}%")
        else:
            print("\nGPU: 未检测到CUDA设备")
        
    except Exception as e:
        print(f"内存监控测试失败: {e}")

def main():
    """主测试函数"""
    print("IndexTTS 增强进度监控功能测试")
    print("=" * 50)
    
    # 检查依赖
    try:
        import psutil
        print("✅ psutil 依赖检查通过")
    except ImportError:
        print("❌ 缺少 psutil 依赖，请运行: pip install psutil")
        return
    
    try:
        import torch
        print("✅ PyTorch 依赖检查通过")
    except ImportError:
        print("❌ 缺少 PyTorch 依赖")
        return
    
    # 运行测试
    test_memory_monitoring()
    test_time_formatting()
    test_time_estimation()
    
    # 只有当IndexTTS可用时才测试
    try:
        test_system_info()
        test_progress_tracking()
    except Exception as e:
        print(f"\n⚠️  IndexTTS相关测试跳过: {e}")
        print("这是正常的，如果您还没有加载TTS模型")
    
    print("\n" + "=" * 50)
    print("测试完成！")
    print("\n新功能说明：")
    print("1. 实时系统监控 - 强制刷新确保显存使用情况实时更新")
    print("2. 智能时间格式化 - 自动换算为秒/分钟/小时，提升可读性")
    print("3. 批次时间预测 - 基于最近批次处理时间的智能预测算法") 
    print("4. 详细进度信息 - 包含当前批次时间、平均批次时间等详细信息")
    print("5. 优化的界面布局 - 分离的进度和系统信息区域")
    print("6. 增强的后台任务监控 - 包含格式化时间信息的任务状态更新")

if __name__ == "__main__":
    main() 