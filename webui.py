#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import sys
import threading
import time
import datetime
import shutil
import tempfile
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
import uuid

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Windows系统优化：设置事件循环策略以避免连接重置错误
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, "indextts"))

import argparse
parser = argparse.ArgumentParser(description="IndexTTS WebUI")
parser.add_argument("--verbose", action="store_true", default=False, help="Enable verbose mode")
parser.add_argument("--port", type=int, default=7866, help="Port to run the web UI on")
parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the web UI on")
parser.add_argument("--model_dir", type=str, default="checkpoints", help="Model checkpoints directory")
cmd_args = parser.parse_args()

if not os.path.exists(cmd_args.model_dir):
    print(f"Model directory {cmd_args.model_dir} does not exist. Please download the model first.")
    sys.exit(1)

for file in [
    "bigvgan_generator.pth",
    "bpe.model",
    "gpt.pth",
    "config.yaml",
]:
    file_path = os.path.join(cmd_args.model_dir, file)
    if not os.path.exists(file_path):
        print(f"Required file {file_path} does not exist. Please download it.")
        sys.exit(1)

import gradio as gr
import pandas as pd

from indextts.infer import IndexTTS
from tools.i18n.i18n import I18nAuto

i18n = I18nAuto(language="zh_CN")
MODE = 'local'
tts = IndexTTS(model_dir=cmd_args.model_dir, cfg_path=os.path.join(cmd_args.model_dir, "config.yaml"),)

# 后台任务管理
task_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="TTS_Task")
task_queue = Queue()
task_status = {}  # 任务状态字典: {task_id: {"status": str, "progress": str, "result": str, "error": str}}
task_lock = threading.Lock()

os.makedirs("outputs", exist_ok=True)
os.makedirs("prompts",exist_ok=True)
os.makedirs("samples",exist_ok=True)

# Helper functions for file processing
def read_txt_file(file_path):
    """Read text from txt file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except:
        try:
            with open(file_path, 'r', encoding='gbk') as f:
                return f.read()
        except:
            return ""

def read_epub_file(file_path):
    """Read text from epub file"""
    try:
        import ebooklib
        from ebooklib import epub
        from bs4 import BeautifulSoup
        import re
        
        book = epub.read_epub(file_path)
        text_content = []
        chapters = []
        
        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                soup = BeautifulSoup(item.get_content(), 'html.parser')
                content = soup.get_text()
                
                # Try to extract chapter title from the content or filename
                chapter_title = None
                
                # Look for chapter titles in h1, h2, h3 tags
                for tag in soup.find_all(['h1', 'h2', 'h3']):
                    title_text = tag.get_text().strip()
                    if title_text and (re.search(r'第.{1,10}章|chapter|Chapter|CHAPTER', title_text) or len(title_text) < 100):
                        chapter_title = title_text
                        break
                
                # If no title found, use filename or create generic title
                if not chapter_title:
                    filename = getattr(item, 'file_name', '')
                    if filename:
                        chapter_title = filename.replace('.xhtml', '').replace('.html', '')
                    else:
                        chapter_title = f"章节 {len(chapters) + 1}"
                
                # Record chapter start position
                current_length = len('\n'.join(text_content))
                if content.strip():  # Only add if content is not empty
                    chapters.append({
                        'title': chapter_title,
                        'start_pos': current_length,
                        'content': content
                    })
                    text_content.append(content)
        
        full_text = '\n'.join(text_content)
        return full_text, chapters
        
    except ImportError:
        return "需要安装 ebooklib 和 beautifulsoup4 才能读取 EPUB 文件", []
    except:
        return "读取 EPUB 文件失败", []

def convert_audio_format(input_path, output_path, format_type="mp3", bitrate="64k", chapters=None):
    """Convert audio to different formats with optional chapter support"""
    try:
        import subprocess
        import tempfile
        import os
        
        print(f">> 开始音频格式转换:")
        print(f"   输入文件: {input_path}")
        print(f"   输出文件: {output_path}")
        print(f"   格式: {format_type}")
        print(f"   比特率: {bitrate}")
        print(f"   章节数: {len(chapters) if chapters else 0}")
        
        # Check if input file exists
        if not os.path.exists(input_path):
            print(f"❌ 错误: 输入文件不存在: {input_path}")
            return False
        
        # Check if ffmpeg is available
        try:
            subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True, encoding='utf-8', errors='ignore')
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("❌ 错误: ffmpeg 未安装或不在PATH中")
            return False
        
        chapter_file_path = None
        
        if format_type == "mp3":
            cmd = ["ffmpeg", "-i", input_path, "-b:a", bitrate, "-y", output_path]
        elif format_type == "m4b":
            # Start with basic input
            cmd = ["ffmpeg", "-i", input_path]
            
            # Add chapters if provided
            if chapters and len(chapters) > 1:
                print(f">> 创建章节元数据文件，共 {len(chapters)} 个章节")
                # Create a temporary chapter file
                with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8', newline='') as chapter_file:
                    chapter_file.write(";FFMETADATA1\n")
                    
                    # Calculate approximate chapter timings
                    # This is a rough estimation based on text length
                    total_chars = sum(len(ch.get('content', '')) for ch in chapters)
                    # Estimate total duration: ~150 characters per minute
                    total_duration = (total_chars / 150) * 60 * 1000  # milliseconds
                    current_time = 0
                    
                    print(f"   总字符数: {total_chars}")
                    print(f"   预计总时长: {total_duration/1000:.1f}秒")
                    
                    for i, chapter in enumerate(chapters):
                        chapter_chars = len(chapter.get('content', ''))
                        # Estimate: ~150 characters per minute (rough speaking rate)
                        chapter_duration = (chapter_chars / 150) * 60 * 1000  # milliseconds
                        
                        start_time = int(current_time)
                        chapter_file.write(f"\n[CHAPTER]\n")
                        chapter_file.write(f"TIMEBASE=1/1000\n")
                        chapter_file.write(f"START={start_time}\n")
                        
                        # End time for current chapter
                        if i < len(chapters) - 1:
                            end_time = int(current_time + chapter_duration)
                        else:
                            # Last chapter goes to end of estimated total duration
                            end_time = int(total_duration)
                        
                        chapter_file.write(f"END={end_time}\n")
                        
                        # Clean chapter title for metadata
                        title = chapter.get('title', f'Chapter {i+1}')
                        title = title.replace('=', '\\=').replace(';', '\\;').replace('#', '\\#')
                        chapter_file.write(f"title={title}\n")
                        
                        print(f"   章节 {i+1}: {title} ({start_time}ms - {end_time}ms)")
                        
                        current_time += chapter_duration
                    
                    chapter_file_path = chapter_file.name
                    print(f"   章节文件: {chapter_file_path}")
                
                # Add metadata file to ffmpeg command
                cmd.extend(["-i", chapter_file_path, "-map_metadata", "1"])
            
            # Add output options
            cmd.extend(["-c:a", "aac", "-b:a", bitrate, "-f", "ipod", "-y", output_path])
        else:
            print(f"❌ 错误: 不支持的格式: {format_type}")
            return False
        
        print(f">> 执行 ffmpeg 命令: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='ignore')
        
        print(f">> ffmpeg 返回码: {result.returncode}")
        if result.stdout:
            print(f">> ffmpeg 输出: {result.stdout}")
        if result.stderr:
            print(f">> ffmpeg 错误: {result.stderr}")
        
        # Clean up temporary chapter file
        if chapter_file_path:
            try:
                os.unlink(chapter_file_path)
                print(f">> 已删除临时章节文件: {chapter_file_path}")
            except Exception as e:
                print(f">> 删除临时文件失败: {e}")
        
        success = result.returncode == 0
        if success:
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path) / 1024 / 1024  # MB
                print(f"✅ 格式转换成功: {output_path} ({file_size:.2f} MB)")
            else:
                print(f"❌ 转换失败: 输出文件不存在")
                success = False
        else:
            print(f"❌ ffmpeg 转换失败，返回码: {result.returncode}")
        
        return success
        
    except Exception as e:
        print(f"❌ 格式转换异常: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def get_sample_files():
    """Get list of sample audio files"""
    samples_dir = "samples"
    if not os.path.exists(samples_dir):
        os.makedirs(samples_dir)
        return []
    
    # Only include standard audio formats that can be loaded by torchaudio
    supported_formats = ['.wav', '.mp3']
    files = []
    
    for file in os.listdir(samples_dir):
        if any(file.lower().endswith(ext) for ext in supported_formats):
            files.append(file)
    
    return sorted(files)

def get_speaker_name_from_path(audio_path):
    """Extract speaker name from audio file path"""
    if audio_path:
        # Handle different formats that Gradio Audio might return
        if isinstance(audio_path, tuple):
            # If it's a tuple, take the first element (usually the file path)
            audio_path = audio_path[0] if audio_path else None
        
        if audio_path and isinstance(audio_path, str):
            filename = os.path.basename(audio_path)
            name = os.path.splitext(filename)[0]
            return name
    return "unknown"

# 后台任务处理系统
def update_task_status(task_id, status=None, progress=None, result=None, error=None):
    """更新任务状态"""
    with task_lock:
        if task_id not in task_status:
            task_status[task_id] = {"status": "unknown", "progress": "", "result": "", "error": ""}
        
        if status is not None:
            task_status[task_id]["status"] = status
        if progress is not None:
            task_status[task_id]["progress"] = progress
        if result is not None:
            task_status[task_id]["result"] = result
        if error is not None:
            task_status[task_id]["error"] = error

def get_task_status(task_id):
    """获取任务状态"""
    with task_lock:
        return task_status.get(task_id, {"status": "not_found", "progress": "", "result": "", "error": "任务不存在"})

def background_audio_generation(task_id, prompt_path, text_to_process, infer_mode, 
                               max_text_tokens_per_sentence, sentences_bucket_max_size,
                               audio_format, output_path, temp_wav_path, chapters, kwargs):
    """后台音频生成函数"""
    try:
        print(f"=== 后台任务 {task_id} 开始 ===")
        update_task_status(task_id, status="🚀 初始化", progress="正在准备生成参数...")
        
        # 创建任务专用的进度回调
        class TaskProgress:
            def __init__(self, task_id):
                self.task_id = task_id
                self.last_update = time.time()
                self.last_system_update = time.time()
                self.start_time = time.time()
            
            def format_time(self, seconds):
                """格式化时间为人类可读格式"""
                if seconds < 60:
                    return f"{seconds:.1f}秒"
                elif seconds < 3600:
                    minutes = seconds / 60
                    return f"{minutes:.1f}分钟"
                else:
                    hours = seconds / 3600
                    return f"{hours:.1f}小时"
            
            def __call__(self, progress=None, desc=None):
                current_time = time.time()
                # 限制更新频率，避免过于频繁的状态更新
                if current_time - self.last_update > 1.0:  # 每秒最多更新一次
                    progress_text = ""
                    if desc:
                        progress_text = f"🎵 {desc}"
                        
                        # 添加时间信息
                        elapsed = current_time - self.start_time
                        elapsed_formatted = self.format_time(elapsed)
                        progress_text += f"\n⏱️ 已用时: {elapsed_formatted}"
                        
                    elif progress is not None:
                        elapsed = current_time - self.start_time
                        elapsed_formatted = self.format_time(elapsed)
                        progress_text = f"🎵 进度: {progress:.1f}%\n⏱️ 已用时: {elapsed_formatted}"
                    
                    # 每5秒添加一次系统信息
                    if current_time - self.last_system_update > 5.0:
                        try:
                            system_info = get_system_status()
                            progress_text += f"\n\n📊 系统状态:\n{system_info}"
                            self.last_system_update = current_time
                        except:
                            pass  # 忽略系统信息获取错误
                    
                    update_task_status(self.task_id, progress=progress_text)
                    self.last_update = current_time
        
        # 设置任务专用的进度回调
        task_progress = TaskProgress(task_id)
        tts.gr_progress = task_progress
        
        # 开始音频生成
        update_task_status(task_id, status="🎵 生成音频", progress="正在生成音频，请耐心等待...")
        print(f"[Task {task_id}] 开始音频生成...")
        
        start_time = time.time()
        if infer_mode == "普通推理":
            wav_output = tts.infer(prompt_path, text_to_process, temp_wav_path, verbose=cmd_args.verbose,
                               max_text_tokens_per_sentence=int(max_text_tokens_per_sentence),
                               **kwargs)
        else:
            # 批次推理
            wav_output = tts.infer_fast(prompt_path, text_to_process, temp_wav_path, verbose=cmd_args.verbose,
                max_text_tokens_per_sentence=int(max_text_tokens_per_sentence),
                sentences_bucket_max_size=(sentences_bucket_max_size),
                **kwargs)
        
        generation_time = time.time() - start_time
        print(f"[Task {task_id}] 音频生成完成: {wav_output}")
        print(f"[Task {task_id}] 生成耗时: {generation_time:.2f} 秒")
        
        # 格式转换
        final_output = wav_output
        if audio_format != "WAV":
            update_task_status(task_id, status="🔄 格式转换", progress=f"正在转换到 {audio_format} 格式...")
            print(f"[Task {task_id}] 转换音频格式到 {audio_format}...")
            
            if audio_format == "MP3":
                if convert_audio_format(wav_output, output_path, "mp3", "64k"):
                    final_output = output_path
                    # Remove temp wav file
                    if os.path.exists(temp_wav_path) and temp_wav_path != output_path:
                        os.remove(temp_wav_path)
            elif audio_format == "M4B":
                conversion_success = convert_audio_format(wav_output, output_path, "m4b", "64k", chapters)
                if conversion_success:
                    final_output = output_path
                    print(f"[Task {task_id}] ✅ M4B转换成功: {output_path}")
                    # Remove temp wav file
                    if os.path.exists(temp_wav_path) and temp_wav_path != output_path:
                        os.remove(temp_wav_path)
                        print(f"[Task {task_id}] >> 已删除临时WAV文件: {temp_wav_path}")
                else:
                    print(f"[Task {task_id}] ❌ M4B转换失败，保留原WAV文件: {wav_output}")
        
        # 任务完成
        file_size = os.path.getsize(final_output) / 1024 / 1024  # MB
        success_info = f"✅ 生成完成！\n⏱️ 总耗时: {generation_time:.2f} 秒\n📁 文件: {os.path.basename(final_output)}\n📏 大小: {file_size:.2f} MB"
        
        update_task_status(task_id, 
                         status="✅ 完成", 
                         progress=success_info,
                         result=final_output)
        
        print(f"=== 后台任务 {task_id} 完成 ===")
        return final_output
        
    except Exception as e:
        error_msg = f"❌ 生成音频时发生错误: {str(e)}"
        print(f"[Task {task_id}] ERROR: {error_msg}")
        import traceback
        error_traceback = traceback.format_exc()
        print(error_traceback)
        
        detailed_error = f"错误类型: {type(e).__name__}\n错误信息: {str(e)}\n\n详细堆栈:\n{error_traceback}"
        update_task_status(task_id, 
                         status="❌ 失败", 
                         error=detailed_error)
        return None

def submit_background_task(prompt_path, text_to_process, infer_mode, 
                          max_text_tokens_per_sentence, sentences_bucket_max_size,
                          audio_format, output_path, temp_wav_path, chapters, kwargs):
    """提交后台任务"""
    task_id = str(uuid.uuid4())[:8]  # 生成短任务ID
    
    # 初始化任务状态
    update_task_status(task_id, status="⏳ 排队中", progress="任务已提交，等待处理...")
    
    # 提交任务到线程池
    future = task_executor.submit(
        background_audio_generation,
        task_id, prompt_path, text_to_process, infer_mode,
        max_text_tokens_per_sentence, sentences_bucket_max_size,
        audio_format, output_path, temp_wav_path, chapters, kwargs
    )
    
    print(f"已提交后台任务: {task_id}")
    return task_id

def get_all_tasks():
    """获取所有任务状态"""
    with task_lock:
        return dict(task_status)

def clear_completed_tasks():
    """清理已完成的任务"""
    with task_lock:
        completed_tasks = []
        for task_id, status in task_status.items():
            if status["status"] in ["✅ 完成", "❌ 失败"]:
                completed_tasks.append(task_id)
        
        for task_id in completed_tasks:
            del task_status[task_id]
        
        return len(completed_tasks)

with open("tests/cases.jsonl", "r", encoding="utf-8") as f:
    example_cases = []
    for line in f:
        line = line.strip()
        if not line:
            continue
        example = json.loads(line)
        example_cases.append([os.path.join("tests", example.get("prompt_audio", "sample_prompt.wav")),
                              example.get("text"), ["普通推理", "批次推理"][example.get("infer_mode", 0)]])

def gen_single(prompt, text, infer_mode, max_text_tokens_per_sentence=120, sentences_bucket_max_size=6,
                auto_save=True, audio_format="MP3", uploaded_file_name="", selected_sample="", full_text="", chapters=None, 
                background_mode=True, *args, progress=gr.Progress()):
    
    def update_status(status, detailed_progress="", system_status="", error_msg="", show_progress=False, show_system=False, show_error=False):
        """更新状态显示"""
        return (
            gr.update(value=status),
            gr.update(value=detailed_progress, visible=show_progress),
            gr.update(value=system_status, visible=show_system),
            gr.update(value=error_msg, visible=show_error)
        )
    try:
        print(f"=== 开始音频生成 ===")
        print(f"输入参数: prompt={prompt}, text={text[:50] if text else None}{'...' if text and len(text) > 50 else ''}")
        print(f"推理模式: {infer_mode}, 格式: {audio_format}, 自动保存: {auto_save}")
        
        # 初始状态更新
        initial_system_info = get_system_status()
        status_updates = update_status("🚀 开始音频生成...", "正在验证输入参数...", initial_system_info, show_progress=True, show_system=True)
        
        # Handle audio input - prioritize selected sample over audio component
        prompt_path = None
        
        # First, try to use selected sample file
        if selected_sample and selected_sample != "无可用文件":
            prompt_path = os.path.join("samples", selected_sample)
            print(f"使用选中的样本文件: {selected_sample}")
        
        # If no sample selected, try to handle direct audio upload (if any)
        elif prompt is not None:
            if isinstance(prompt, tuple) and len(prompt) == 2:
                # Audio data format: (sample_rate, audio_array)
                # For now, we need to save this to a temporary file
                import numpy as np
                import soundfile as sf
                
                sample_rate, audio_data = prompt
                print(f"接收到音频数据: 采样率={sample_rate}, 数据形状={audio_data.shape}")
                
                # Create temporary wav file
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                    # Convert int16 to float32 if needed
                    if audio_data.dtype == np.int16:
                        audio_data = audio_data.astype(np.float32) / 32768.0
                    sf.write(tmp_file.name, audio_data, sample_rate)
                    prompt_path = tmp_file.name
                    print(f"已保存临时音频文件: {prompt_path}")
            elif isinstance(prompt, str):
                prompt_path = prompt
        
        print(f"最终参考音频路径: {prompt_path}")
        
        # Validate inputs
        if not prompt_path:
            error_msg = "❌ 错误：未选择参考音频文件"
            print(error_msg)
            status_updates = update_status("❌ 生成失败", "", "", error_msg, show_error=True)
            return gr.update(value=None, visible=True), *status_updates
        
        if not os.path.exists(prompt_path):
            error_msg = f"❌ 错误：参考音频文件不存在: {prompt_path}"
            print(error_msg)
            status_updates = update_status("❌ 生成失败", "", "", error_msg, show_error=True)
            return gr.update(value=None, visible=True), *status_updates
        
        # Check if TTS model is loaded
        if not hasattr(tts, 'infer') or not hasattr(tts, 'tokenizer'):
            error_msg = "❌ 错误：TTS模型未正确加载"
            print(error_msg)
            status_updates = update_status("❌ 生成失败", "", "", error_msg, show_error=True)
            return gr.update(value=None, visible=True), *status_updates
    

    
        # Validate text input
        text_to_process = full_text if full_text.strip() else text
        if not text_to_process or not text_to_process.strip():
            error_msg = "❌ 错误：未输入文本内容"
            print(error_msg)
            status_updates = update_status("❌ 生成失败", "", "", error_msg, show_error=True)
            return gr.update(value=None, visible=True), *status_updates
        
        print(f"待处理文本长度: {len(text_to_process)}")
        
        # 检查是否需要后台处理
        text_length = len(text_to_process)
        estimated_time = text_length / 100  # 粗略估算：每100字符约1秒
        
        # 如果文本很长（超过5000字符）或预计时间超过60秒，强制使用后台模式
        if background_mode and (text_length > 5000 or estimated_time > 60):
            print(f"文本较长({text_length}字符)，预计耗时{estimated_time:.1f}秒，切换到后台处理模式")
            
            # 准备后台任务参数
            # 更新状态：准备生成
            detailed_info = f"📝 文本长度: {len(text_to_process)} 字符\n🎵 参考音频: {os.path.basename(prompt_path)}\n⚙️ 推理模式: {infer_mode}\n📁 输出格式: {audio_format}\n⏰ 预计耗时: {estimated_time:.1f}秒"
            current_system_info = get_system_status()
            status_updates = update_status("📋 准备后台任务...", detailed_info, current_system_info, show_progress=True, show_system=True)
            
            # Generate date and speaker name
            date = datetime.datetime.now().strftime("%Y%m%d")
            speaker_name = get_speaker_name_from_path(prompt_path)
            
            # Create output filename based on source
            if uploaded_file_name:
                # If text comes from uploaded file: 文件名_日期_音色
                base_name = os.path.splitext(uploaded_file_name)[0]
                filename = f"{base_name}_{date}_{speaker_name}"
            else:
                # Regular text input: 日期_音色
                filename = f"{date}_{speaker_name}"
            
            # Set output path with proper extension
            if audio_format == "MP3":
                output_path = os.path.join("outputs", f"{filename}.mp3")
                temp_wav_path = os.path.join("outputs", f"temp_{date}_{int(time.time())}.wav")
            elif audio_format == "M4B":
                output_path = os.path.join("outputs", f"{filename}.m4b")
                temp_wav_path = os.path.join("outputs", f"temp_{date}_{int(time.time())}.wav")
            else:  # WAV
                output_path = os.path.join("outputs", f"{filename}.wav")
                temp_wav_path = output_path
            
            # Ensure outputs directory exists
            os.makedirs("outputs", exist_ok=True)
            
            # 准备生成参数
            do_sample, top_p, top_k, temperature, \
                length_penalty, num_beams, repetition_penalty, max_mel_tokens = args
            kwargs = {
                "do_sample": bool(do_sample),
                "top_p": float(top_p),
                "top_k": int(top_k) if int(top_k) > 0 else None,
                "temperature": float(temperature),
                "length_penalty": float(length_penalty),
                "num_beams": num_beams,
                "repetition_penalty": float(repetition_penalty),
                "max_mel_tokens": int(max_mel_tokens),
            }
            
            # 提交后台任务
            task_id = submit_background_task(
                prompt_path, text_to_process, infer_mode,
                max_text_tokens_per_sentence, sentences_bucket_max_size,
                audio_format, output_path, temp_wav_path, chapters, kwargs
            )
            
            # 返回任务信息
            task_info = f"🚀 后台任务已提交\n📋 任务ID: {task_id}\n📝 文本长度: {text_length} 字符\n⏰ 预计耗时: {estimated_time:.1f}秒\n\n💡 您可以关闭浏览器，任务将继续在后台运行。\n📁 完成后文件将保存到: {os.path.basename(output_path)}\n\n🔄 请前往【任务管理】页面查看进度。"
            final_system_info = get_system_status()
            status_updates = update_status("🚀 后台任务已提交", task_info, final_system_info, show_progress=True, show_system=True)
            
            # Clean up temporary prompt file if it was created
            if prompt_path and tempfile.gettempdir() in prompt_path:
                try:
                    os.remove(prompt_path)
                    print(f"已清理临时参考音频文件: {prompt_path}")
                except:
                    pass  # Ignore cleanup errors
            
            return gr.update(value=None, visible=True), *status_updates
        
        # 继续前台处理（短文本）
        # 更新状态：准备生成
        detailed_info = f"📝 文本长度: {len(text_to_process)} 字符\n🎵 参考音频: {os.path.basename(prompt_path)}\n⚙️ 推理模式: {infer_mode}\n📁 输出格式: {audio_format}"
        prep_system_info = get_system_status()
        status_updates = update_status("📋 准备生成参数...", detailed_info, prep_system_info, show_progress=True, show_system=True)
        
        # Generate date and speaker name
        date = datetime.datetime.now().strftime("%Y%m%d")
        speaker_name = get_speaker_name_from_path(prompt_path)
        
        # Create output filename based on source
        if uploaded_file_name:
            # If text comes from uploaded file: 文件名_日期_音色
            base_name = os.path.splitext(uploaded_file_name)[0]
            filename = f"{base_name}_{date}_{speaker_name}"
        else:
            # Regular text input: 日期_音色
            filename = f"{date}_{speaker_name}"
        
        print(f"输出文件名: {filename}")
        
        # Set output path with proper extension
        if audio_format == "MP3":
            output_path = os.path.join("outputs", f"{filename}.mp3")
            temp_wav_path = os.path.join("outputs", f"temp_{date}_{int(time.time())}.wav")  # Temp file without dot prefix
        elif audio_format == "M4B":
            output_path = os.path.join("outputs", f"{filename}.m4b")
            temp_wav_path = os.path.join("outputs", f"temp_{date}_{int(time.time())}.wav")  # Temp file without dot prefix
        else:  # WAV
            output_path = os.path.join("outputs", f"{filename}.wav")
            temp_wav_path = output_path
        
        print(f"输出路径: {output_path}")
        
        # Ensure outputs directory exists
        os.makedirs("outputs", exist_ok=True)
        
        # 创建增强的进度回调
        class EnhancedProgress:
            def __init__(self, original_progress, update_func):
                self.original_progress = original_progress
                self.update_func = update_func
                self.last_system_update = time.time()
                self.start_time = time.time()
            
            def format_time(self, seconds):
                """格式化时间为人类可读格式"""
                if seconds < 60:
                    return f"{seconds:.1f}秒"
                elif seconds < 3600:
                    minutes = seconds / 60
                    return f"{minutes:.1f}分钟"
                else:
                    hours = seconds / 3600
                    return f"{hours:.1f}小时"
            
            def __call__(self, value, desc=None):
                current_time = time.time()
                # 每2秒更新一次系统信息，避免过于频繁
                if current_time - self.last_system_update > 2.0:
                    try:
                        system_info = get_system_status()
                        # 添加时间信息到描述中
                        elapsed = current_time - self.start_time
                        elapsed_formatted = self.format_time(elapsed)
                        
                        enhanced_desc = desc or "正在生成音频..."
                        enhanced_desc += f"\n⏱️ 已用时: {elapsed_formatted}"
                        
                        self.update_func("🎵 正在生成音频...", enhanced_desc, system_info, show_progress=True, show_system=True)
                        self.last_system_update = current_time
                    except:
                        pass  # 忽略系统信息更新错误
                
                # 调用原始进度回调
                if self.original_progress:
                    self.original_progress(value, desc=desc)
        
        # 设置增强的进度回调
        enhanced_progress = EnhancedProgress(progress, update_status)
        tts.gr_progress = enhanced_progress
        do_sample, top_p, top_k, temperature, \
            length_penalty, num_beams, repetition_penalty, max_mel_tokens = args
        kwargs = {
            "do_sample": bool(do_sample),
            "top_p": float(top_p),
            "top_k": int(top_k) if int(top_k) > 0 else None,
            "temperature": float(temperature),
            "length_penalty": float(length_penalty),
            "num_beams": num_beams,
            "repetition_penalty": float(repetition_penalty),
            "max_mel_tokens": int(max_mel_tokens),
        }
        
        print(f"生成参数: {kwargs}")
        
        # 更新状态：开始生成
        generation_info = f"🎯 推理模式: {infer_mode}\n📊 最大Token数: {max_text_tokens_per_sentence}"
        if infer_mode == "批次推理":
            generation_info += f"\n🗂️ 分桶大小: {sentences_bucket_max_size}"
        generation_info += f"\n🎛️ 温度: {kwargs.get('temperature', 1.0)}\n🎲 采样: {'开启' if kwargs.get('do_sample', True) else '关闭'}"
        gen_system_info = get_system_status()
        status_updates = update_status("🎵 正在生成音频...", generation_info, gen_system_info, show_progress=True, show_system=True)
        
        # Generate audio
        print(f"开始音频生成...")
        start_time = time.time()
        if infer_mode == "普通推理":
            wav_output = tts.infer(prompt_path, text_to_process, temp_wav_path, verbose=cmd_args.verbose,
                               max_text_tokens_per_sentence=int(max_text_tokens_per_sentence),
                               **kwargs)
        else:
            # 批次推理
            wav_output = tts.infer_fast(prompt_path, text_to_process, temp_wav_path, verbose=cmd_args.verbose,
                max_text_tokens_per_sentence=int(max_text_tokens_per_sentence),
                sentences_bucket_max_size=(sentences_bucket_max_size),
                **kwargs)
        
        generation_time = time.time() - start_time
        print(f"音频生成完成: {wav_output}")
        print(f"生成耗时: {generation_time:.2f} 秒")
        
        # Convert audio format if needed
        final_output = wav_output
        if auto_save and audio_format != "WAV":
            print(f"转换音频格式到 {audio_format}...")
            # 更新状态：格式转换
            conversion_info = f"🔄 正在转换到 {audio_format} 格式...\n📁 输出路径: {output_path}"
            if audio_format == "M4B" and chapters:
                conversion_info += f"\n📖 添加章节书签: {len(chapters)} 个章节"
            conv_system_info = get_system_status()
            status_updates = update_status("🔄 转换音频格式...", conversion_info, conv_system_info, show_progress=True, show_system=True)
            
            if audio_format == "MP3":
                if convert_audio_format(wav_output, output_path, "mp3", "64k"):
                    final_output = output_path
                    # Remove temp wav file
                    if os.path.exists(temp_wav_path) and temp_wav_path != output_path:
                        os.remove(temp_wav_path)
            elif audio_format == "M4B":
                # Pass chapters info for M4B format to add bookmarks
                print(f">> 开始M4B格式转换...")
                print(f"   输入文件: {wav_output}")
                print(f"   输出文件: {output_path}")
                print(f"   章节信息: {chapters}")
                
                conversion_success = convert_audio_format(wav_output, output_path, "m4b", "64k", chapters)
                if conversion_success:
                    final_output = output_path
                    print(f"✅ M4B转换成功: {output_path}")
                    # Remove temp wav file
                    if os.path.exists(temp_wav_path) and temp_wav_path != output_path:
                        os.remove(temp_wav_path)
                        print(f">> 已删除临时WAV文件: {temp_wav_path}")
                else:
                    print(f"❌ M4B转换失败，保留原WAV文件: {wav_output}")
                    # Keep the original WAV file if conversion fails
        
        print(f"最终输出文件: {final_output}")
        print(f"=== 音频生成完成 ===")
        
        # 最终成功状态
        success_info = f"✅ 生成成功！\n⏱️ 总耗时: {generation_time:.2f} 秒\n📁 文件: {os.path.basename(final_output)}\n📏 大小: {os.path.getsize(final_output) / 1024 / 1024:.2f} MB"
        final_system_info = get_system_status()
        status_updates = update_status("✅ 生成完成", success_info, final_system_info, show_progress=True, show_system=True)
        
        # Clean up temporary prompt file if it was created
        if prompt_path and tempfile.gettempdir() in prompt_path:
            try:
                os.remove(prompt_path)
                print(f"已清理临时参考音频文件: {prompt_path}")
            except:
                pass  # Ignore cleanup errors
        
        return gr.update(value=final_output, visible=True), *status_updates
    
    except Exception as e:
        error_msg = f"❌ 生成音频时发生错误: {str(e)}"
        print(f"ERROR: {error_msg}")
        import traceback
        error_traceback = traceback.format_exc()
        print(error_traceback)
        
        # 错误状态更新
        detailed_error = f"错误类型: {type(e).__name__}\n错误信息: {str(e)}\n\n详细堆栈:\n{error_traceback}"
        status_updates = update_status("❌ 生成失败", "", "", detailed_error, show_error=True)
        
        # Clean up temporary prompt file if it was created
        try:
            if 'prompt_path' in locals() and prompt_path and tempfile.gettempdir() in prompt_path:
                os.remove(prompt_path)
                print(f"已清理临时参考音频文件: {prompt_path}")
        except:
            pass  # Ignore cleanup errors
            
        return gr.update(value=None, visible=True), *status_updates

def get_system_status():
    """获取系统状态信息"""
    try:
        import psutil
        import os
        
        # GPU信息
        gpu_info = ""
        if hasattr(tts, 'device') and 'cuda' in str(tts.device):
            try:
                import torch
                # 强制同步确保获取最新显存信息
                torch.cuda.synchronize(tts.device)
                
                gpu_allocated = torch.cuda.memory_allocated(tts.device) / 1024**3
                gpu_reserved = torch.cuda.memory_reserved(tts.device) / 1024**3
                gpu_total = torch.cuda.get_device_properties(tts.device).total_memory / 1024**3
                gpu_usage = (gpu_allocated / gpu_total) * 100
                gpu_name = torch.cuda.get_device_name(tts.device)
                
                gpu_info = f"""🎮 GPU信息:
{gpu_name}
已分配: {gpu_allocated:.2f}GB
已缓存: {gpu_reserved:.2f}GB
总容量: {gpu_total:.2f}GB
使用率: {gpu_usage:.1f}%"""
            except Exception as e:
                gpu_info = f"🎮 GPU信息获取失败: {str(e)}"
        else:
            gpu_info = "🎮 当前使用CPU模式"
        
        # 系统内存信息
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=0.05)  # 更短的interval
        process_memory = psutil.Process(os.getpid()).memory_info().rss / 1024**3
        
        system_info = f"""💾 系统内存:
使用: {memory.used/1024**3:.2f}GB
总计: {memory.total/1024**3:.2f}GB
使用率: {memory.percent:.1f}%

🖥️ CPU使用率: {cpu_percent:.1f}%

📊 进程内存: {process_memory:.2f}GB"""
        
        return f"{gpu_info}\n\n{system_info}"
        
    except Exception as e:
        return f"❌ 系统信息获取失败: {str(e)}"

def clear_gpu_cache():
    """Clear GPU cache and return memory info"""
    try:
        if hasattr(tts, 'comprehensive_memory_cleanup'):
            # 使用新的全面内存清理方法
            if hasattr(tts, 'device') and 'cuda' in str(tts.device):
                import torch
                memory_before = torch.cuda.memory_allocated(tts.device) / 1024**3  # GB
                tts.comprehensive_memory_cleanup(verbose=True)
                memory_after = torch.cuda.memory_allocated(tts.device) / 1024**3  # GB
                
                message = f"GPU内存全面清理完成\n清理前: {memory_before:.2f}GB\n清理后: {memory_after:.2f}GB\n释放: {memory_before - memory_after:.2f}GB"
            else:
                tts.comprehensive_memory_cleanup(verbose=True)
                message = "内存清理完成（非CUDA设备）"
        elif hasattr(tts, 'torch_empty_cache'):
            # 回退到普通清理方法
            memory_before = 0
            memory_after = 0
            
            if hasattr(tts, 'device') and 'cuda' in str(tts.device):
                import torch
                memory_before = torch.cuda.memory_allocated(tts.device) / 1024**3  # GB
                tts.torch_empty_cache(verbose=True)
                memory_after = torch.cuda.memory_allocated(tts.device) / 1024**3  # GB
                
                message = f"GPU缓存已清理\n清理前: {memory_before:.2f}GB\n清理后: {memory_after:.2f}GB\n释放: {memory_before - memory_after:.2f}GB"
            else:
                tts.torch_empty_cache()
                message = "缓存已清理（非CUDA设备）"
        else:
            message = "TTS模型未加载，无法清理缓存"
            
        print(f">> {message}")
        return message
        
    except Exception as e:
        error_message = f"清理缓存时发生错误: {str(e)}"
        print(f">> {error_message}")
        return error_message

def update_prompt_audio():
    update_button = gr.update(interactive=True)
    return update_button

def process_uploaded_file(file):
    """Process uploaded text/epub file"""
    if file is None:
        return "", "", "", ""
    
    file_path = file.name
    filename = os.path.basename(file_path)
    file_ext = os.path.splitext(filename)[1].lower()
    
    chapters = []  # Initialize chapters list
    
    if file_ext == '.txt':
        content = read_txt_file(file_path)
    elif file_ext == '.epub':
        content, chapters = read_epub_file(file_path)
    else:
        return "不支持的文件格式，仅支持 .txt 和 .epub 文件", "", "", ""
    
    # Limit preview length to prevent browser crashes with large files
    max_preview_chars = 10000  # 约10,000字符预览
    if len(content) > max_preview_chars:
        preview_content = content[:max_preview_chars] + f"\n\n... (文件过长，仅显示前{max_preview_chars}字符作为预览。完整内容将用于音频生成。)"
        if chapters:
            preview_content += f"\n\n检测到 {len(chapters)} 个章节，生成M4B格式时将添加章节书签。"
        return preview_content, filename, content, chapters  # Return preview, filename, full content, and chapters
    
    if chapters:
        content += f"\n\n检测到 {len(chapters)} 个章节，生成M4B格式时将添加章节书签。"
    
    return content, filename, content, chapters  # Return same content for both preview and full, plus chapters

def refresh_sample_files():
    """Refresh the list of sample files"""
    files = get_sample_files()
    if not files:
        return gr.update(choices=["无可用文件"], value="无可用文件"), gr.update(value=None)
    
    # Update dropdown and preview audio
    default_audio_path = os.path.join("samples", files[0])
    if not os.path.exists(default_audio_path):
        default_audio_path = None
    
    return (
        gr.update(choices=files, value=files[0]),
        gr.update(value=default_audio_path)
    )

def on_sample_audio_change(selected_file):
    """Handle sample audio selection change"""
    if selected_file and selected_file != "无可用文件":
        file_path = os.path.join("samples", selected_file)
        if os.path.exists(file_path):
            # Since we only include standard audio formats, we can always preview
            return gr.update(value=file_path, label="参考音频预览", visible=True)
    return gr.update(value=None, label="参考音频预览", visible=True)

with gr.Blocks(title="IndexTTS Demo") as demo:
    mutex = threading.Lock()
    gr.HTML('''
    <h2><center>IndexTTS: An Industrial-Level Controllable and Efficient Zero-Shot Text-To-Speech System</h2>
    <h2><center>(一款工业级可控且高效的零样本文本转语音系统)</h2>
<p align="center">
<a href='https://arxiv.org/abs/2502.05512'><img src='https://img.shields.io/badge/ArXiv-2502.05512-red'></a>
</p>
    ''')
    with gr.Tab("音频生成"):
        with gr.Row():
            with gr.Column(scale=1):
                # 参考音频选择
                gr.Markdown("**参考音频选择**")
                with gr.Row():
                    sample_files = get_sample_files()
                    default_choices = sample_files if sample_files else ["无可用文件"]
                    default_value = sample_files[0] if sample_files else "无可用文件"
                    
                    sample_dropdown = gr.Dropdown(
                        label="选择样本音频",
                        choices=default_choices,
                        value=default_value,
                        interactive=True
                    )
                    refresh_btn = gr.Button("刷新", size="sm")
                
                # 设置默认预览音频
                default_audio_path = None
                if sample_files:
                    default_audio_path = os.path.join("samples", sample_files[0])
                    if not os.path.exists(default_audio_path):
                        default_audio_path = None
                
                prompt_audio = gr.Audio(
                    label="参考音频预览",
                    key="prompt_audio", 
                    interactive=False,
                    value=default_audio_path
                )
                
                # 文件上传区域
                gr.Markdown("**文本输入**")
                uploaded_file = gr.File(
                    label="上传文本文件 (支持 .txt 和 .epub)",
                    file_types=[".txt", ".epub"],
                    key="uploaded_file"
                )
                uploaded_filename = gr.State("")
                full_text_content = gr.State("")
                chapters_info = gr.State([])
                
            with gr.Column(scale=2):
                input_text_single = gr.TextArea(
                    label="文本内容",
                    key="input_text_single", 
                    placeholder="请输入目标文本或上传文件", 
                    info="当前模型版本{}".format(tts.model_version or "1.0"),
                    lines=10
                )
                
                with gr.Row():
                    with gr.Column(scale=1):
                        infer_mode = gr.Radio(
                            choices=["普通推理", "批次推理"], 
                            label="推理模式",
                            info="批次推理：更适合长句，性能翻倍",
                            value="普通推理"
                        )
                    
                    # 保存设置
                    with gr.Column(scale=1):
                        auto_save = gr.Checkbox(label="自动保存", value=True, info="生成完成后自动保存到outputs文件夹")
                        background_mode = gr.Checkbox(
                            label="智能后台处理", 
                            value=True, 
                            info="长文本自动后台处理，避免连接超时"
                        )
                        audio_format = gr.Radio(
                            choices=["WAV", "MP3", "M4B"], 
                            label="音频格式",
                            value="MP3",
                            info="MP3: 64kbps压缩格式\nM4B: 有声书格式"
                        )
                
                gen_button = gr.Button("生成语音", key="gen_button", interactive=True, variant="primary", size="lg")
                
                # 内存管理区域
                with gr.Row():
                    clear_cache_btn = gr.Button("清理GPU缓存", variant="secondary", size="sm")
                    cache_info = gr.Textbox(label="缓存信息", interactive=False, max_lines=4, visible=False)
                
                # 生成状态显示区域
                with gr.Accordion("生成状态和系统监控", open=True):
                    with gr.Row():
                        with gr.Column(scale=2):
                            status_info = gr.Textbox(
                                label="当前状态", 
                                value="等待开始生成...",
                                interactive=False,
                                max_lines=3
                            )
                            progress_info = gr.Textbox(
                                label="详细进度",
                                placeholder="进度信息将在生成过程中显示...",
                                interactive=False,
                                max_lines=8,
                                visible=False
                            )
                        with gr.Column(scale=1):
                            system_info = gr.Textbox(
                                label="系统信息",
                                placeholder="系统监控信息将在生成时显示...",
                                interactive=False,
                                max_lines=8,
                                visible=False
                            )
                    
                    error_info = gr.Textbox(
                        label="错误信息",
                        interactive=False,
                        max_lines=4,
                        visible=False
                    )
                    
        output_audio = gr.Audio(label="生成结果", visible=True, key="output_audio")
        with gr.Accordion("高级生成参数设置", open=False):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("**GPT2 采样设置** _参数会影响音频多样性和生成速度详见[Generation strategies](https://huggingface.co/docs/transformers/main/en/generation_strategies)_")
                    with gr.Row():
                        do_sample = gr.Checkbox(label="do_sample", value=True, info="是否进行采样")
                        temperature = gr.Slider(label="temperature", minimum=0.1, maximum=2.0, value=1.0, step=0.1)
                    with gr.Row():
                        top_p = gr.Slider(label="top_p", minimum=0.0, maximum=1.0, value=0.8, step=0.01)
                        top_k = gr.Slider(label="top_k", minimum=0, maximum=100, value=30, step=1)
                        num_beams = gr.Slider(label="num_beams", value=3, minimum=1, maximum=10, step=1)
                    with gr.Row():
                        repetition_penalty = gr.Number(label="repetition_penalty", precision=None, value=10.0, minimum=0.1, maximum=20.0, step=0.1)
                        length_penalty = gr.Number(label="length_penalty", precision=None, value=0.0, minimum=-2.0, maximum=2.0, step=0.1)
                    max_mel_tokens = gr.Slider(label="max_mel_tokens", value=600, minimum=50, maximum=tts.cfg.gpt.max_mel_tokens, step=10, info="生成Token最大数量，过小导致音频被截断", key="max_mel_tokens")
                    # with gr.Row():
                    #     typical_sampling = gr.Checkbox(label="typical_sampling", value=False, info="不建议使用")
                    #     typical_mass = gr.Slider(label="typical_mass", value=0.9, minimum=0.0, maximum=1.0, step=0.1)
                with gr.Column(scale=2):
                    gr.Markdown("**分句设置** _参数会影响音频质量和生成速度_")
                    with gr.Row():
                        max_text_tokens_per_sentence = gr.Slider(
                            label="分句最大Token数", value=120, minimum=20, maximum=tts.cfg.gpt.max_text_tokens, step=2, key="max_text_tokens_per_sentence",
                            info="建议80~200之间，值越大，分句越长；值越小，分句越碎；过小过大都可能导致音频质量不高",
                        )
                        sentences_bucket_max_size = gr.Slider(
                            label="分句分桶的最大容量（批次推理生效）", value=8, minimum=1, maximum=16, step=1, key="sentences_bucket_max_size",
                            info="建议4-10之间，值越大，一批次推理包含的分句数越多，过大可能导致内存溢出",
                        )
                    with gr.Accordion("预览分句结果", open=True) as sentences_settings:
                        sentences_preview = gr.Dataframe(
                            headers=["序号", "分句内容", "Token数"],
                            key="sentences_preview",
                            wrap=True,
                        )
            advanced_params = [
                do_sample, top_p, top_k, temperature,
                length_penalty, num_beams, repetition_penalty, max_mel_tokens,
                # typical_sampling, typical_mass,
            ]
        
        if len(example_cases) > 0:
            gr.Examples(
                examples=example_cases,
                inputs=[prompt_audio, input_text_single, infer_mode],
            )
    
    # 任务管理页面
    with gr.Tab("任务管理"):
        gr.Markdown("## 后台任务管理")
        gr.Markdown("此页面显示所有后台音频生成任务的状态。长文本会自动提交到后台处理，您可以关闭浏览器，任务将继续运行。")
        
        with gr.Row():
            refresh_tasks_btn = gr.Button("刷新任务状态", variant="primary", size="sm")
            clear_tasks_btn = gr.Button("清理已完成任务", variant="secondary", size="sm")
        
        # 任务列表显示
        tasks_display = gr.Dataframe(
            headers=["任务ID", "状态", "进度信息", "结果文件", "错误信息"],
            label="任务列表",
            interactive=False,
            wrap=True
        )
        
        # 任务详情显示
        with gr.Accordion("任务详情", open=False):
            task_id_input = gr.Textbox(label="任务ID", placeholder="输入任务ID查看详情")
            task_detail_btn = gr.Button("查看详情", size="sm")
            
            task_detail_status = gr.Textbox(label="任务状态", interactive=False)
            task_detail_progress = gr.Textbox(label="进度详情", interactive=False, lines=5)
            task_detail_result = gr.Textbox(label="结果文件", interactive=False)
            task_detail_error = gr.Textbox(label="错误信息", interactive=False, lines=5, visible=False)
            
            # 下载结果文件按钮
            download_result_btn = gr.Button("下载结果文件", variant="primary", size="sm", visible=False)
            result_file_download = gr.File(label="下载文件", visible=False)
        
        # 任务管理函数
        def refresh_tasks():
            """刷新任务列表"""
            tasks = get_all_tasks()
            data = []
            for task_id, task_info in tasks.items():
                status = task_info.get("status", "未知")
                progress = task_info.get("progress", "")[:100] + "..." if len(task_info.get("progress", "")) > 100 else task_info.get("progress", "")
                result = os.path.basename(task_info.get("result", "")) if task_info.get("result") else ""
                error = task_info.get("error", "")[:50] + "..." if len(task_info.get("error", "")) > 50 else task_info.get("error", "")
                data.append([task_id, status, progress, result, error])
            
            return gr.update(value=data)
        
        def clear_tasks():
            """清理已完成的任务"""
            cleared_count = clear_completed_tasks()
            # 刷新任务列表
            tasks = get_all_tasks()
            data = []
            for task_id, task_info in tasks.items():
                status = task_info.get("status", "未知")
                progress = task_info.get("progress", "")[:100] + "..." if len(task_info.get("progress", "")) > 100 else task_info.get("progress", "")
                result = os.path.basename(task_info.get("result", "")) if task_info.get("result") else ""
                error = task_info.get("error", "")[:50] + "..." if len(task_info.get("error", "")) > 50 else task_info.get("error", "")
                data.append([task_id, status, progress, result, error])
            
            return gr.update(value=data)
        
        def show_task_detail(task_id):
            """显示任务详情"""
            if not task_id:
                return "", "", "", "", gr.update(visible=False), gr.update(visible=False)
            
            task_info = get_task_status(task_id)
            status = task_info.get("status", "任务不存在")
            progress = task_info.get("progress", "")
            result = task_info.get("result", "")
            error = task_info.get("error", "")
            
            # 检查是否有结果文件可下载
            show_download = bool(result and os.path.exists(result))
            show_error = bool(error)
            
            return (
                status,
                progress,
                result,
                error,
                gr.update(visible=show_download),
                gr.update(visible=show_error)
            )
        
        def prepare_download(task_id):
            """准备下载结果文件"""
            task_info = get_task_status(task_id)
            result_file = task_info.get("result", "")
            
            if result_file and os.path.exists(result_file):
                return gr.update(value=result_file, visible=True)
            else:
                return gr.update(value=None, visible=False)
        
        # 绑定事件
        refresh_tasks_btn.click(refresh_tasks, outputs=[tasks_display])
        clear_tasks_btn.click(clear_tasks, outputs=[tasks_display])
        
        task_detail_btn.click(
            show_task_detail,
            inputs=[task_id_input],
            outputs=[task_detail_status, task_detail_progress, task_detail_result, task_detail_error, download_result_btn, task_detail_error]
        )
        
        download_result_btn.click(
            prepare_download,
            inputs=[task_id_input],
            outputs=[result_file_download]
        )
        
        # 自动刷新任务状态（可选）
        # 注意：这会增加服务器负载，建议手动刷新
        # demo.load(refresh_tasks, outputs=[tasks_display], every=5)

    def on_input_text_change(text, max_tokens_per_sentence):
        if text and len(text) > 0:
            text_tokens_list = tts.tokenizer.tokenize(text)

            sentences = tts.tokenizer.split_sentences(text_tokens_list, max_tokens_per_sentence=int(max_tokens_per_sentence))
            data = []
            for i, s in enumerate(sentences):
                sentence_str = ''.join(s)
                tokens_count = len(s)
                data.append([i, sentence_str, tokens_count])
            
            return {
                sentences_preview: gr.update(value=data, visible=True, type="array"),
            }
        else:
            df = pd.DataFrame([], columns=["序号", "分句内容", "Token数"])
            return {
                sentences_preview: gr.update(value=df)
            }

    # Event handlers
    input_text_single.change(
        on_input_text_change,
        inputs=[input_text_single, max_text_tokens_per_sentence],
        outputs=[sentences_preview]
    )
    max_text_tokens_per_sentence.change(
        on_input_text_change,
        inputs=[input_text_single, max_text_tokens_per_sentence],
        outputs=[sentences_preview]
    )
    
    # Handle sample audio selection
    sample_dropdown.change(
        on_sample_audio_change,
        inputs=[sample_dropdown],
        outputs=[prompt_audio]
    )
    
    # Handle refresh button
    refresh_btn.click(
        refresh_sample_files,
        outputs=[sample_dropdown, prompt_audio]
    )
    
    # Handle file upload
    uploaded_file.upload(
        process_uploaded_file,
        inputs=[uploaded_file],
        outputs=[input_text_single, uploaded_filename, full_text_content, chapters_info]
    )

    # Handle generation with new parameters
    gen_button.click(
        gen_single,
        inputs=[
            prompt_audio, input_text_single, infer_mode,
            max_text_tokens_per_sentence, sentences_bucket_max_size,
            auto_save, audio_format, uploaded_filename, sample_dropdown, full_text_content, chapters_info, background_mode,
            *advanced_params,
        ],
        outputs=[output_audio, status_info, progress_info, system_info, error_info]
    )
    
    # Handle cache clearing
    def handle_cache_clear():
        message = clear_gpu_cache()
        return gr.update(value=message, visible=True)
    
    clear_cache_btn.click(
        handle_cache_clear,
        outputs=[cache_info]
    )


if __name__ == "__main__":
    try:
        demo.queue(20)
        demo.launch(
            server_name=cmd_args.host, 
            server_port=cmd_args.port, 
            inbrowser=True,
            share=False,  # 不使用公共链接
            debug=False,  # 关闭调试模式
            quiet=True,   # 减少日志输出
            show_error=True,  # 显示错误信息
            allowed_paths=["outputs", "samples", "tests"]  # 允许访问的路径
        )
    except KeyboardInterrupt:
        print("\n程序被用户中断，正在安全退出...")
    except Exception as e:
        print(f"启动WebUI时发生错误: {e}")
        import traceback
        traceback.print_exc()
