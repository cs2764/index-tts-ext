# IndexTTS WebUI 批量处理与智能章节解析功能

## 📚 新增功能概述

本次更新为 IndexTTS WebUI 增加了两个重要功能：

1. **批量文本转音频** - 支持批量上传多个文本文件，自动转换为音频
2. **智能中文章节解析** - 自动识别各种中文章节格式，支持章节分段和M4B书签

## 🚀 批量处理功能

### 功能特点

- **批量上传**：一次上传多个`.txt`文件
- **智能编码检测**：自动检测文件编码（UTF-8、GBK等）
- **文本清理**：自动合并空行、移除多余空格
- **章节识别**：智能识别中文章节结构
- **后台处理**：长时间任务在后台运行，不会阻塞界面
- **统一管理**：所有生成的音频文件保存在统一的批量文件夹中

### 使用方法

1. 进入「批量转换」页面
2. 上传多个`.txt`文件
3. 选择参考音频（从samples文件夹中选择）
4. 配置生成设置：
   - 推理模式：普通推理 / 批次推理
   - 音频格式：WAV / MP3 / M4B
   - 音频码率：32-320 kbps
   - 文本处理选项
5. 点击「开始批量转换」
6. 在页面中监控进度，或前往「任务管理」页面查看详情

### 输出结构

```
outputs/
├── batch_20241201_143022/          # 批量任务文件夹
│   ├── 小说1_音色名.mp3
│   ├── 小说2_音色名.mp3
│   └── 小说3_音色名.mp3
└── batch_20241201_150045/          # 另一个批量任务
    ├── 文档1_音色名.m4b
    └── 文档2_音色名.m4b
```

## 🧠 智能中文章节解析

### 支持的章节格式

智能章节解析器支持以下格式：

#### 高置信度格式
- `第一章 标题`
- `第二回 标题` 
- `第三节 标题`
- `卷一 标题`

#### 中高置信度格式
- `序言`
- `前言`
- `引子`
- `楔子`
- `后记`
- `番外`
- `尾声`

#### 括号格式
- `(一) 标题`
- `（二）标题`
- `(1) 标题`
- `（2）标题`

#### 序号格式
- `一、标题`
- `二、标题`
- `1. 标题`
- `2. 标题`

### 智能特性

- **多模式识别**：同时使用多种模式识别章节
- **智能过滤**：自动过滤文件名、URL、代码等非章节内容
- **置信度评分**：根据章节格式给出置信度分数
- **重复检测**：避免重复识别同一位置的章节
- **距离控制**：确保章节之间有合理的间距

### 应用场景

1. **单文件处理**：上传TXT或EPUB文件时自动识别章节
2. **批量处理**：批量转换时为每个文件识别章节
3. **章节分段**：根据章节自动分割音频文件
4. **M4B书签**：生成M4B格式时自动添加章节书签

## 🔧 技术实现

### 核心类和函数

#### SmartChapterParser 类
```python
class SmartChapterParser:
    def __init__(self, min_chapter_distance=50, merge_title_distance=25)
    def parse(self, text: str) -> List[Chapter]
```

#### 文本处理函数
```python
def clean_text(text, merge_lines=True, remove_spaces=True)
def detect_file_encoding(file_path)  
def smart_chinese_chapter_detection(text)
```

#### 批量处理函数
```python
def submit_batch_task(...)
def batch_audio_generation(...)
def update_batch_task_status(...)
```

### 依赖库

- `chardet` - 文件编码检测（可选）
- `re` - 正则表达式匹配
- `dataclasses` - 数据结构定义
- `concurrent.futures` - 线程池管理

## 💡 使用建议

### 批量处理最佳实践

1. **文件准备**：
   - 确保文本文件编码正确（推荐UTF-8）
   - 文件名使用有意义的名称
   - 文本内容格式规范

2. **性能优化**：
   - 大文件（>10MB）建议使用批次推理模式
   - 音频码率可根据需求调整（64kbps通常足够）
   - 避免同时处理过多文件

3. **章节识别**：
   - 章节标题应独占一行
   - 章节标题前后最好有空行
   - 避免使用特殊字符作为章节标题

### 故障排除

1. **编码问题**：
   - 安装chardet：`pip install chardet`
   - 手动转换文件编码为UTF-8

2. **章节识别不准确**：
   - 检查章节标题格式
   - 调整章节标题的空行分隔
   - 查看章节预览确认识别结果

3. **批量任务失败**：
   - 检查samples文件夹中是否有参考音频
   - 确认模型文件完整
   - 查看控制台错误信息

## 📋 更新日志

### v1.1.0 (2024-12-01)
- ✨ 新增批量文本转音频功能
- 🧠 集成智能中文章节解析器
- 🔧 增强文件编码检测
- 📊 添加批量任务管理界面
- 🎯 改进章节分段和M4B书签功能

## 🤝 贡献

欢迎提交问题和建议！如果您在使用过程中遇到问题，请：

1. 检查控制台输出的错误信息
2. 确认文件格式和编码
3. 提供详细的错误描述和重现步骤

## 📄 许可证

本项目遵循原有的许可证条款。 