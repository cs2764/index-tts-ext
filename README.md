
<div align="center">
<img src='assets/index_icon.png' width="250"/>
</div>


<h2><center>IndexTTS: An Industrial-Level Controllable and Efficient Zero-Shot Text-To-Speech System</h2>

<p align="center">
<a href='https://arxiv.org/abs/2502.05512'><img src='https://img.shields.io/badge/ArXiv-2502.05512-red'></a>

## 👉🏻 IndexTTS 👈🏻

[[HuggingFace Demo]](https://huggingface.co/spaces/IndexTeam/IndexTTS)   [[ModelScope Demo]](https://modelscope.cn/studios/IndexTeam/IndexTTS-Demo) \
[[Paper]](https://arxiv.org/abs/2502.05512)  [[Demos]](https://index-tts.github.io)  

**IndexTTS** is a GPT-style text-to-speech (TTS) model mainly based on XTTS and Tortoise. It is capable of correcting the pronunciation of Chinese characters using pinyin and controlling pauses at any position through punctuation marks. We enhanced multiple modules of the system, including the improvement of speaker condition feature representation, and the integration of BigVGAN2 to optimize audio quality. Trained on tens of thousands of hours of data, our system achieves state-of-the-art performance, outperforming current popular TTS systems such as XTTS, CosyVoice2, Fish-Speech, and F5-TTS.

**Latest Enhancements (v0.2.2)**: This version introduces revolutionary intelligent sentence splitting system and fixes critical segmentation functionality. Key additions include semantic-aware sentence breaking with 6-level priority system (commas → semicolons → conjunctions → prepositions → relation words → hyphens), dynamic programming optimization for balanced segment lengths, and intelligent fallback mechanisms. Also fixed segmentation bugs including kwargs variable errors, TTS inference failure handling, and IndexError exceptions. The system now processes long sentences at natural linguistic boundaries, significantly improving TTS quality while maintaining full backward compatibility.

<span style="font-size:16px;">  
Experience **IndexTTS**: Please contact <u>xuanwu@bilibili.com</u> for more detailed information. </span>
### Contact
QQ群（二群）：1048202584 \
Discord：https://discord.gg/uT32E7KDmy  \
简历：indexspeech@bilibili.com  \
欢迎大家来交流讨论！
## 📣 Updates

- `2025/07/02` 🧠 **v0.2.2 - Intelligent Sentence Splitting & Segmentation Fixes**: Revolutionary semantic-aware sentence splitting with 6-level priority system for natural linguistic boundaries. Fixed critical segmentation bugs including kwargs errors, TTS inference failures, and IndexError exceptions. Significantly improves TTS quality for long sentences while maintaining full backward compatibility.
- `2025/01/16` 📚 **v0.2.1 - Batch Processing & Smart Chapter Parsing**: Added comprehensive batch file processing with intelligent Chinese chapter detection, supporting multiple formats (第X章, 卷X, 序言, (一)章节, etc.). Features include batch task management with ThreadPoolExecutor, encoding detection using chardet library, text cleaning functions, unified output folder structure, and advanced progress tracking for multiple files.
- `2025/01/16` 🎨 **v0.2.0 - Major WebUI Redesign**: Complete interface overhaul with modern design system, enhanced chapter management with controlled preview height (max 10 chapters, 300px), intelligent file organization with automatic folder creation, professional component grouping with emoji navigation, and streamlined user experience. Chapter-based audio splitting now generates files in organized folders with standardized naming (filename_1, filename_2, etc.).
- `2025/01/16` 🎉 **Latest UI/UX Enhancements**: Removed GPU memory display for cleaner interface, added intelligent ETA prediction with multi-algorithm fusion, integrated audio bitrate control (32-320kbps), and introduced chapter-based audio splitting for EPUB files.
- `2025/01/15` ✨ **Enhanced Features by Claude 4-Sonnet**: Added comprehensive memory optimization, real-time system monitoring, intelligent backend processing, and enhanced web interface for improved user experience.
- `2025/05/14` 🔥🔥 We release the **IndexTTS-1.5**, Significantly improve the model's stability and its performance in the English language.
- `2025/03/25` 🔥 We release IndexTTS-1.0 model parameters and inference code.
- `2025/02/12` 🔥 We submitted our paper on arXiv, and released our demos and test sets.

## 🖥️ Method

The overview of IndexTTS is shown as follows.

<picture>
  <img src="assets/IndexTTS.png"  width="800"/>
</picture>


The main improvements and contributions are summarized as follows:
 - In Chinese scenarios, we have introduced a character-pinyin hybrid modeling approach. This allows for quick correction of mispronounced characters.
 - **IndexTTS** incorporate a conformer conditioning encoder and a BigVGAN2-based speechcode decoder. This improves training stability, voice timbre similarity, and sound quality.
 - We release all test sets here, including those for polysyllabic words, subjective and objective test sets.



## Model Download
| 🤗**HuggingFace**                                          | **ModelScope** |
|----------------------------------------------------------|----------------------------------------------------------|
| [IndexTTS](https://huggingface.co/IndexTeam/Index-TTS) | [IndexTTS](https://modelscope.cn/models/IndexTeam/Index-TTS) |
| [😁IndexTTS-1.5](https://huggingface.co/IndexTeam/IndexTTS-1.5) | [IndexTTS-1.5](https://modelscope.cn/models/IndexTeam/IndexTTS-1.5) |


## 📑 Evaluation

**Word Error Rate (WER) Results for IndexTTS and Baseline Models on the** [**seed-test**](https://github.com/BytedanceSpeech/seed-tts-eval)

| **WER**                | **test_zh** | **test_en** | **test_hard** |
|:----------------------:|:-----------:|:-----------:|:-------------:|
| **Human**              | 1.26        | 2.14        | -             |
| **SeedTTS**            | 1.002       | 1.945       | **6.243**     |
| **CosyVoice 2**        | 1.45        | 2.57        | 6.83          |
| **F5TTS**              | 1.56        | 1.83        | 8.67          |
| **FireRedTTS**         | 1.51        | 3.82        | 17.45         |
| **MaskGCT**            | 2.27        | 2.62        | 10.27         |
| **Spark-TTS**          | 1.2         | 1.98        | -             |
| **MegaTTS 3**          | 1.36        | 1.82        | -             |
| **IndexTTS**           | 0.937       | 1.936       | 6.831         |
| **IndexTTS-1.5**       | **0.821**   | **1.606**   | 6.565         |


**Word Error Rate (WER) Results for IndexTTS and Baseline Models on the other opensource test**


|    **Model**    | **aishell1_test** | **commonvoice_20_test_zh** | **commonvoice_20_test_en** | **librispeech_test_clean** |  **avg** |
|:---------------:|:-----------------:|:--------------------------:|:--------------------------:|:--------------------------:|:--------:|
|    **Human**    |        2.0        |            9.5             |            10.0            |            2.4             |   5.1    |
| **CosyVoice 2** |        1.8        |            9.1             |            7.3             |            4.9             |   5.9    |
|    **F5TTS**    |        3.9        |            11.7            |            5.4             |            7.8             |   8.2    |
|  **Fishspeech** |        2.4        |            11.4            |            8.8             |            8.0             |   8.3    |
|  **FireRedTTS** |        2.2        |            11.0            |            16.3            |            5.7             |   7.7    |
|     **XTTS**    |        3.0        |            11.4            |            7.1             |            3.5             |   6.0    |
|   **IndexTTS**  |      1.3          |          7.0               |            5.3             |          2.1             | 3.7       |
|   **IndexTTS-1.5**  |      **1.2**     |          **6.8**          |          **3.9**          |          **1.7**          | **3.1** |


**Speaker Similarity (SS) Results for IndexTTS and Baseline Models**

|    **Model**    | **aishell1_test** | **commonvoice_20_test_zh** | **commonvoice_20_test_en** | **librispeech_test_clean** |  **avg**  |
|:---------------:|:-----------------:|:--------------------------:|:--------------------------:|:--------------------------:|:---------:|
|    **Human**    |       0.846       |            0.809           |            0.820           |            0.858           |   0.836   |
| **CosyVoice 2** |     **0.796**     |            0.743           |            0.742           |          **0.837**         | **0.788** |
|    **F5TTS**    |       0.743       |          **0.747**         |            0.746           |            0.828           |   0.779   |
|  **Fishspeech** |       0.488       |            0.552           |            0.622           |            0.701           |   0.612   |
|  **FireRedTTS** |       0.579       |            0.593           |            0.587           |            0.698           |   0.631   |
|     **XTTS**    |       0.573       |            0.586           |            0.648           |            0.761           |   0.663   |
|   **IndexTTS**  |       0.744       |            0.742           |          **0.758**         |            0.823           |   0.776   |
|   **IndexTTS-1.5**  |       0.741       |            0.722           |          0.753         |            0.819           |   0.771   |



**MOS Scores for Zero-Shot Cloned Voice**

| **Model**       | **Prosody** | **Timbre** | **Quality** |  **AVG**  |
|-----------------|:-----------:|:----------:|:-----------:|:---------:|
| **CosyVoice 2** |    3.67     |    4.05    |    3.73     |   3.81    |
| **F5TTS**       |    3.56     |    3.88    |    3.56     |   3.66    |
| **Fishspeech**  |    3.40     |    3.63    |    3.69     |   3.57    |
| **FireRedTTS**  |    3.79     |    3.72    |    3.60     |   3.70    |
| **XTTS**        |    3.23     |    2.99    |    3.10     |   3.11    |
| **IndexTTS**    |    **3.79**     |    **4.20**    |    **4.05**     |   **4.01**    |


## 🚀 Enhanced Features (Powered by Claude 4-Sonnet)

This repository includes significant enhancements generated by Claude 4-Sonnet to improve user experience and system performance:

### 📚 Batch Processing & Smart Chapter Parsing (v0.2.1)
- **Intelligent Chinese Chapter Detection**: Advanced SmartChapterParser supporting multiple Chinese formats (第X章, 卷X, 序言, (一)章节, etc.)
- **Batch File Processing**: Convert multiple text files to audio simultaneously with unified output management
- **Text Processing Pipeline**: Clean text functions with merge lines and remove spaces options
- **Encoding Detection**: Automatic file encoding detection using chardet library for robust file reading
- **Task Management System**: ThreadPoolExecutor-based batch processing with progress tracking and status monitoring
- **Unified Output Structure**: Organized batch results with timestamped folders and standardized naming conventions

### 🧠 Smart Memory Management
- **Streaming Processing Algorithm**: Replaces memory-accumulative processing to prevent GPU memory leaks
- **Comprehensive Memory Cleanup**: Automatic KV cache clearing and PyTorch garbage collection
- **Multi-stage Memory Optimization**: Cleanup at start, between batches, and at completion
- **GPU Memory Leak Prevention**: Solves the issue of continuously increasing VRAM usage

### 📊 Real-time System Monitoring  
- **Streamlined Interface**: Removed GPU memory display for cleaner, more focused user experience
- **Smart Time Prediction**: Intelligent ETA calculation using multiple algorithms (linear, EWMA, trend analysis)
- **Enhanced Progress Display**: Detailed progress with confidence indicators and processing speed metrics
- **System Resource Monitoring**: CPU usage, system memory, and process memory tracking with real-time updates

### 🎛️ Intelligent Backend Processing
- **Smart Task Distribution**: Automatic foreground/background processing based on text length
- **Background Task Management**: Process long texts without blocking the interface
- **Task Status Tracking**: Complete task lifecycle monitoring with real-time updates
- **Concurrent Processing**: ThreadPoolExecutor-based background task system

### 🖥️ Enhanced Web Interface
- **Clean UI Design**: Removed example cases for a more professional, streamlined interface
- **Flexible Audio Control**: Adjustable bitrate (32-320kbps) for MP3/M4B formats with quality optimization
- **Chapter-based Processing**: Split EPUB content into multiple audio files with configurable chapters per file
- **Advanced Progress Analytics**: Multi-algorithm time prediction with confidence levels and speed monitoring
- **Real-time Status Updates**: Live monitoring during audio generation with intelligent update intervals

### 🎨 Modern UI/UX Design (v0.2.0)
- **Professional Theme System**: Soft blue theme with gradient backgrounds and modern color palette
- **Intelligent Layout Organization**: Left-side control panels for audio/text input, right-side generation settings and monitoring
- **Component Grouping**: Visual separation with rounded borders, shadows, and organized sections
- **Emoji-Enhanced Navigation**: Intuitive icons for all components (🎵 audio, 📝 text, ⚙️ settings, 📊 monitoring)
- **Responsive Chapter Display**: Height-controlled preview (300px max) with scrollable content, limited to 10 chapters for optimal performance
- **Smart File Organization**: Automatic folder creation for chapter splits with standardized naming (filename_1, filename_2, etc.)
- **Enhanced User Experience**: Cleaner status displays, improved error messaging, and streamlined workflow

### 📈 Performance Optimizations
- **Intelligent ETA Algorithm**: Fusion of linear prediction, EWMA smoothing, trend analysis, and stage-based adjustments
- **Outlier Detection**: Automatic filtering of anomalous processing times for accurate predictions
- **Confidence Assessment**: High/Medium/Basic precision indicators based on data quality
- **Batch Processing Optimization**: Enhanced batch size and memory efficiency with smart time tracking
- **Multi-file Generation**: Efficient chapter-based audio splitting for long-form content

### 🎵 Audio Quality & Control
- **Dynamic Bitrate Selection**: User-controllable audio quality from 32kbps to 320kbps
- **Format-specific Optimization**: Intelligent bitrate application for MP3 and M4B formats
- **Chapter Segmentation**: Split long content into multiple files with customizable chapter grouping
- **Book-style Processing**: Enhanced support for EPUB files with automatic chapter detection

## 📖 使用指南

📝 **详细文档**: 查看 [IndexTTS使用和优化指南.md](./IndexTTS使用和优化指南.md) 获取完整的使用说明、性能优化建议和故障排除方案。

## Usage Instructions
### Environment Setup
1. Download this repository:
```bash
git clone https://github.com/cs2764/index-tts-ext.git
```
2. Install dependencies:

Create a new conda environment and install dependencies:
 
```bash
conda create -n index-tts python=3.10
conda activate index-tts
apt-get install ffmpeg
# or use conda to install ffmpeg
conda install -c conda-forge ffmpeg
```

Install [PyTorch](https://pytorch.org/get-started/locally/), e.g.:
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
```

> [!NOTE]
> If you are using Windows you may encounter [an error](https://github.com/index-tts/index-tts/issues/61) when installing `pynini`:
`ERROR: Failed building wheel for pynini`
> In this case, please install `pynini` via `conda`:
> ```bash
> # after conda activate index-tts
> conda install -c conda-forge pynini==2.1.6
> pip install WeTextProcessing --no-deps
> ```

Install `IndexTTS` as a package:
```bash
cd index-tts
pip install -e .
```

Install additional dependencies for enhanced monitoring features:
```bash
pip install psutil
```

3. Download models:

Download by `huggingface-cli`:

```bash
huggingface-cli download IndexTeam/IndexTTS-1.5 \
  config.yaml bigvgan_discriminator.pth bigvgan_generator.pth bpe.model dvae.pth gpt.pth unigram_12000.vocab \
  --local-dir checkpoints
```

Recommended for China users. 如果下载速度慢，可以使用镜像：
```bash
export HF_ENDPOINT="https://hf-mirror.com"
```

Or by `wget`:

```bash
wget https://huggingface.co/IndexTeam/IndexTTS-1.5/resolve/main/bigvgan_discriminator.pth -P checkpoints
wget https://huggingface.co/IndexTeam/IndexTTS-1.5/resolve/main/bigvgan_generator.pth -P checkpoints
wget https://huggingface.co/IndexTeam/IndexTTS-1.5/resolve/main/bpe.model -P checkpoints
wget https://huggingface.co/IndexTeam/IndexTTS-1.5/resolve/main/dvae.pth -P checkpoints
wget https://huggingface.co/IndexTeam/IndexTTS-1.5/resolve/main/gpt.pth -P checkpoints
wget https://huggingface.co/IndexTeam/IndexTTS-1.5/resolve/main/unigram_12000.vocab -P checkpoints
wget https://huggingface.co/IndexTeam/IndexTTS-1.5/resolve/main/config.yaml -P checkpoints
```

> [!NOTE]
> If you prefer to use the `IndexTTS-1.0` model, please replace `IndexTeam/IndexTTS-1.5` with `IndexTeam/IndexTTS` in the above commands.


4. Run test script:


```bash
# Please put your prompt audio in 'test_data' and rename it to 'input.wav'
python indextts/infer.py
```

5. Use as command line tool:

```bash
# Make sure pytorch has been installed before running this command
indextts "大家好，我现在正在bilibili 体验 ai 科技，说实话，来之前我绝对想不到！AI技术已经发展到这样匪夷所思的地步了！" \
  --voice reference_voice.wav \
  --model_dir checkpoints \
  --config checkpoints/config.yaml \
  --output output.wav
```

Use `--help` to see more options.
```bash
indextts --help
```

#### Windows 部署快速指南 / Windows Deployment Quick Guide

📄 **For Windows users**: We provide a simplified deployment script `start.txt` that contains all the necessary commands to deploy and run the webui on Windows systems. This file includes:
- Conda environment setup
- Windows-specific dependency installations (including pynini workaround)
- Model download commands
- WebUI launch instructions

You can follow the commands in `start.txt` step by step for a streamlined Windows deployment experience.

#### Web Demo
```bash
pip install -e ".[webui]" --no-build-isolation
python webui.py

# use another model version:
python webui.py --model_dir IndexTTS-1.5
```

Open your browser and visit `http://127.0.0.1:7860` to see the demo.

**Latest Features in Web Demo (v0.2.0):**
- **Modern Interface Design**: Complete UI overhaul with gradient backgrounds, organized component groups, and professional theme system
- **Smart Chapter Management**: EPUB chapter preview with height control (300px max), limited display (10 chapters), and scrollable content
- **Intelligent File Organization**: Automatic folder creation for chapter splits with standardized naming convention (filename_1, filename_2, etc.)
- **Enhanced Navigation**: Emoji-enhanced labels and intuitive component grouping for improved user experience
- **Responsive Layout**: Left-side control panels (audio/text input) and right-side generation settings with optimized spacing
- **Intelligent Time Prediction**: Multi-algorithm ETA calculation with confidence indicators (High/Medium/Basic precision)
- **Audio Quality Control**: Adjustable bitrate slider (32-320kbps) for MP3/M4B formats
- **Chapter-based Audio Splitting**: Split EPUB files into multiple audio files with configurable chapters per file
- **Streamlined Interface**: Clean, professional UI without example cases for better focus
- **Advanced Progress Analytics**: Real-time processing speed, batch time analysis, and smart time formatting
- **Enhanced System Monitoring**: CPU usage, system memory tracking with optimized update intervals
- **Background Task Management**: Process long texts without blocking the interface with comprehensive task monitoring

**How to Use New Features (v0.2.0):**

1. **Modern Interface Navigation**:
   - **Left Panel**: Use the organized control panels for reference audio selection and text input
   - **Right Panel**: Access generation settings, progress monitoring, and system information
   - **Component Groups**: Navigate using emoji-enhanced labels for intuitive operation
   - **Responsive Design**: Interface adapts to your workflow with optimized component spacing

2. **Smart Chapter Management** (for EPUB files):
   - Upload an EPUB file to see automatic chapter detection with preview
   - **Chapter Display**: View up to 10 chapters at once with scrollable 300px height limit
   - **Intelligent Preview**: Each chapter shows 80-character content preview for quick identification
   - **Usage Tips**: Access helpful hints about chapter splitting and M4B bookmark features

3. **Enhanced File Organization**:
   - **Automatic Folders**: Chapter splits create organized folders (e.g., "novel_20250116_azure")
   - **Standardized Naming**: Files follow pattern "filename_1.mp3", "filename_2.mp3", etc.
   - **Format Support**: Works with all audio formats (WAV/MP3/M4B) maintaining consistent structure

4. **Audio Quality Control**:
   - Select your desired audio format (WAV/MP3/M4B)
   - Adjust the bitrate slider (32-320kbps) for MP3/M4B formats
   - Higher bitrate = better quality but larger file size

5. **Chapter-based Audio Splitting**:
   - Upload an EPUB file to automatically detect chapters
   - Enable "📂 启用章节分段" (Chapter Splitting) in the settings
   - Set "📄 每个文件包含章节数" (Chapters per file) from 1-20
   - System will generate multiple audio files in organized folders

6. **Enhanced Progress Monitoring**:
   - Watch real-time ETA predictions with confidence levels
   - Monitor processing speed and batch time analysis
   - View system resource usage in the monitoring panel
   - For long texts, tasks automatically switch to background processing

#### Sample Code
```python
from indextts.infer import IndexTTS

# Initialize IndexTTS with enhanced monitoring capabilities
tts = IndexTTS(model_dir="checkpoints", cfg_path="checkpoints/config.yaml")

# Set up progress callback for real-time monitoring with intelligent ETA
def progress_callback(progress, desc=None):
    print(f"Progress: {progress:.1%} - {desc}")

tts.gr_progress = progress_callback

voice = "reference_voice.wav"
text = "大家好，我现在正在bilibili 体验 ai 科技，说实话，来之前我绝对想不到！AI技术已经发展到这样匪夷所思的地步了！比如说，现在正在说话的其实是B站为我现场复刻的数字分身，简直就是平行宇宙的另一个我了。如果大家也想体验更多深入的AIGC功能，可以访问 bilibili studio，相信我，你们也会吃惊的。"

# Enhanced inference with memory optimization and intelligent time prediction
tts.infer(voice, text, "output.wav", verbose=True)

# For long texts, use fast inference mode with optimized memory management
tts.infer_fast(voice, text, "output.wav", verbose=True, 
               max_text_tokens_per_sentence=100, 
               sentences_bucket_max_size=6)

# Example: Processing EPUB with enhanced chapter management (v0.2.0)
# The webui automatically detects chapters with intelligent preview (max 10 chapters shown)
# Files are organized in folders with standardized naming: filename_1, filename_2, etc.
# Each file can contain 1-20 chapters as configured by the user
# Generated files are saved in organized folders: "bookname_20250116_speaker/"
```

**Latest API Features (v0.2.0):**
- `verbose=True`: Enable detailed logging with intelligent time prediction and confidence indicators
- `infer_fast()`: Optimized inference for long texts with smart memory management and batch time tracking
- **Intelligent Progress Callbacks**: Real-time system information with multi-algorithm ETA calculation
- **Automatic Memory Cleanup**: Enhanced GPU cache management with comprehensive cleanup cycles
- **Smart Time Formatting**: Automatic conversion to human-readable time formats (seconds/minutes/hours)
- **Enhanced Chapter Processing**: Support for EPUB files with intelligent preview and organized file output
- **Smart File Organization**: Automatic folder creation with standardized naming convention (filename_1, filename_2, etc.)
- **Optimized Chapter Display**: Height-controlled preview (300px max) with configurable chapter limits (1-20 per file)
- **Professional UI Integration**: Modern theme system with gradient backgrounds and emoji-enhanced navigation

#### Testing Enhanced Features
Run the comprehensive test suite to verify all enhanced features:
```bash
# Test intelligent ETA algorithm and progress features
python test_enhanced_progress.py

# Test webui enhancements (bitrate control, chapter splitting)
python -c "
from webui import test_chapter_split_logic, test_bitrate_options
test_chapter_split_logic()
test_bitrate_options()
print('✅ WebUI enhancements tested successfully!')
"
```

This test suite validates:
- **Smart Time Prediction**: Multi-algorithm ETA calculation with confidence assessment
- **Chapter Splitting Logic**: Proper grouping of chapters into audio files
- **Bitrate Control**: Dynamic bitrate application for different audio formats
- **Progress Analytics**: Batch time tracking and processing speed monitoring
- **Memory Optimization**: Comprehensive cleanup and leak prevention
- **UI Enhancements**: Streamlined interface and professional user experience

## Acknowledge
1. [tortoise-tts](https://github.com/neonbjb/tortoise-tts)
2. [XTTSv2](https://github.com/coqui-ai/TTS)
3. [BigVGAN](https://github.com/NVIDIA/BigVGAN)
4. [wenet](https://github.com/wenet-e2e/wenet/tree/main)
5. [icefall](https://github.com/k2-fsa/icefall)

## 📚 Citation

🌟 If you find our work helpful, please leave us a star and cite our paper.

```
@article{deng2025indextts,
  title={IndexTTS: An Industrial-Level Controllable and Efficient Zero-Shot Text-To-Speech System},
  author={Wei Deng, Siyi Zhou, Jingchen Shu, Jinchao Wang, Lu Wang},
  journal={arXiv preprint arXiv:2502.05512},
  year={2025}
}
```
