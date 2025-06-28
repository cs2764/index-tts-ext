import os
import sys
import time
from subprocess import CalledProcessError
from typing import Dict, List, Tuple

import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence
from omegaconf import OmegaConf
from tqdm import tqdm

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from indextts.BigVGAN.models import BigVGAN as Generator
from indextts.gpt.model import UnifiedVoice
from indextts.utils.checkpoint import load_checkpoint
from indextts.utils.feature_extractors import MelSpectrogramFeatures

from indextts.utils.front import TextNormalizer, TextTokenizer


class IndexTTS:
    def __init__(
        self, cfg_path="checkpoints/config.yaml", model_dir="checkpoints", is_fp16=True, device=None, use_cuda_kernel=None,
    ):
        """
        Args:
            cfg_path (str): path to the config file.
            model_dir (str): path to the model directory.
            is_fp16 (bool): whether to use fp16.
            device (str): device to use (e.g., 'cuda:0', 'cpu'). If None, it will be set automatically based on the availability of CUDA or MPS.
            use_cuda_kernel (None | bool): whether to use BigVGan custom fused activation CUDA kernel, only for CUDA device.
        """
        if device is not None:
            self.device = device
            self.is_fp16 = False if device == "cpu" else is_fp16
            self.use_cuda_kernel = use_cuda_kernel is not None and use_cuda_kernel and device.startswith("cuda")
        elif torch.cuda.is_available():
            self.device = "cuda:0"
            self.is_fp16 = is_fp16
            self.use_cuda_kernel = use_cuda_kernel is None or use_cuda_kernel
        elif hasattr(torch, "mps") and torch.backends.mps.is_available():
            self.device = "mps"
            self.is_fp16 = False # Use float16 on MPS is overhead than float32
            self.use_cuda_kernel = False
        else:
            self.device = "cpu"
            self.is_fp16 = False
            self.use_cuda_kernel = False
            print(">> Be patient, it may take a while to run in CPU mode.")

        self.cfg = OmegaConf.load(cfg_path)
        self.model_dir = model_dir
        self.dtype = torch.float16 if self.is_fp16 else None
        self.stop_mel_token = self.cfg.gpt.stop_mel_token

        # Comment-off to load the VQ-VAE model for debugging tokenizer
        #   https://github.com/index-tts/index-tts/issues/34
        #
        # from indextts.vqvae.xtts_dvae import DiscreteVAE
        # self.dvae = DiscreteVAE(**self.cfg.vqvae)
        # self.dvae_path = os.path.join(self.model_dir, self.cfg.dvae_checkpoint)
        # load_checkpoint(self.dvae, self.dvae_path)
        # self.dvae = self.dvae.to(self.device)
        # if self.is_fp16:
        #     self.dvae.eval().half()
        # else:
        #     self.dvae.eval()
        # print(">> vqvae weights restored from:", self.dvae_path)
        self.gpt = UnifiedVoice(**self.cfg.gpt)
        self.gpt_path = os.path.join(self.model_dir, self.cfg.gpt_checkpoint)
        load_checkpoint(self.gpt, self.gpt_path)
        self.gpt = self.gpt.to(self.device)
        if self.is_fp16:
            self.gpt.eval().half()
        else:
            self.gpt.eval()
        print(">> GPT weights restored from:", self.gpt_path)
        if self.is_fp16:
            try:
                import deepspeed

                use_deepspeed = True
            except (ImportError, OSError, CalledProcessError) as e:
                use_deepspeed = False
                print(f">> DeepSpeed加载失败，回退到标准推理: {e}")
                print("See more details https://www.deepspeed.ai/tutorials/advanced-install/")

            self.gpt.post_init_gpt2_config(use_deepspeed=use_deepspeed, kv_cache=True, half=True)
        else:
            self.gpt.post_init_gpt2_config(use_deepspeed=False, kv_cache=True, half=False)

        if self.use_cuda_kernel:
            # preload the CUDA kernel for BigVGAN
            try:
                from indextts.BigVGAN.alias_free_activation.cuda import load as anti_alias_activation_loader
                anti_alias_activation_cuda = anti_alias_activation_loader.load()
                print(">> Preload custom CUDA kernel for BigVGAN", anti_alias_activation_cuda)
            except Exception as e:
                print(">> Failed to load custom CUDA kernel for BigVGAN. Falling back to torch.", e, file=sys.stderr)
                print(" Reinstall with `pip install -e . --no-deps --no-build-isolation` to prebuild `anti_alias_activation_cuda` kernel.", file=sys.stderr)
                print(
                    "See more details: https://github.com/index-tts/index-tts/issues/164#issuecomment-2903453206", file=sys.stderr
                )
                self.use_cuda_kernel = False
        self.bigvgan = Generator(self.cfg.bigvgan, use_cuda_kernel=self.use_cuda_kernel)
        self.bigvgan_path = os.path.join(self.model_dir, self.cfg.bigvgan_checkpoint)
        vocoder_dict = torch.load(self.bigvgan_path, map_location="cpu")
        self.bigvgan.load_state_dict(vocoder_dict["generator"])
        self.bigvgan = self.bigvgan.to(self.device)
        # remove weight norm on eval mode
        self.bigvgan.remove_weight_norm()
        self.bigvgan.eval()
        print(">> bigvgan weights restored from:", self.bigvgan_path)
        self.bpe_path = os.path.join(self.model_dir, self.cfg.dataset["bpe_model"])
        self.normalizer = TextNormalizer()
        self.normalizer.load()
        print(">> TextNormalizer loaded")
        self.tokenizer = TextTokenizer(self.bpe_path, self.normalizer)
        print(">> bpe model loaded from:", self.bpe_path)
        # 缓存参考音频mel：
        self.cache_audio_prompt = None
        self.cache_cond_mel = None
        # 进度引用显示（可选）
        self.gr_progress = None
        self.model_version = self.cfg.version if hasattr(self.cfg, "version") else None

    def remove_long_silence(self, codes: torch.Tensor, silent_token=52, max_consecutive=30):
        """
        Shrink special tokens (silent_token and stop_mel_token) in codes
        codes: [B, T]
        """
        code_lens = []
        codes_list = []
        device = codes.device
        dtype = codes.dtype
        isfix = False
        for i in range(0, codes.shape[0]):
            code = codes[i]
            if not torch.any(code == self.stop_mel_token).item():
                len_ = code.size(0)
            else:
                stop_mel_idx = (code == self.stop_mel_token).nonzero(as_tuple=False)
                len_ = stop_mel_idx[0].item() if len(stop_mel_idx) > 0 else code.size(0)

            count = torch.sum(code == silent_token).item()
            if count > max_consecutive:
                # code = code.cpu().tolist()
                ncode_idx = []
                n = 0
                for k in range(len_):
                    assert code[k] != self.stop_mel_token, f"stop_mel_token {self.stop_mel_token} should be shrinked here"
                    if code[k] != silent_token:
                        ncode_idx.append(k)
                        n = 0
                    elif code[k] == silent_token and n < 10:
                        ncode_idx.append(k)
                        n += 1
                    # if (k == 0 and code[k] == 52) or (code[k] == 52 and code[k-1] == 52):
                    #    n += 1
                # new code
                len_ = len(ncode_idx)
                codes_list.append(code[ncode_idx])
                isfix = True
            else:
                # shrink to len_
                codes_list.append(code[:len_])
            code_lens.append(len_)
        if isfix:
            if len(codes_list) > 1:
                codes = pad_sequence(codes_list, batch_first=True, padding_value=self.stop_mel_token)
            else:
                codes = codes_list[0].unsqueeze(0)
        else:
            # unchanged
            pass
        # clip codes to max length
        max_len = max(code_lens)
        if max_len < codes.shape[1]:
            codes = codes[:, :max_len]
        code_lens = torch.tensor(code_lens, dtype=torch.long, device=device)
        return codes, code_lens

    def bucket_sentences(self, sentences, bucket_max_size=4) -> List[List[Dict]]:
        """
        Sentence data bucketing.
        if ``bucket_max_size=1``, return all sentences in one bucket.
        """
        outputs: List[Dict] = []
        for idx, sent in enumerate(sentences):
            outputs.append({"idx": idx, "sent": sent, "len": len(sent)})
       
        if len(outputs) > bucket_max_size:
            # split sentences into buckets by sentence length
            buckets: List[List[Dict]] = []
            factor = 1.5
            last_bucket = None
            last_bucket_sent_len_median = 0

            for sent in sorted(outputs, key=lambda x: x["len"]):
                current_sent_len = sent["len"]
                if current_sent_len == 0:
                    print(">> skip empty sentence")
                    continue
                if last_bucket is None \
                        or current_sent_len >= int(last_bucket_sent_len_median * factor) \
                        or len(last_bucket) >= bucket_max_size:
                    # new bucket
                    buckets.append([sent])
                    last_bucket = buckets[-1]
                    last_bucket_sent_len_median = current_sent_len
                else:
                    # current bucket can hold more sentences
                    last_bucket.append(sent) # sorted
                    mid = len(last_bucket) // 2
                    last_bucket_sent_len_median = last_bucket[mid]["len"]
            last_bucket=None
            # merge all buckets with size 1
            out_buckets: List[List[Dict]] = []
            only_ones: List[Dict] = []
            for b in buckets:
                if len(b) == 1:
                    only_ones.append(b[0])
                else:
                    out_buckets.append(b)
            if len(only_ones) > 0:
                # merge into previous buckets if possible
                # print("only_ones:", [(o["idx"], o["len"]) for o in only_ones])
                for i in range(len(out_buckets)):
                    b = out_buckets[i]
                    if len(b) < bucket_max_size:
                        b.append(only_ones.pop(0))
                        if len(only_ones) == 0:
                            break
                # combined all remaining sized 1 buckets
                if len(only_ones) > 0:
                    out_buckets.extend([only_ones[i:i+bucket_max_size] for i in range(0, len(only_ones), bucket_max_size)])
            return out_buckets
        return [outputs]

    def pad_tokens_cat(self, tokens: List[torch.Tensor]) -> torch.Tensor:
        if self.model_version and self.model_version >= 1.5:
            # 1.5版本以上，直接使用stop_text_token 右侧填充，填充到最大长度
            # [1, N] -> [N,]
            tokens = [t.squeeze(0) for t in tokens]
            return pad_sequence(tokens, batch_first=True, padding_value=self.cfg.gpt.stop_text_token, padding_side="right")
        max_len = max(t.size(1) for t in tokens)
        outputs = []
        for tensor in tokens:
            pad_len = max_len - tensor.size(1)
            if pad_len > 0:
                n = min(8, pad_len)
                tensor = torch.nn.functional.pad(tensor, (0, n), value=self.cfg.gpt.stop_text_token)
                tensor = torch.nn.functional.pad(tensor, (0, pad_len - n), value=self.cfg.gpt.start_text_token)
            tensor = tensor[:, :max_len]
            outputs.append(tensor)
        tokens = torch.cat(outputs, dim=0)
        return tokens

    def torch_empty_cache(self, verbose=False):
        try:
            if "cuda" in str(self.device):
                if verbose:
                    # 打印内存使用情况（调试用）
                    memory_allocated = torch.cuda.memory_allocated(self.device) / 1024**3  # GB
                    memory_cached = torch.cuda.memory_reserved(self.device) / 1024**3  # GB
                    print(f">> GPU Memory before cleanup: Allocated={memory_allocated:.2f}GB, Cached={memory_cached:.2f}GB")
                torch.cuda.empty_cache()
                if verbose:
                    memory_allocated_after = torch.cuda.memory_allocated(self.device) / 1024**3  # GB
                    memory_cached_after = torch.cuda.memory_reserved(self.device) / 1024**3  # GB
                    print(f">> GPU Memory after cleanup: Allocated={memory_allocated_after:.2f}GB, Cached={memory_cached_after:.2f}GB")
            elif "mps" in str(self.device):
                torch.mps.empty_cache()
        except Exception as e:
            if verbose:
                print(f">> Warning: Failed to clear cache: {e}")

    def comprehensive_memory_cleanup(self, verbose=False):
        """
        执行全面的内存清理，包括模型缓存、KV缓存等
        """
        try:
            if verbose:
                memory_before = torch.cuda.memory_allocated(self.device) / 1024**3 if "cuda" in str(self.device) else 0
                print(f">> 开始全面内存清理，当前占用: {memory_before:.2f}GB")
            
            # 清理GPT模型的KV缓存
            if hasattr(self.gpt, 'inference_model') and hasattr(self.gpt.inference_model, 'cached_mel_emb'):
                self.gpt.inference_model.cached_mel_emb = None
                if verbose:
                    print(">> 已清理GPT模型的cached_mel_emb")
            
            # 如果有past_key_values缓存，也清理掉
            if hasattr(self.gpt, 'inference_model'):
                # 尝试清理transformer的缓存
                for module in self.gpt.inference_model.modules():
                    if hasattr(module, 'past_key_values'):
                        module.past_key_values = None
                    if hasattr(module, '_cache'):
                        module._cache = None
            
            # 清理BigVGAN模型的潜在缓存
            if hasattr(self.bigvgan, '_cache'):
                self.bigvgan._cache = None
                
            # 清理条件音频缓存（可选，根据需要）
            # 注意：这会导致下次使用相同音频时重新计算
            # self.cache_cond_mel = None
            # self.cache_audio_prompt = None
            
            # 执行PyTorch的内存清理
            self.torch_empty_cache(verbose=False)
            
            # 强制Python垃圾回收
            import gc
            gc.collect()
            
            if verbose:
                memory_after = torch.cuda.memory_allocated(self.device) / 1024**3 if "cuda" in str(self.device) else 0
                freed = memory_before - memory_after
                print(f">> 全面内存清理完成，释放: {freed:.2f}GB，当前占用: {memory_after:.2f}GB")
                
        except Exception as e:
            if verbose:
                print(f">> Warning: 全面内存清理时发生错误: {e}")

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
    
    def get_system_info(self, force_refresh=True):
        """获取系统信息包括显存、内存使用情况"""
        import psutil
        import os
        
        system_info = {}
        
        # GPU信息 - 强制刷新确保实时性
        if "cuda" in str(self.device):
            try:
                import torch
                # 强制同步确保获取最新的显存使用情况
                if force_refresh:
                    torch.cuda.synchronize(self.device)
                
                gpu_memory_allocated = torch.cuda.memory_allocated(self.device) / 1024**3  # GB
                gpu_memory_reserved = torch.cuda.memory_reserved(self.device) / 1024**3  # GB
                gpu_memory_total = torch.cuda.get_device_properties(self.device).total_memory / 1024**3  # GB
                
                system_info.update({
                    "gpu_memory_allocated": gpu_memory_allocated,
                    "gpu_memory_reserved": gpu_memory_reserved,
                    "gpu_memory_total": gpu_memory_total,
                    "gpu_memory_usage_percent": (gpu_memory_allocated / gpu_memory_total) * 100,
                    "gpu_name": torch.cuda.get_device_name(self.device)
                })
            except Exception as e:
                system_info["gpu_error"] = str(e)
        
        # CPU和内存信息
        try:
            # 使用较短的interval确保实时性
            cpu_percent = psutil.cpu_percent(interval=0.05)
            memory = psutil.virtual_memory()
            
            system_info.update({
                "cpu_percent": cpu_percent,
                "memory_used": memory.used / 1024**3,  # GB
                "memory_total": memory.total / 1024**3,  # GB
                "memory_percent": memory.percent,
                "process_memory": psutil.Process(os.getpid()).memory_info().rss / 1024**3  # GB
            })
        except Exception as e:
            system_info["system_error"] = str(e)
            
        return system_info

    def _set_gr_progress(self, value, desc, start_time=None, total_items=None, current_item=None, batch_times=None):
        """增强的进度更新，包含时间估算和系统信息"""
        # 获取实时系统信息
        system_info = self.get_system_info(force_refresh=True)
        
        # 时间计算
        time_info = ""
        if start_time is not None:
            elapsed_time = time.perf_counter() - start_time
            elapsed_formatted = self.format_time(elapsed_time)
            
            if total_items and current_item and current_item > 0:
                # 基于批次时间的智能预测
                if batch_times and len(batch_times) > 0:
                    # 使用最近几个批次的平均时间进行预测
                    recent_batches = batch_times[-min(3, len(batch_times)):]  # 最近3个批次
                    avg_batch_time = sum(recent_batches) / len(recent_batches)
                    remaining_batches = total_items - current_item
                    estimated_remaining = avg_batch_time * remaining_batches
                    
                    remaining_formatted = self.format_time(estimated_remaining)
                    time_info = f"\n⏱️ 已用时: {elapsed_formatted} | 预计剩余: {remaining_formatted}"
                    
                    # 添加批次速度信息
                    if len(batch_times) > 1:
                        last_batch_time = batch_times[-1]
                        last_batch_formatted = self.format_time(last_batch_time)
                        time_info += f"\n📊 当前批次: {last_batch_formatted} | 平均批次: {self.format_time(avg_batch_time)}"
                else:
                    # 回退到简单的线性预测
                    time_per_item = elapsed_time / current_item
                    remaining_items = total_items - current_item
                    estimated_remaining = time_per_item * remaining_items
                    
                    remaining_formatted = self.format_time(estimated_remaining)
                    time_info = f"\n⏱️ 已用时: {elapsed_formatted} | 预计剩余: {remaining_formatted}"
            else:
                time_info = f"\n⏱️ 已用时: {elapsed_formatted}"
        
        # 构建系统信息字符串
        sys_info = ""
        if "gpu_memory_allocated" in system_info:
            gpu_usage = system_info["gpu_memory_usage_percent"]
            gpu_used = system_info["gpu_memory_allocated"]
            gpu_total = system_info["gpu_memory_total"]
            sys_info += f"\n🎮 GPU: {gpu_used:.1f}/{gpu_total:.1f}GB ({gpu_usage:.1f}%)"
        
        if "memory_percent" in system_info:
            mem_percent = system_info["memory_percent"]
            mem_used = system_info["memory_used"]
            mem_total = system_info["memory_total"]
            process_mem = system_info["process_memory"]
            sys_info += f"\n💾 系统内存: {mem_used:.1f}/{mem_total:.1f}GB ({mem_percent:.1f}%) | 进程: {process_mem:.1f}GB"
        
        if "cpu_percent" in system_info:
            cpu_percent = system_info["cpu_percent"]
            sys_info += f"\n🖥️ CPU: {cpu_percent:.1f}%"
        
        # 完整的描述信息
        full_desc = desc + time_info + sys_info
        
        if self.gr_progress is not None:
            self.gr_progress(value, desc=full_desc)
        
        # 控制台输出（简化版本）
        console_msg = f">> 进度 {value*100:.1f}%: {desc}"
        if start_time:
            elapsed = time.perf_counter() - start_time
            console_msg += f" (已用时: {self.format_time(elapsed)})"
        if "gpu_memory_allocated" in system_info:
            console_msg += f" [GPU: {system_info['gpu_memory_allocated']:.1f}GB]"
        print(console_msg)

    # 快速推理：对于"多句长文本"，可实现至少 2~10 倍以上的速度提升~ （First modified by sunnyboxs 2025-04-16）
    def infer_fast(self, audio_prompt, text, output_path, verbose=False, max_text_tokens_per_sentence=100, sentences_bucket_max_size=4, **generation_kwargs):
        """
        Args:
            ``max_text_tokens_per_sentence``: 分句的最大token数，默认``100``，可以根据GPU硬件情况调整
                - 越小，batch 越多，推理速度越*快*，占用内存更多，可能影响质量
                - 越大，batch 越少，推理速度越*慢*，占用内存和质量更接近于非快速推理
            ``sentences_bucket_max_size``: 分句分桶的最大容量，默认``4``，可以根据GPU内存调整
                - 越大，bucket数量越少，batch越多，推理速度越*快*，占用内存更多，可能影响质量
                - 越小，bucket数量越多，batch越少，推理速度越*慢*，占用内存和质量更接近于非快速推理
        """
        print(">> start fast inference...")
        
        # 开始时清理缓存并监控内存
        self.comprehensive_memory_cleanup(verbose=verbose)
        
        start_time = time.perf_counter()
        self._set_gr_progress(0, "start fast inference...", start_time=start_time)
        if verbose:
            print(f"origin text:{text}")

        # 如果参考音频改变了，才需要重新生成 cond_mel, 提升速度
        if self.cache_cond_mel is None or self.cache_audio_prompt != audio_prompt:
            audio, sr = torchaudio.load(audio_prompt)
            audio = torch.mean(audio, dim=0, keepdim=True)
            if audio.shape[0] > 1:
                audio = audio[0].unsqueeze(0)
            audio = torchaudio.transforms.Resample(sr, 24000)(audio)
            cond_mel = MelSpectrogramFeatures()(audio).to(self.device)
            cond_mel_frame = cond_mel.shape[-1]
            if verbose:
                print(f"cond_mel shape: {cond_mel.shape}", "dtype:", cond_mel.dtype)

            self.cache_audio_prompt = audio_prompt
            self.cache_cond_mel = cond_mel
        else:
            cond_mel = self.cache_cond_mel
            cond_mel_frame = cond_mel.shape[-1]
            pass

        auto_conditioning = cond_mel
        cond_mel_lengths = torch.tensor([cond_mel_frame], device=self.device)

        # text_tokens
        text_tokens_list = self.tokenizer.tokenize(text)

        sentences = self.tokenizer.split_sentences(text_tokens_list, max_tokens_per_sentence=max_text_tokens_per_sentence)
        if verbose:
            print(">> text token count:", len(text_tokens_list))
            print("   splited sentences count:", len(sentences))
            print("   max_text_tokens_per_sentence:", max_text_tokens_per_sentence)
            print(*sentences, sep="\n")
        do_sample = generation_kwargs.pop("do_sample", True)
        top_p = generation_kwargs.pop("top_p", 0.8)
        top_k = generation_kwargs.pop("top_k", 30)
        temperature = generation_kwargs.pop("temperature", 1.0)
        autoregressive_batch_size = 1
        length_penalty = generation_kwargs.pop("length_penalty", 0.0)
        num_beams = generation_kwargs.pop("num_beams", 3)
        repetition_penalty = generation_kwargs.pop("repetition_penalty", 10.0)
        max_mel_tokens = generation_kwargs.pop("max_mel_tokens", 600)
        sampling_rate = 24000
        
        # 改进：使用流式处理避免大量内存累积
        wavs = []
        gpt_gen_time = 0
        gpt_forward_time = 0
        bigvgan_time = 0

        # text processing
        all_text_tokens: List[List[torch.Tensor]] = []
        self._set_gr_progress(0.1, "text processing...", start_time=start_time)
        bucket_max_size = sentences_bucket_max_size if self.device != "cpu" else 1
        all_sentences = self.bucket_sentences(sentences, bucket_max_size=bucket_max_size)
        bucket_count = len(all_sentences)
        if verbose:
            print(">> sentences bucket_count:", bucket_count,
                  "bucket sizes:", [(len(s), [t["idx"] for t in s]) for s in all_sentences],
                  "bucket_max_size:", bucket_max_size)
        
        # 详细的分句信息
        total_sentences = len(sentences)
        print(f">> 文本分析完成: {total_sentences} 个句子, {bucket_count} 个批次")
        self._set_gr_progress(0.15, f"文本分析完成: {total_sentences} 个句子, {bucket_count} 个批次", start_time=start_time)
        for sentences in all_sentences:
            temp_tokens: List[torch.Tensor] = []
            all_text_tokens.append(temp_tokens)
            for item in sentences:
                sent = item["sent"]
                text_tokens = self.tokenizer.convert_tokens_to_ids(sent)
                text_tokens = torch.tensor(text_tokens, dtype=torch.int32, device=self.device).unsqueeze(0)
                if verbose:
                    print(text_tokens)
                    print(f"text_tokens shape: {text_tokens.shape}, text_tokens type: {text_tokens.dtype}")
                    # debug tokenizer
                    text_token_syms = self.tokenizer.convert_ids_to_tokens(text_tokens[0].tolist())
                    print("text_token_syms is same as sentence tokens", text_token_syms == sent) 
                temp_tokens.append(text_tokens)
        
        # 改进的处理策略：流式处理，避免大量内存累积
        print(">> 开始流式批次处理...")
        self._set_gr_progress(0.2, "开始流式批次处理...", start_time=start_time)
        
        all_batch_num = sum(len(s) for s in all_sentences)
        processed_num = 0
        batch_idx = 0
        
        # 批次时间跟踪
        batch_times = []  # 记录每个批次的处理时间
        batch_start_time = time.perf_counter()
        
        # 用于按顺序收集最终的音频片段
        ordered_wavs = {}  # {original_idx: wav_tensor}
        
        # 逐批次处理，避免内存累积
        for batch_sentences, item_tokens in zip(all_sentences, all_text_tokens):
            # 记录当前批次开始时间
            current_batch_start = time.perf_counter()
            
            batch_num = len(item_tokens)
            if batch_num > 1:
                batch_text_tokens = self.pad_tokens_cat(item_tokens)
            else:
                batch_text_tokens = item_tokens[0]
            processed_num += batch_num
            
            # GPT推理阶段
            progress_percent = 0.2 + 0.6 * processed_num/all_batch_num
            self._set_gr_progress(progress_percent, f"处理批次 {batch_idx+1}/{bucket_count} - 大小: {batch_num}", 
                                start_time=start_time, total_items=bucket_count, current_item=batch_idx, batch_times=batch_times)
            
            m_start_time = time.perf_counter()
            with torch.no_grad():
                with torch.amp.autocast(batch_text_tokens.device.type, enabled=self.dtype is not None, dtype=self.dtype):
                    batch_codes = self.gpt.inference_speech(auto_conditioning, batch_text_tokens,
                                        cond_mel_lengths=cond_mel_lengths,
                                        do_sample=do_sample,
                                        top_p=top_p,
                                        top_k=top_k,
                                        temperature=temperature,
                                        num_return_sequences=autoregressive_batch_size,
                                        length_penalty=length_penalty,
                                        num_beams=num_beams,
                                        repetition_penalty=repetition_penalty,
                                        max_generate_length=max_mel_tokens,
                                        **generation_kwargs)
            gpt_gen_time += time.perf_counter() - m_start_time
            
            # 立即处理当前批次的latents和音频生成
            batch_wavs = []
            has_warned = False
            
            for i in range(batch_codes.shape[0]):
                codes = batch_codes[i]  # [x]
                if not has_warned and codes[-1] != self.stop_mel_token:
                    warnings.warn(
                        f"WARN: generation stopped due to exceeding `max_mel_tokens` ({max_mel_tokens}). "
                        f"Consider reducing `max_text_tokens_per_sentence`({max_text_tokens_per_sentence}) or increasing `max_mel_tokens`.",
                        category=RuntimeWarning
                    )
                    has_warned = True
                codes = codes.unsqueeze(0)  # [x] -> [1, x]
                if verbose:
                    print("codes:", codes.shape)
                    print(codes)
                codes, code_lens = self.remove_long_silence(codes, silent_token=52, max_consecutive=30)
                if verbose:
                    print("fix codes:", codes.shape)
                    print(codes)
                    print("code_lens:", code_lens)
                text_tokens = item_tokens[i]
                original_idx = batch_sentences[i]["idx"]
                
                # 立即生成latent和音频
                m_start_time = time.perf_counter()
                with torch.no_grad():
                    with torch.amp.autocast(text_tokens.device.type, enabled=self.dtype is not None, dtype=self.dtype):
                        latent = self.gpt(auto_conditioning, text_tokens,
                                        torch.tensor([text_tokens.shape[-1]], device=text_tokens.device), codes,
                                        code_lens*self.gpt.mel_length_compression,
                                        cond_mel_lengths=torch.tensor([auto_conditioning.shape[-1]], device=text_tokens.device),
                                        return_latent=True, clip_inputs=False)
                        gpt_forward_time += time.perf_counter() - m_start_time
                        
                        # 立即进行BigVGAN解码
                        m_start_time = time.perf_counter()
                        wav, _ = self.bigvgan(latent, auto_conditioning.transpose(1, 2))
                        bigvgan_time += time.perf_counter() - m_start_time
                        wav = wav.squeeze(1)
                        wav = torch.clamp(32767 * wav, -32767.0, 32767.0)
                        
                        # 存储到有序字典中，稍后按原始顺序合并
                        ordered_wavs[original_idx] = wav.cpu()
                        
                # 立即清理当前处理的张量
                del codes, text_tokens, latent, wav
                
            # 清理当前批次的数据
            del batch_text_tokens, batch_codes
            
            # 记录当前批次完成时间
            current_batch_time = time.perf_counter() - current_batch_start
            batch_times.append(current_batch_time)
            
            # 定期清理GPU缓存
            if (batch_idx + 1) % max(1, bucket_max_size // 2) == 0:
                self.comprehensive_memory_cleanup(verbose=verbose)
                if verbose:
                    memory_allocated = torch.cuda.memory_allocated(self.device) / 1024**3 if "cuda" in str(self.device) else 0
                    print(f">> 批次 {batch_idx+1} 处理完成，当前显存占用: {memory_allocated:.2f}GB，批次耗时: {self.format_time(current_batch_time)}")
            
            batch_idx += 1
        
        # 清理大型变量释放内存
        del all_text_tokens, all_sentences
        self.torch_empty_cache()  # 执行一次强制内存清理
        
        # 按原始顺序合并音频
        print(">> 合并音频片段...")
        self._set_gr_progress(0.9, "合并音频片段...", start_time=start_time)
        
        # 按索引顺序排序并合并
        sorted_indices = sorted(ordered_wavs.keys())
        for idx in sorted_indices:
            wavs.append(ordered_wavs[idx])
        
        # 清理有序字典
        del ordered_wavs
        
        end_time = time.perf_counter()
        self.torch_empty_cache()  # 最终清理所有GPU缓存

        # wav audio output
        self._set_gr_progress(0.95, "保存音频文件...", start_time=start_time)
        wav = torch.cat(wavs, dim=1)
        wav_length = wav.shape[-1] / sampling_rate
        ref_audio_length = cond_mel_frame * 256 / sampling_rate
        rtf = (end_time - start_time) / wav_length
        
        print(f">> 参考音频长度: {ref_audio_length:.2f} 秒")
        print(f">> GPT生成时间: {gpt_gen_time:.2f} 秒")
        print(f">> GPT前向时间: {gpt_forward_time:.2f} 秒") 
        print(f">> BigVGAN解码时间: {bigvgan_time:.2f} 秒")
        print(f">> 总推理时间: {end_time - start_time:.2f} 秒")
        print(f">> 生成音频长度: {wav_length:.2f} 秒")
        print(f">> 批次信息: {all_batch_num} 个句子, {bucket_count} 个批次, 分桶大小: {bucket_max_size}")
        print(f">> 实时率(RTF): {rtf:.4f} ({'快于实时' if rtf < 1.0 else '慢于实时'})")
        
        # 更新最终进度
        self._set_gr_progress(0.98, f"音频生成完成 - 时长: {wav_length:.1f}秒, RTF: {rtf:.3f}", start_time=start_time)
        
        # 结束时再次清理内存
        if verbose:
            print(">> Final memory cleanup...")
        self.torch_empty_cache(verbose=verbose)

        # save audio
        wav = wav.cpu()  # to cpu
        if output_path:
            # 直接保存音频到指定路径中
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            torchaudio.save(output_path, wav.type(torch.int16), sampling_rate)
            print(">> wav file saved to:", output_path)
            return output_path
        else:
            # 返回以符合Gradio的格式要求
            wav_data = wav.type(torch.int16)
            wav_data = wav_data.numpy().T
            return (sampling_rate, wav_data)

    # 原始推理模式
    def infer(self, audio_prompt, text, output_path, verbose=False, max_text_tokens_per_sentence=120, **generation_kwargs):
        print(">> start inference...")
        start_time = time.perf_counter()
        self._set_gr_progress(0, "start inference...", start_time=start_time)
        if verbose:
            print(f"origin text:{text}")

        # 如果参考音频改变了，才需要重新生成 cond_mel, 提升速度
        if self.cache_cond_mel is None or self.cache_audio_prompt != audio_prompt:
            audio, sr = torchaudio.load(audio_prompt)
            audio = torch.mean(audio, dim=0, keepdim=True)
            if audio.shape[0] > 1:
                audio = audio[0].unsqueeze(0)
            audio = torchaudio.transforms.Resample(sr, 24000)(audio)
            cond_mel = MelSpectrogramFeatures()(audio).to(self.device)
            cond_mel_frame = cond_mel.shape[-1]
            if verbose:
                print(f"cond_mel shape: {cond_mel.shape}", "dtype:", cond_mel.dtype)

            self.cache_audio_prompt = audio_prompt
            self.cache_cond_mel = cond_mel
        else:
            cond_mel = self.cache_cond_mel
            cond_mel_frame = cond_mel.shape[-1]
            pass

        self._set_gr_progress(0.1, "text processing...", start_time=start_time)
        auto_conditioning = cond_mel
        text_tokens_list = self.tokenizer.tokenize(text)
        sentences = self.tokenizer.split_sentences(text_tokens_list, max_text_tokens_per_sentence)
        if verbose:
            print("text token count:", len(text_tokens_list))
            print("sentences count:", len(sentences))
            print("max_text_tokens_per_sentence:", max_text_tokens_per_sentence)
            print(*sentences, sep="\n")
        do_sample = generation_kwargs.pop("do_sample", True)
        top_p = generation_kwargs.pop("top_p", 0.8)
        top_k = generation_kwargs.pop("top_k", 30)
        temperature = generation_kwargs.pop("temperature", 1.0)
        autoregressive_batch_size = 1
        length_penalty = generation_kwargs.pop("length_penalty", 0.0)
        num_beams = generation_kwargs.pop("num_beams", 3)
        repetition_penalty = generation_kwargs.pop("repetition_penalty", 10.0)
        max_mel_tokens = generation_kwargs.pop("max_mel_tokens", 600)
        sampling_rate = 24000
        # lang = "EN"
        # lang = "ZH"
        wavs = []
        gpt_gen_time = 0
        gpt_forward_time = 0
        bigvgan_time = 0
        progress = 0
        has_warned = False
        for sent in sentences:
            text_tokens = self.tokenizer.convert_tokens_to_ids(sent)
            text_tokens = torch.tensor(text_tokens, dtype=torch.int32, device=self.device).unsqueeze(0)
            # text_tokens = F.pad(text_tokens, (0, 1))  # This may not be necessary.
            # text_tokens = F.pad(text_tokens, (1, 0), value=0)
            # text_tokens = F.pad(text_tokens, (0, 1), value=1)
            if verbose:
                print(text_tokens)
                print(f"text_tokens shape: {text_tokens.shape}, text_tokens type: {text_tokens.dtype}")
                # debug tokenizer
                text_token_syms = self.tokenizer.convert_ids_to_tokens(text_tokens[0].tolist())
                print("text_token_syms is same as sentence tokens", text_token_syms == sent)

            # text_len = torch.IntTensor([text_tokens.size(1)], device=text_tokens.device)
            # print(text_len)
            progress += 1
            self._set_gr_progress(0.2 + 0.4 * (progress-1) / len(sentences), f"gpt inference latent... {progress}/{len(sentences)}", 
                                start_time=start_time, total_items=len(sentences), current_item=progress-1)
            m_start_time = time.perf_counter()
            with torch.no_grad():
                with torch.amp.autocast(text_tokens.device.type, enabled=self.dtype is not None, dtype=self.dtype):
                    codes = self.gpt.inference_speech(auto_conditioning, text_tokens,
                                                        cond_mel_lengths=torch.tensor([auto_conditioning.shape[-1]],
                                                                                      device=text_tokens.device),
                                                        # text_lengths=text_len,
                                                        do_sample=do_sample,
                                                        top_p=top_p,
                                                        top_k=top_k,
                                                        temperature=temperature,
                                                        num_return_sequences=autoregressive_batch_size,
                                                        length_penalty=length_penalty,
                                                        num_beams=num_beams,
                                                        repetition_penalty=repetition_penalty,
                                                        max_generate_length=max_mel_tokens,
                                                        **generation_kwargs)
                gpt_gen_time += time.perf_counter() - m_start_time
                if not has_warned and (codes[:, -1] != self.stop_mel_token).any():
                    warnings.warn(
                        f"WARN: generation stopped due to exceeding `max_mel_tokens` ({max_mel_tokens}). "
                        f"Input text tokens: {text_tokens.shape[1]}. "
                        f"Consider reducing `max_text_tokens_per_sentence`({max_text_tokens_per_sentence}) or increasing `max_mel_tokens`.",
                        category=RuntimeWarning
                    )
                    has_warned = True

                code_lens = torch.tensor([codes.shape[-1]], device=codes.device, dtype=codes.dtype)
                if verbose:
                    print(codes, type(codes))
                    print(f"codes shape: {codes.shape}, codes type: {codes.dtype}")
                    print(f"code len: {code_lens}")

                # remove ultra-long silence if exits
                # temporarily fix the long silence bug.
                codes, code_lens = self.remove_long_silence(codes, silent_token=52, max_consecutive=30)
                if verbose:
                    print(codes, type(codes))
                    print(f"fix codes shape: {codes.shape}, codes type: {codes.dtype}")
                    print(f"code len: {code_lens}")
                self._set_gr_progress(0.2 + 0.4 * progress / len(sentences), f"gpt inference speech... {progress}/{len(sentences)}", 
                                    start_time=start_time, total_items=len(sentences), current_item=progress)
                m_start_time = time.perf_counter()
                # latent, text_lens_out, code_lens_out = \
                with torch.amp.autocast(text_tokens.device.type, enabled=self.dtype is not None, dtype=self.dtype):
                    latent = \
                        self.gpt(auto_conditioning, text_tokens,
                                    torch.tensor([text_tokens.shape[-1]], device=text_tokens.device), codes,
                                    code_lens*self.gpt.mel_length_compression,
                                    cond_mel_lengths=torch.tensor([auto_conditioning.shape[-1]], device=text_tokens.device),
                                    return_latent=True, clip_inputs=False)
                    gpt_forward_time += time.perf_counter() - m_start_time

                    m_start_time = time.perf_counter()
                    wav, _ = self.bigvgan(latent, auto_conditioning.transpose(1, 2))
                    bigvgan_time += time.perf_counter() - m_start_time
                    wav = wav.squeeze(1)

                wav = torch.clamp(32767 * wav, -32767.0, 32767.0)
                if verbose:
                    print(f"wav shape: {wav.shape}", "min:", wav.min(), "max:", wav.max())
                # wavs.append(wav[:, :-512])
                wavs.append(wav.cpu())  # to cpu before saving
        end_time = time.perf_counter()
        self._set_gr_progress(0.9, "save audio...", start_time=start_time)
        wav = torch.cat(wavs, dim=1)
        wav_length = wav.shape[-1] / sampling_rate
        print(f">> Reference audio length: {cond_mel_frame * 256 / sampling_rate:.2f} seconds")
        print(f">> gpt_gen_time: {gpt_gen_time:.2f} seconds")
        print(f">> gpt_forward_time: {gpt_forward_time:.2f} seconds")
        print(f">> bigvgan_time: {bigvgan_time:.2f} seconds")
        print(f">> Total inference time: {end_time - start_time:.2f} seconds")
        print(f">> Generated audio length: {wav_length:.2f} seconds")
        print(f">> RTF: {(end_time - start_time) / wav_length:.4f}")

        # save audio
        wav = wav.cpu()  # to cpu
        if output_path:
            # 直接保存音频到指定路径中
            if os.path.isfile(output_path):
                os.remove(output_path)
                print(">> remove old wav file:", output_path)
            if os.path.dirname(output_path) != "":
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
            torchaudio.save(output_path, wav.type(torch.int16), sampling_rate)
            print(">> wav file saved to:", output_path)
            return output_path
        else:
            # 返回以符合Gradio的格式要求
            wav_data = wav.type(torch.int16)
            wav_data = wav_data.numpy().T
            return (sampling_rate, wav_data)


if __name__ == "__main__":
    prompt_wav="test_data/input.wav"
    #text="晕 XUAN4 是 一 种 GAN3 觉"
    #text='大家好，我现在正在bilibili 体验 ai 科技，说实话，来之前我绝对想不到！AI技术已经发展到这样匪夷所思的地步了！'
    text="There is a vehicle arriving in dock number 7?"

    tts = IndexTTS(cfg_path="checkpoints/config.yaml", model_dir="checkpoints", is_fp16=True, use_cuda_kernel=False)
    tts.infer(audio_prompt=prompt_wav, text=text, output_path="gen.wav", verbose=True)
