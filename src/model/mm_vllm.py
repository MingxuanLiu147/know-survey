"""Optional vLLM backend for the native MLLMU prompts and scoring protocol."""

import json
from pathlib import Path

from model.multimodal import get_mm_processor


class MMVLLM:
    """Batch ordered image/text requests with the saved generation settings."""

    def __init__(self, model_cfg, eval_cfg, seed=42):
        from vllm import LLM, SamplingParams

        self.processor = get_mm_processor(model_cfg)
        self.sampling_class = SamplingParams
        path = Path(str(model_cfg.model_path))
        settings = json.loads((path / "generation_config.json").read_text())
        self.eos = settings["eos_token_id"]
        if isinstance(self.eos, int):
            self.eos = [self.eos]
        self.repetition_penalty = float(settings.get("repetition_penalty", 1.0))
        self.seed = int(seed)
        self.batch_size = int(eval_cfg.get("inference_batch_size", 128))
        if self.batch_size < 1:
            raise ValueError("inference_batch_size must be positive")
        image_processor = self.processor.image_processor
        processor_kwargs = {
            "min_pixels": image_processor.min_pixels,
            "max_pixels": image_processor.max_pixels,
            "use_fast": type(image_processor).__name__.endswith("Fast"),
        }
        self.processor_kwargs = processor_kwargs
        cache_bytes = eval_cfg.get("kv_cache_memory_bytes")
        cache_kwargs = (
            {"kv_cache_memory_bytes": int(cache_bytes)}
            if cache_bytes is not None
            else {}
        )
        self.engine = LLM(
            model=str(path),
            tokenizer=str(path),
            dtype="bfloat16",
            tensor_parallel_size=1,
            seed=self.seed,
            gpu_memory_utilization=float(eval_cfg.get("gpu_memory_utilization", 0.8)),
            max_model_len=int(eval_cfg.get("max_model_len", 1024)),
            max_num_seqs=self.batch_size,
            limit_mm_per_prompt={"image": 1, "video": 0},
            mm_processor_kwargs=processor_kwargs,
            enforce_eager=bool(eval_cfg.get("enforce_eager", True)),
            enable_prefix_caching=True,
            disable_log_stats=True,
            **cache_kwargs,
        )
        self.last_token_ids = []
        self.last_prompt_token_ids = []

    def generate_content_batch(self, processor, requests, max_new_tokens):
        """Return results in input order; decoding matches native greedy settings."""
        params = self.sampling_class(
            temperature=0.0,
            max_tokens=int(max_new_tokens),
            seed=self.seed,
            repetition_penalty=self.repetition_penalty,
            stop_token_ids=self.eos,
            skip_special_tokens=True,
        )
        texts = []
        self.last_token_ids = []
        self.last_prompt_token_ids = []
        for start in range(0, len(requests), self.batch_size):
            prompts = []
            for content, images in requests[start : start + self.batch_size]:
                prompt = processor.apply_chat_template(
                    [{"role": "user", "content": content}],
                    add_generation_prompt=True,
                )
                # vLLM's Qwen processing info defaults use_fast to True before
                # merging engine defaults. Per-request kwargs preserve the
                # saved image processor on the actual inference path as well.
                item = {
                    "prompt": prompt,
                    "mm_processor_kwargs": self.processor_kwargs,
                }
                if images:
                    if len(images) != 1:
                        raise ValueError(
                            "This vLLM protocol supports zero-shot, one-image requests only"
                        )
                    item["multi_modal_data"] = {"image": images}
                prompts.append(item)
            outputs = self.engine.generate(prompts, params, use_tqdm=False)
            if len(outputs) != len(prompts):
                raise RuntimeError("vLLM returned an incomplete batch")
            for output in outputs:
                tokens = list(output.outputs[0].token_ids)
                texts.append(
                    processor.tokenizer.decode(tokens, skip_special_tokens=True).strip()
                )
                self.last_token_ids.append(tokens)
                self.last_prompt_token_ids.append(list(output.prompt_token_ids))
        return texts
