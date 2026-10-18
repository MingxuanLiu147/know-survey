# Knowledge Injection (微调) 训练器模块
# 支持 LoRA, DoRA, AdaLoRA 等参数高效微调方法

from trainer.inject.base import InjectTrainer
from trainer.inject.lora import LoRATrainer
from trainer.inject.dora import DoRATrainer
from trainer.inject.adalora import AdaLoRATrainer
from trainer.inject.loreft import LoReFTTrainer

__all__ = [
    "InjectTrainer",
    "LoRATrainer",
    "DoRATrainer",
    "AdaLoRATrainer",
    "LoReFTTrainer",
]
