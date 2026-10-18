"""
Know-Surgery 训练器注册表
========================

统一管理所有训练器类，支持三大操作：
- Unlearn: 知识删除（GradAscent, SimNPO, RMU 等）
- Inject: 知识注入（LoRA, DoRA 等）
- Edit: 知识编辑（ROME, MEMIT, MEND 等）
"""

import torch
from typing import Dict, Any
from omegaconf import DictConfig
from transformers import Trainer, TrainingArguments

from trainer.base import FinetuneTrainer

# Unlearning 训练器
from trainer.unlearn.grad_ascent import GradAscent
from trainer.unlearn.grad_diff import GradDiff
from trainer.unlearn.npo import NPO
from trainer.unlearn.dpo import DPO
from trainer.unlearn.simnpo import SimNPO
from trainer.unlearn.rmu import RMU
from trainer.unlearn.undial import UNDIAL
from trainer.unlearn.ceu import CEU
from trainer.unlearn.satimp import SatImp
from trainer.unlearn.wga import WGA
from trainer.unlearn.pdu import PDU

# Knowledge Injection 训练器
from trainer.inject.base import InjectTrainer
from trainer.inject.lora import LoRATrainer
from trainer.inject.dora import DoRATrainer
from trainer.inject.adalora import AdaLoRATrainer
from trainer.inject.loreft import LoReFTTrainer

# Knowledge Editing 训练器
from trainer.edit.base import EditTrainer
from trainer.edit.rome import ROMEEditor
from trainer.edit.memit import MEMITEditor
from trainer.edit.mend import MENDEditor
from trainer.edit.ft import FTEditor
from trainer.edit.alphaedit import AlphaEditEditor
from trainer.edit.grace import GRACEEditor
from trainer.edit.wise import WISEEditor
from trainer.edit.wise_merge import WISEMergeEditor
from trainer.edit.ike import IKEEditor
from trainer.edit.serac import SERACEditor
from trainer.edit.serac_compact import SERACCompactEditor
from trainer.edit.malmen import MALMENEditor
from trainer.edit.instructedit import InstructEditEditor
from trainer.edit.anyedit import AnyEditEditor, unke_research_available

# Multimodal Knowledge Editing 训练器
from trainer.edit.mm_ike import MMIKEEditor
from trainer.edit.mm_lora import MMLoRAEditor
from trainer.edit.mm_grace import MMGRACEEditor
from trainer.edit.mm_wise import MMWISEEditor
from trainer.edit.mm_mend import MMMENDEditor
from trainer.edit.mm_serac import MMSERACEditor


import logging

logger = logging.getLogger(__name__)

TRAINER_REGISTRY: Dict[str, Any] = {}


def _register_trainer(trainer_class):
    TRAINER_REGISTRY[trainer_class.__name__] = trainer_class


def load_trainer_args(trainer_args: DictConfig, dataset):
    trainer_args = dict(trainer_args)
    warmup_epochs = trainer_args.pop("warmup_epochs", None)
    if warmup_epochs:
        if dataset is None:
            raise ValueError(
                "warmup_epochs is set but train_dataset is None. "
                "Please check if your data mode is set correctly (e.g. mode='unlearn') "
                "or if the dataset is loaded properly."
            )
        batch_size = trainer_args["per_device_train_batch_size"]
        grad_accum_steps = trainer_args["gradient_accumulation_steps"]
        num_devices = torch.cuda.device_count()
        dataset_len = len(dataset)
        trainer_args["warmup_steps"] = int(
            (warmup_epochs * dataset_len)
            // (batch_size * grad_accum_steps * num_devices)
        )

    trainer_args = TrainingArguments(**trainer_args)
    return trainer_args


def load_trainer(
    trainer_cfg: DictConfig,
    model,
    train_dataset=None,
    eval_dataset=None,
    tokenizer=None,
    data_collator=None,
    evaluators=None,
    template_args=None,
):
    trainer_args = trainer_cfg.args
    method_args = trainer_cfg.get("method_args", {})
    trainer_args = load_trainer_args(trainer_args, train_dataset)
    trainer_handler_name = trainer_cfg.get("handler")
    assert trainer_handler_name is not None, ValueError(
        f"{trainer_handler_name} handler not set"
    )
    trainer_cls = TRAINER_REGISTRY.get(trainer_handler_name, None)
    assert trainer_cls is not None, NotImplementedError(
        f"{trainer_handler_name} not implemented or not registered"
    )
    trainer = trainer_cls(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        args=trainer_args,
        evaluators=evaluators,
        template_args=template_args,
        **method_args,
    )
    logger.info(
        f"{trainer_handler_name} Trainer loaded, output_dir: {trainer_args.output_dir}"
    )
    return trainer, trainer_args


# Register Finetuning Trainer
_register_trainer(Trainer)
_register_trainer(FinetuneTrainer)

# Register Unlearning Trainer
_register_trainer(GradAscent)
_register_trainer(GradDiff)
_register_trainer(NPO)
_register_trainer(DPO)
_register_trainer(SimNPO)
_register_trainer(RMU)
_register_trainer(UNDIAL)
_register_trainer(CEU)
_register_trainer(SatImp)
_register_trainer(WGA)
_register_trainer(PDU)

# Register Knowledge Injection Trainer
_register_trainer(InjectTrainer)
_register_trainer(LoRATrainer)
_register_trainer(DoRATrainer)
_register_trainer(AdaLoRATrainer)
_register_trainer(LoReFTTrainer)

# Register Knowledge Editing Trainer
_register_trainer(EditTrainer)
_register_trainer(ROMEEditor)
_register_trainer(MEMITEditor)
_register_trainer(MENDEditor)
_register_trainer(FTEditor)
_register_trainer(AlphaEditEditor)
_register_trainer(GRACEEditor)
_register_trainer(WISEEditor)
_register_trainer(WISEMergeEditor)
_register_trainer(IKEEditor)
_register_trainer(SERACEditor)
_register_trainer(SERACCompactEditor)
_register_trainer(MALMENEditor)
_register_trainer(InstructEditEditor)
_register_trainer(AnyEditEditor)

# Register Multimodal Knowledge Editing Trainer
_register_trainer(MMIKEEditor)
_register_trainer(MMLoRAEditor)
_register_trainer(MMGRACEEditor)
_register_trainer(MMWISEEditor)
_register_trainer(MMMENDEditor)
_register_trainer(MMSERACEditor)

# Register only when the local research backend is actually installed.
if unke_research_available():
    from trainer.edit.unke_editor import UNKEEditor

    _register_trainer(UNKEEditor)
