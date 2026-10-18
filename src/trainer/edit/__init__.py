# Knowledge Editing (知识编辑) 训练器模块
# 支持 ROME, MEMIT, MEND, MALMEN, IKE, SERAC, GRACE, WISE,
# AlphaEdit, UNKE, InstructEdit, AnyEdit 等知识编辑方法
# 以及多模态版本: MM-IKE, MM-LoRA, MM-GRACE, MM-WISE, MM-MEND, MM-SERAC

from trainer.edit.base import EditTrainer
from trainer.edit.rome import ROMEEditor
from trainer.edit.memit import MEMITEditor
from trainer.edit.mend import MENDEditor
from trainer.edit.malmen import MALMENEditor
from trainer.edit.ike import IKEEditor
from trainer.edit.serac import SERACEditor
from trainer.edit.serac_compact import SERACCompactEditor
from trainer.edit.grace import GRACEEditor
from trainer.edit.wise import WISEEditor
from trainer.edit.wise_merge import WISEMergeEditor
from trainer.edit.alphaedit import AlphaEditEditor
from trainer.edit.instructedit import InstructEditEditor
from trainer.edit.anyedit import AnyEditEditor, unke_research_available

from trainer.edit.mm_mixin import MMEditMixin
from trainer.edit.mm_ike import MMIKEEditor
from trainer.edit.mm_lora import MMLoRAEditor
from trainer.edit.mm_grace import MMGRACEEditor
from trainer.edit.mm_wise import MMWISEEditor
from trainer.edit.mm_mend import MMMENDEditor
from trainer.edit.mm_serac import MMSERACEditor

__all__ = [
    "EditTrainer",
    "ROMEEditor",
    "MEMITEditor",
    "MENDEditor",
    "MALMENEditor",
    "IKEEditor",
    "SERACEditor",
    "SERACCompactEditor",
    "GRACEEditor",
    "WISEEditor",
    "WISEMergeEditor",
    "AlphaEditEditor",
    "InstructEditEditor",
    "AnyEditEditor",
    "MMEditMixin",
    "MMIKEEditor",
    "MMLoRAEditor",
    "MMGRACEEditor",
    "MMWISEEditor",
    "MMMENDEditor",
    "MMSERACEditor",
]

# Research-only code is omitted from public distributions.
if unke_research_available():
    from trainer.edit.unke_editor import UNKEEditor as UNKEEditor

    __all__.append("UNKEEditor")
