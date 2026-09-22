import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss, DC_and_BCE_loss
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper


class nnUNetTrainerFuseSSMZOH_v3(nnUNetTrainer):
    """
    FuseUNet fusessm_zoh v3: training recipe + deep supervision enabled
    - AdamW + CosineAnnealingLR
    - batch_dice=False (per-sample dice)
    - Loss weight: 0.6 Dice + 0.4 CE
    - 300 epochs
    - weight_decay=1e-2
    - Deep Supervision enabled
    """

    def __init__(self, plans, configuration, fold, dataset_json, unpack_dataset=True, device=None):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 300
        self.initial_lr = 1e-3
        self.weight_decay = 1e-2
        self.enable_deep_supervision = True
        # Increase foreground oversampling to help with small classes
        self.oversample_foreground_percent = 0.5

    def configure_optimizers(self):
        optimizer = AdamW(
            self.network.parameters(),
            lr=self.initial_lr,
            weight_decay=self.weight_decay,
            amsgrad=True
        )
        lr_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.num_epochs,
            eta_min=1e-5
        )
        return optimizer, lr_scheduler

    def _build_loss(self):
        if self.label_manager.has_regions:
            loss = DC_and_BCE_loss(
                {},
                {'batch_dice': False, 'do_bg': True, 'smooth': 1e-5, 'ddp': self.is_ddp},
                use_ignore_label=self.label_manager.ignore_label is not None,
                dice_class=MemoryEfficientSoftDiceLoss
            )
        else:
            loss = DC_and_CE_loss(
                {'batch_dice': False, 'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp},
                {},
                weight_ce=0.4,
                weight_dice=0.6,
                ignore_label=self.label_manager.ignore_label,
                dice_class=MemoryEfficientSoftDiceLoss
            )
        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2**i) for i in range(len(deep_supervision_scales))])
            weights[-1] = 0
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)
        return loss
