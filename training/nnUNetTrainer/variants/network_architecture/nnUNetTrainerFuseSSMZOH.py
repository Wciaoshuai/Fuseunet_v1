from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainerFuseSSMZOH(nnUNetTrainer):
    def __init__(self, plans, configuration, fold, dataset_json, unpack_dataset=True, device=None):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 300
