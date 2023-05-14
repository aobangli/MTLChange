from torch.utils.data import DataLoader
from Source.ChangePrediction.data_loader import load_data
from Source.ChangePrediction.TrainConfig import *
from Source.ChangePrediction.models import snr_trans
from Source.ChangePrediction.models.aitm import AITMModel

from trainer.MultiTrainer import MultiTrainer
from trainer.MultiTrainerWeightingLoss import MultiTrainerWeightingLoss
import Source.ChangePrediction.loss_weighting_strategy.EW as EW_strategy
import Source.ChangePrediction.loss_weighting_strategy.UW as UW_strategy
import Source.ChangePrediction.loss_weighting_strategy.DWA as DWA_strategy


def transfer_task_types():
    types = []
    for label_type in label_types:
        if label_type == TaskType.Binary_Classification:
            types.append('binary')
        elif label_type == TaskType.Regression:
            types.append('regression')
        elif label_type == TaskType.Multiple_Classification:
            types.append('multiclass')
        else:
            raise ValueError("task must be binary, multiclass or regression, {} is illegal".format(label_type))
    return types


def multi_weighting_test():
    model_args_dict = {
        'num_feature': len(feature_list),
        'SB_hidden': 1024,
        'SB_output': 512,
        'towers_hidden': 16,
        'num_experts': 5,
        'num_tasks': num_of_labels,
        'task_types': transfer_task_types(),
        'task_out_dims': task_out_dims
    }

    weight_args_dict = EW_strategy.default_args_dict

    optim_args_dict = {
        'optim': 'adam',
        'lr': snr_trans_config['lr'],
        # 'weight_decay': widedeep_config['l2_regularization']
    }

    weighting_trainer = MultiTrainerWeightingLoss(
        model=snr_trans.Model,
        weighting=EW_strategy.EW,
        config=snr_trans_config,
        model_args_dict=model_args_dict,
        weight_args_dict=weight_args_dict,
        optim_args_dict=optim_args_dict,
        device=train_device
    )

    weighting_trainer.train(train_dataset)
    weighting_trainer.test(test_dataset)


if __name__ == '__main__':
    snr_trans_config = \
        {
            'model_name': 'snr_trans',
            'num_epoch': 15,
            'batch_size': 256,
            'lr': 1e-4,
            'l2_regularization': 1e-4,
        }

    train_dataset, test_dataset = load_data.load_dataset()

    multi_weighting_test()
