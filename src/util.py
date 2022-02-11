import os
import random
import numpy as np
import torch


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_model_weights(model, weight_fn):
    model_state_dict = torch.load(weight_fn)
    try:
        model.load_state_dict(model_state_dict, strict=True)
    except RuntimeError:
        # if model was trained in parallel
        from collections import OrderedDict
        new_model_state_dict = OrderedDict()

        for k, v in model_state_dict.items():
            k = k.replace('module.', '')
            new_model_state_dict[k] = v

        model.load_state_dict(new_model_state_dict, strict=True)

    return model

def freeze_weights(model):
    for param in model.parameters():
        param.requires_grad = False


def get_sampler(df, dataset):
    labels = [dataset.encode(label) for label in df['label'].to_list()]
    _, class_counts = np.unique(df['label'], return_counts=True)
    total_num = sum(class_counts)

    class_weights = [total_num/class_counts[i] for i in range(len(class_counts))]
    weights = [class_weights[labels[i]] for i in range(int(total_num))]
    sampler = torch.utils.data.WeightedRandomSampler(torch.DoubleTensor(weights), int(total_num), replacement=True)

    return sampler