import os
import argparse
import pandas as pd
import torch
from sklearn.model_selection import StratifiedShuffleSplit
import timm
from src import (
    seed_everything,
    load_model_weights,
    freeze_weights,
    get_sampler,
    ModelTrainer, 
    ModelTrainerCSV,
    DaconDataset,
    DaconLSTM,
    DaconModel,
    accuracy_function, 
    get_train_transforms, 
    get_valid_transforms,
)


def main():
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    seed_everything(42)

    parser = argparse.ArgumentParser()
    parser.add_argument('--base_folder', type=str, default='./data/train')
    parser.add_argument('--label_fn', type=str, default='./data/train.csv')
    parser.add_argument('--save_folder', type=str, default='./checkpoint')
    parser.add_argument('--kfold_idx', type=int, default=0)

    parser.add_argument('--model', type=str, default='tf_efficientnetv2_s')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    
    parser.add_argument('--device', type=str, default=device)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--comments', type=str, default=None)

    args = parser.parse_args()
    
    assert os.path.isdir(args.base_folder), 'wrong path'
    assert os.path.isfile(args.label_fn), 'wrong path'

    print('=' * 50)
    print('[info msg] arguments\n')
    for key, value in vars(args).items():
        print(key, ":", value)
    print('=' * 50)  

    label_df = pd.read_csv(args.label_fn)

    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)

    for fold_idx, (train_idx, valid_idx) in enumerate(sss.split(X=label_df['image'], y=label_df['label'])):
        if args.kfold_idx == fold_idx:
            train_df = label_df.iloc[train_idx]
            valid_df = label_df.iloc[valid_idx]

    train_dataset = DaconDataset(
        base_folder=args.base_folder,
        label_df=train_df,
        transforms=get_train_transforms(),
    )

    valid_dataset = DaconDataset(
        base_folder=args.base_folder,
        label_df=valid_df,
        transforms=get_valid_transforms(),
    )

    train_sampler = get_sampler(
        df=train_df,
        dataset=train_dataset
    )

    train_data_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            # shuffle=True,
            num_workers=8,
        )

    valid_data_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=8,
        )

    model_cnn = timm.create_model(args.model, pretrained=True, num_classes=25)
    # model_cnn = load_model_weights(model=model_cnn, weight_fn=r'C:\Users\bed1\src\dacon_farm\checkpoint\0128222433_lb0.935156149\model_best.pth')
    rnn_model = DaconLSTM()
    rnn_model = load_model_weights(model=rnn_model, weight_fn=r'C:\Users\bed1\src\dacon_farm\checkpoint\0201101938_rnn\best_model.pth')
    # freeze_weights(model_cnn)
    # freeze_weights(rnn_model)
    model = DaconModel(
        model_cnn=model_cnn,
        model_rnn=rnn_model
    )
    
    loss = torch.nn.CrossEntropyLoss()
    metric = accuracy_function
    optimizer = torch.optim.Adam(model.parameters(), args.lr, betas=(0.9, 0.99))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=25, eta_min=args.lr / 1e3)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer=optimizer,
    #     mode='max',
    #     patience=3,
    #     factor=0.5,
    #     verbose=True
    # )

    trainer = ModelTrainerCSV(
            model=model,
            train_loader=train_data_loader,
            valid_loader=valid_data_loader,
            loss_func=loss,
            metric_func=metric,
            optimizer=optimizer,
            device=args.device,
            save_dir=args.save_folder,
            mode='max', 
            scheduler=scheduler, 
            num_epochs=args.epochs,
            num_snapshops=None,
            parallel=False,
            use_amp=True,
            use_cutmix=True,
            use_csv=True,
            use_wandb=True,            
        )

    trainer.initWandb(
        project_name='dacon_farm',
        run_name=args.comments,
        args=args,
    )

    trainer.train()
    
    with open(os.path.join(trainer.save_dir, 'config.txt'), 'w') as f:
        for key, value in vars(args).items():
            f.write('{} : {}\n'.format(key, value)) 


if __name__ == '__main__':
    main()