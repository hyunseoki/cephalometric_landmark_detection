import os
import argparse
import torch
from src import (    
    LandmarkDataset,
    ModelTrainer,
    UNet,
    load_model_weights,
    seed_everything,
    get_train_transforms, 
    get_valid_transforms,
)

def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    seed_everything(42)

    parser = argparse.ArgumentParser()
    parser.add_argument('--base_folder', type=str, default='./data')
    parser.add_argument('--save_folder', type=str, default='./checkpoint')

    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    
    parser.add_argument('--device', type=str, default=device)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--comments', type=str, default=None)

    args = parser.parse_args()
    
    assert os.path.isdir(args.base_folder), 'wrong path'

    print('=' * 50)
    print('[info msg] arguments')
    for key, value in vars(args).items():
        print(key, ":", value)
    print('=' * 50)

    train_dataset = LandmarkDataset(
        base_folder=os.path.join(args.base_folder, 'train'),
        transforms=get_train_transforms(),
    )

    valid_dataset = LandmarkDataset(
        base_folder=os.path.join(args.base_folder, 'val'),
        transforms=get_valid_transforms(),
    )

    train_data_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=8,
        )

    valid_data_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=8,
        )

    model = UNet()
    if args.resume != None:
        model = load_model_weights(model, args.resume)
    
    loss = torch.nn.MSELoss()
    metric = loss
    optimizer = torch.optim.Adam(model.parameters(), args.lr, betas=(0.9, 0.99))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=25)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 500)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer=optimizer,
    #     mode='max',
    #     patience=3,
    #     factor=0.5,
    #     verbose=True
    # )

    trainer = ModelTrainer(
            model=model,
            train_loader=train_data_loader,
            valid_loader=valid_data_loader,
            loss_func=loss,
            metric_func=metric,
            optimizer=optimizer,
            device=args.device,
            save_dir=args.save_folder,
            mode='min', 
            scheduler=scheduler, 
            num_epochs=args.epochs,
            num_snapshops=int(args.epochs // 25),
            parallel=False,
            use_amp=True,
            use_wandb=False,            
        )

    # trainer.initWandb(
    #     project_name='dacon_farm',
    #     run_name=args.comments,
    #     args=args,
    # )

    trainer.train()
    
    with open(os.path.join(trainer.save_dir, 'config.txt'), 'w') as f:
        for key, value in vars(args).items():
            f.write('{} : {}\n'.format(key, value)) 


if __name__ == '__main__':
    main()