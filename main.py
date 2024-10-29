import argparse
import torch
import torch.optim as optim
from sklearn.model_selection import KFold
from dataset import SwallowDataset, collate_fn
from torch.utils.data import DataLoader
from model import CFSCNet
from train import Trainclass
import pandas as pd
from torch.optim.lr_scheduler import OneCycleLR
# from torchsummary import summary
import statistics
import os
from data_util import initialize_weights_normal, predict_calculate_overlap


def main():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                        help='input batch size for testing (default: 256)')
    parser.add_argument('--epochs', type=int, default=300, metavar='N',
                        help='number of epochs to train (default: 300)')
    parser.add_argument('--lr', type=float, default=0.00005, metavar='LR',
                        help='learning rate (default: 0.00005)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--output-pth', type=str, default='CFSCNet.pth', metavar='N',
                        help='save path of pth file')
    parser.add_argument('--rate', type=int, default=4000,
                        help='data rate')
    parser.add_argument('--splits', type=int, default=10,
                        help='fold splits number')
    parser.add_argument('--win-duration', type=int, default=3,
                        help='data rate')
    parser.add_argument('--hop-duration', type=int, default=2,
                        help='data rate')
    parser.add_argument('--data-dir', type=str, default="data",
                        help='directory of data')

    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 2} if use_cuda else {}

    meta_data = pd.read_csv(os.path.join("data", "meta_data.csv"))
    patients = meta_data['patient'].drop_duplicates()
    kf = KFold(n_splits=args.splits, shuffle=True, random_state=args.seed)
    fold_avg_acc, fold_avg_sen, fold_avg_spe, fold_avg_auc, fold_avg_jsc = [], [], [], [], []

    for fold, (train_idx, val_idx) in enumerate(kf.split(patients), 1):
        train_patient = patients.iloc[train_idx].tolist()
        test_patient = patients.iloc[val_idx].tolist()
        train_list, test_list = [], []
        for _, row in meta_data.iterrows():
            patient = row['patient']
            file_name = row['file_name']
            if patient in train_patient:
                train_list.append(file_name)
            elif patient in test_patient:
                test_list.append(file_name)
        Aspiration_train_dataset = SwallowDataset("data",
                                                  train_list,
                                                  args.win_duration,
                                                  args.hop_duration,
                                                  args.rate,
                                                  args.batch_size)
        Aspiration_test_dataset = SwallowDataset("data",
                                                 test_list,
                                                 args.win_duration,
                                                 args.hop_duration,
                                                 args.rate,
                                                 args.test_batch_size)

        train_loader = DataLoader(
            Aspiration_train_dataset,
            batch_size=1,
            collate_fn=collate_fn,
            **kwargs
        )

        test_loader = DataLoader(
            Aspiration_test_dataset,
            batch_size=1,
            collate_fn=collate_fn,
            **kwargs
        )
        model = CFSCNet()
        model.apply(initialize_weights_normal)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.01)
        model = model.to(device)
        # summary(model, input_size=(3, 12000), device=device)
        loss_fn = torch.nn.BCELoss(reduction='none')
        scheduler = OneCycleLR(optimizer,
                               max_lr=args.lr,
                               steps_per_epoch=len(train_loader),
                               epochs=args.epochs,
                               anneal_strategy='cos')
        train = Trainclass(args, model, loss_fn, device, train_loader, 
                           test_loader, optimizer, scheduler)
        fold_acc, fold_sen, fold_spe, fold_auc = train.run(fold)
        fold_avg_acc.append(fold_acc)
        fold_avg_sen.append(fold_sen)
        fold_avg_spe.append(fold_spe)
        fold_avg_auc.append(fold_auc)
        jsc_list = predict_calculate_overlap(model, f"{args.data_dir}/signals", test_loader, args.output_pth, device)
        fold_jsc = statistics.mean(jsc_list)
        fold_avg_jsc.append(fold_jsc)
        print(f"Fold[{fold}] Avg_jsc={fold_jsc:.2f}%(±{statistics.stdev(jsc_list)})")
    print(f"Final Avg Result: avg_acc={statistics.mean(fold_avg_acc):.2f}%(±{statistics.stdev(fold_avg_acc)}) "
          f"avg_sen={statistics.mean(fold_avg_sen):.2f}% (±{statistics.stdev(fold_avg_sen)}) " +
          f"avg_spe={statistics.mean(fold_avg_spe):.2f}% (±{statistics.stdev(fold_avg_spe)}) " + 
          f"avg_auc={statistics.mean(fold_avg_auc):.2f}% (±{statistics.stdev(fold_avg_auc)}) " +
          f"avg_jsc={statistics.mean(fold_avg_jsc):.2f}% (±{statistics.stdev(fold_avg_jsc)})")


if __name__ == '__main__':
    main()
