import os
import argparse
import numpy as np
import model
import utils
from torch.utils.data import DataLoader
import warnings
from train import *

warnings.filterwarnings(action='ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

parser = argparse.ArgumentParser(description="Cross-subject")
parser.add_argument("-g", "--gpu", type=int, default=0)
parser.add_argument("-te", "--test", type=str, default='9')
parser.add_argument("-m", "--model", type=str, default='EEGNet')
parser.add_argument("-e", "--epoch", type=int, default=100)
parser.add_argument("-l", "--learningrate", type=float, default=0.0005)
parser.add_argument("-t", "--is_training", type=str, default='train')
parser.add_argument("-b", "--batch", type=int, default=128)
parser.add_argument("-d", "--data", type=str, default='BCI4_2b')
parser.add_argument("--size", type=int, default=750)
parser.add_argument("--rest", type=bool, default=False)
parser.add_argument("--cdist", type=float, default = 1e-5)
parser.add_argument("--checkpoint", type=str)

args = parser.parse_args()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

torch.multiprocessing.set_sharing_strategy('file_system')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
args.device = device
test = list(map(int, args.test.split(',')))

if args.data == 'BCI4_2b':
    test = list(map(int, args.test.split(',')))
    train = [idx for idx in list(range(1, 10)) if not idx in test]
    args.datanum = 2
    args.chnum = 3
    args.sessionnum = 5
if args.data == 'OpenBMI':
    test = list(map(int, args.test.split(',')))
    train = [idx for idx in list(range(1, 55)) if not idx in test]
    args.datanum = 2
    args.chnum = 62
    args.sessionnum = 2
if args.data == 'BCI4_2a':
    test = list(map(int, args.test.split(',')))
    train = [idx for idx in list(range(1, 10)) if not idx in test]
    args.datanum = 4
    args.chnum = 22
    args.sessionnum = 2

args.train = train
net = model.Distribution(args).to(device)
args.hyper = [1, 0.5, 0.05, 0]
args.hyper_finetune = [1, 0, 0, 0]
loss = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=args.learningrate)

if args.is_training == 'train':
    os.makedirs(args.checkpoint, exist_ok=True)
    dataset = [] # Set dataset # return x_anc, x_pos, x_neg, label, x_anc_rest, x_pos_rest, x_neg_rest
    data_tr = DataLoader(dataset, batch_size=args.batch, shuffle=True, num_workers=8, drop_last=True)
    data_val = DataLoader(dataset, batch_size=len(dataset), shuffle=False, num_workers=8)
    manage = utils.find_best_model(args.checkpoint, args)
    manage.code_copy(os.path.join(args.checkpoint, 'run'))

    for epoch in range(args.epoch):
        net.train()
        x, y, output, acc_epoch, loss_epoch, count_epoch = one_epoch_distribution(net, data_tr, optimizer, loss, args.hyper, args, train='train')
        print('Epoch  : ', epoch + 1, 'Acc: ', round(acc_epoch, 7), 'Loss: ', np.round(loss_epoch['Total'], 5))

        net.eval()
        with torch.no_grad():
            x, y, output, acc_epoch, loss_epoch, count_epoch = one_epoch_distribution(net, data_val, optimizer, loss, args.hyper, args, train='val')
            manage.update(net, args.checkpoint, epoch, acc_epoch, loss_epoch['CLS'])
            print('Val    : ', epoch + 1, 'Acc: ', round(acc_epoch, 7), 'Loss: ', np.round(loss_epoch['Total'], 5))

    manage.training_finish(net, args.checkpoint)

elif args.is_training == 'test':
    manage = utils.test_model(args.checkpoint, args, 'result')
    dataset = [] # Set dataset # return x_anc, x_pos, x_neg, label, x_anc_rest, x_pos_rest, x_neg_rest
    dataset_RS = [] # Set dataset # return x_anc_rest
    restore = torch.load(os.path.join(args.checkpoint, 'model-best.pth'), map_location=torch.device('cpu'))
    net.load_state_dict(restore, strict=True)

    for session in range(args.sessionnum):
        torch.cuda.empty_cache()
        net.load_state_dict(restore, strict=True)
        restore_before = net.state_dict()
        data_test_RS = DataLoader(dataset_RS, batch_size=1, shuffle=False, num_workers=8)
        RS, RS_update, RS_update_label = generate_signal_from_RS(net, data_test_RS, loss, args.hyper, args, train='train')
        RS_update = torch.cat(RS_update, dim=0)
        RS = torch.cat(RS, dim=0)
        RS_update_label = torch.cat(RS_update_label, dim=0)
        net.load_state_dict(restore_before, strict=True)

        dataset_RS.testset[session]['x'] = torch.cat([RS, RS_update], dim=2)
        dataset_RS.testset[session]['y'] = RS_update_label

        net.train()
        data_test_RS = DataLoader(dataset_RS, batch_size=args.batch, shuffle=True, num_workers=8)
        net.load_state_dict(restore_before)
        optimizer = torch.optim.Adam(net.parameters(), lr=args.learningrate/10 )
        kd_loss = torch.nn.KLDivLoss().to(args.device)
        for epoch in range(10):
            args.cdist = 0
            x, y, output, acc_epoch, loss_epoch, count_epoch = one_epoch_distribution(net, data_test_RS, optimizer, kd_loss, args.hyper_finetune, args, train='RS')

        dataset.is_training = 'test'
        dataset.finetuning = False
        data_test = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8)
        net.eval()
        with torch.no_grad():
            x, y, output, acc_epoch, loss_epoch, count_epoch = one_epoch_distribution(net, data_test, optimizer, loss, args.hyper, args, train='val')
            print(str(test[0])+'S'+str(session)+': ', count_epoch, 'Acc: ', round(acc_epoch, 7), 'Loss: ', round(loss_epoch['Total'], 7))
            manage.total_result(str(test[0])+'S'+str(session), count_epoch, round(acc_epoch, 7), round(loss_epoch['Total'], 7))