from importlib.metadata import requires
import os
from zipfile import BadZipFile
import torch
import numpy as np
import argparse
import time
import torch.nn.functional as Functional

from tqdm import trange
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.autograd import Variable
from WREN import *

dataset_path = 'C:\\Users\\danil\\PycharmProjects\\PFE_Test_Inteligencije\\wren\\wild-relation-network-main\\wrenLib\\neutral'
save_path_model = 'C:\\Users\\danil\\PycharmProjects\\PFE_Test_Inteligencije\\wren\\wild-relation-network-main\\wrenLib\\cuva'
save_path_log = 'C:\\Users\\danil\\PycharmProjects\\PFE_Test_Inteligencije\\wren\\wild-relation-network-main\\wrenLib\\cuvalogs'

os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='WReN')
parser.add_argument('--num_neg', type=int, default=4)

parser.add_argument('--fig_type', type=str, default='*')
parser.add_argument('--dataset', type=str, default='pgm')
parser.add_argument('--root', type=str, default='../dataset/rpm')
parser.add_argument('--moja_putanja', type=str, default=dataset_path)

parser.add_argument('--train_mode', type=bool, default=True)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--img_size', type=int, default=160)
parser.add_argument('--workers', type=int, default=1)
parser.add_argument('--seed', type=int, default=123)

args = parser.parse_args()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#print(device)

if torch.cuda.is_available:
    torch.cuda.manual_seed(args.seed)


tf = transforms.Compose([ToTensor()])
train_set = dataset(args.moja_putanja, 'train', args.fig_type, args.img_size, tf, args.train_mode)
test_set = dataset(args.moja_putanja, 'test', args.fig_type, args.img_size, tf)
val_set = dataset(args.moja_putanja, 'val', args.fig_type, args.img_size, tf)

#print('test length', len(test_set), args.fig_type)

train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)

save_name = args.model_name + '_' + args.fig_type + '_' + str(args.num_neg) + '_' + str(args.img_size) + '_' + str(
    args.batch_size)

if not os.path.exists(save_path_model):
    os.makedirs(save_path_model)
if not os.path.exists(save_path_log):
    os.makedirs(save_path_log)

model = Wild_Relation_Network()
if torch.cuda.device_count() > 0:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = torch.nn.DataParallel(model)
model.to(device)

# model.load_state_dict(torch.load(save_path_model+'/model_01.pth'))

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)


###cuvanje logova
time_now = datetime.now().strftime('%D-%H:%M:%S')
save_log_name = os.path.join(save_path_log, 'sacuvani.txt')
with open(save_log_name, 'a') as f:
    f.write('\n------ lr: {:f}, batch_size: {:d}, img_size: {:d}, time: {:s} ------\n'.format(
        args.lr, args.batch_size, args.img_size, time_now))
f.close()





def compute_loss(predict): #!Cross entropy loss!
    pseudo_target = torch.zeros(predict.shape)
    pseudo_target = Variable(pseudo_target, requires_grad=False).to(device)
    pseudo_target[:, 0:2] = 1

    return Functional.binary_cross_entropy_with_logits(predict, pseudo_target)




def train(epoch):
    model.train()
    metrics = {'loss': []}

    train_loader_iter = iter(train_loader)
    for batch_idx in trange(len(train_loader_iter)):
        try:
            image, _ = next(train_loader_iter)
        except BadZipFile:
            continue

        image = Variable(image, requires_grad=True).to(device)

        predict = model(image)
        loss = compute_loss(predict)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        metrics['loss'].append(loss.item())

        if batch_idx > 1 and batch_idx % 10 == 0:
            print('Epoch: {:d}/{:d},  Loss: {:.3f}'.format(epoch, args.epochs, np.mean(metrics['loss'])))
            model.eval()
            metricsa = {'correct': [], 'count': []}
            val_loader_iter = iter(val_loader)
            """for i in range(500):
                image, target = next(val_loader_iter)
                image = Variable(image, requires_grad=False).to(device)
                target = Variable(target, requires_grad=False).to(device)

                with torch.no_grad():
                    predict = model(image)

                pred = torch.max(predict[:, 2:], 1)[1]
                correct = pred.eq(target.data).cpu().sum().numpy()

                metricsa['correct'].append(correct)
                metricsa['count'].append(target.size(0))

                accuracy = 100 * np.sum(metricsa['correct']) / np.sum(metricsa['count'])

            print('After 1000 validation test Accuracy: {:.3f} \n'.format(accuracy))"""

    print('Epoch: {:d}/{:d},  Loss: {:.3f}'.format(epoch, args.epochs, np.mean(metrics['loss'])))

    return metrics


def test(epoch):
    model.eval()
    metrics = {'correct': [], 'count': []}

    test_loader_iter = iter(test_loader)
    for _ in trange(len(test_loader_iter)):
        image, target = next(test_loader_iter)

        image = Variable(image, requires_grad=False).to(device)
        target = Variable(target, requires_grad=False).to(device)

        with torch.no_grad():
            predict = model(image)

        pred = torch.max(predict[:, :], 1)[1] ###predict[:, 2:]
        correct = pred.eq(target.data).cpu().sum().numpy()

        metrics['correct'].append(correct)
        metrics['count'].append(target.size(0))

        accuracy = 100 * np.sum(metrics['correct']) / np.sum(metrics['count'])

    print('Testing Epoch: {:d}/{:d}, Accuracy: {:.3f} \n'.format(epoch, args.epochs, accuracy))

    return metrics


if __name__ == '__main__':
    for epoch in range(1, args.epochs + 1):

        # metrics_test = test(epoch)
        # break

        metrics_train = train(epoch)

        # Save model
        if epoch > 0:
            save_name = os.path.join(save_path_model, 'model_{:02d}.pth'.format(epoch))
            torch.save(model.state_dict(), save_name)

        metrics_test = test(epoch)

        loss_train = np.mean(metrics_train['loss'])
        acc_test = 100 * np.sum(metrics_test['correct']) / np.sum(metrics_test['count'])

        time_now = datetime.now().strftime('%H:%M:%S')

        with open(save_log_name, 'a') as f:
            f.write('Epoch {:02d}: Accuracy: {:.3f}, Loss: {:.3f}, Time: {:s}\n'.format(
                epoch, acc_test, loss_train, time_now))
        f.close()