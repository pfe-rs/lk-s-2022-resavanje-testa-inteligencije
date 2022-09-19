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

<<<<<<< Updated upstream
dataset_path = 'D:\\testiranje lmao\\testovi'
save_path_model = 'D:\\testiranje lmao\`cuva'
save_path_log = 'D:\\testiranje lmao\cuvalogs'

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

=======
dataset_path = 'C:\\Users\\danil\\PycharmProjects\\PFE_Test_Inteligencije\\wren\\wild-relation-network-main\\wrenLib\\neutral'#lokacija dataseta
#Dataset je podeljen u 3 foldera: train, test i val koji predstavljaju lokaciju train, test i validation seta.
save_path_model = 'C:\\Users\\danil\\PycharmProjects\\PFE_Test_Inteligencije\\wren\\wild-relation-network-main\\wrenLib\\cuva'#lokacija cuvanja modela
save_path_log = 'C:\\Users\\danil\\PycharmProjects\\PFE_Test_Inteligencije\\wren\\wild-relation-network-main\\wrenLib\\cuvalogs' #lokacija cuvanja logova

os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
torch.backends.cudnn.benchmark = True #optimizacija cuda algoritma
>>>>>>> Stashed changes

num_neg = 4
fig_type = "*" #oznaka da se uzimaju svi fajlovi u folderu
train_mode = True
learn_rate = 0.0001
num_epochs = 2
batch_size = 16
img_size = 160
workers = 1
seed = 123

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if torch.cuda.is_available:
    torch.cuda.manual_seed(seed)


tf = transforms.Compose([ToTensor()])
train_set = dataset(dataset_path, 'train', fig_type, img_size, tf, train_mode) #ucitavamo dataset
test_set = dataset(dataset_path, 'test', fig_type, img_size, tf)
val_set = dataset(dataset_path, 'val', fig_type, img_size, tf)


train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True) #pravimo batcheve
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)

save_name = 'WReN' + '_' + fig_type + '_' + str(num_neg) + '_' + str(img_size) + '_' + str(batch_size) #Ime za cuvanje modela

if not os.path.exists(save_path_model):
    os.makedirs(save_path_model) #Formiramo foldere za cuvanje modela i logova u slucaju da ne postoje
if not os.path.exists(save_path_log):
    os.makedirs(save_path_log)

model = Wild_Relation_Network()
if torch.cuda.device_count() > 0:
    print("Let's use", torch.cuda.device_count(), "GPUs!")

model.to(device)

model.load_state_dict(torch.load(save_path_model+'/model_05.pth', map_location=torch.device('cpu')))#Ucitavanje treniranog modela

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learn_rate)#Koristimo Adam optimizer sa lr kao sto je
# navedeno u radu Measuring abstract reasoning in neural networks


###cuvanje logova
time_now = datetime.now().strftime('%D-%H:%M:%S') #ucitavamo trenutno vreme i pretvaramo te brojeve u string
save_log_name = os.path.join(save_path_log, 'save_log.txt') #ime save log tekstualnog dokumenta
train_log_name = os.path.join(save_path_log, "train_log.txt")#ime train log tekstualnog dokumenta

with open(save_log_name, 'a') as f:
    f.write('\n------ lr: {:f}, batch_size: {:d}, img_size: {:d}, time: {:s} ------\n'.format(
        learn_rate, batch_size, img_size, time_now))
<<<<<<< Updated upstream
f.close()

loss_fn = nn.CrossEntropyLoss().to(device)
=======
f.close()#Zapisivanje lr, batch_size, img_size i vreme u trenutku pokretanja u save log fajl
>>>>>>> Stashed changes

loss_fn = nn.CrossEntropyLoss()#Za loss funkciju koristimo crossEntropyLoss kao u radu

def validation_accuracy():
    """Izracunavamo validation accuracy tako sto ucitavamo batcheve i za svaki batch zapisujemo broj tacnih odogovra i
    broj predjenih validation primera. Nakon izracunamo tacnost deljenjem broja tacno uradjenih validation primera sa
     ukupnim brojem primera"""
    model.eval()
    iter_val = iter(val_loader)
    metricsa = {'correct': [], 'count': []}# Kako ne bismo ucitali vrednosti prethodnih merenja verovatnoce
    for i in range(len(iter_val)):#Pokrecemo za svaki batch
        image, target = next(iter_val)
        image = Variable(image, requires_grad=False).to(device)
        target = Variable(target, requires_grad=False).to(device)

        with torch.no_grad():
            predict = model(image)

        pred = torch.max(predict[:, :], 1)[1]
        correct = pred.eq(target.data).cpu().sum().numpy() #poredjenje predvidjanja koje dobijamo iz modela i tacnih odgovora

        metricsa['correct'].append(correct)
        metricsa['count'].append(target.size(0))

    acc_val = 100 * np.sum(metricsa['correct']) / np.sum(metricsa['count'])
    model.train() #Kako ne bi model ostao u eval modu posle prvog izracunavanja preciznosti u toku epohe
    return acc_val


def train(epoch):
    """Treniramo model i proveravamo na svakih 120_000 test primera preciznost u validation setu"""
    model.train()
    metrics = {'loss': [], 'correct': [], 'count': []}  #Ucitavamo praznu matricu za zapisivanje loss-a, tacnih
    # predvidjanja modela i ukupnog broja predjenih primera

    train_loader_iter = iter(train_loader) #ucitavamo iteracije train seta podeljenog na batcheve

    for batch_idx in trange(len(train_loader_iter)):
        try:#proveravamo da li su fajlovi pokvareni
            image, target = next(train_loader_iter)
        except BadZipFile:# ako jesu samo preskacemo iteraciju
            continue

        image = Variable(image, requires_grad=True).to(device)

        target = target.to(device)
        predict = model(image)

        loss = loss_fn(predict, target)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        metrics['loss'].append(loss.item())

        pred = torch.max(predict[:, :], 1)[1]  # Oznacavamo sa 1 tacan odogovor a sa 0 sve ostale
        correct = pred.eq(target.data).cpu().sum().numpy()

        metrics['correct'].append(correct)
        metrics['count'].append(target.size(0))

        if batch_idx > 1 and batch_idx % 120_000 == 0:
            print('Epoch: {:d}/{:d},  Loss: {:.8f}'.format(epoch, num_epochs, np.mean(metrics['loss'])))

            acc_val = validation_accuracy()
            print(' Validation Accuracy: {:.8f} \n'.format(acc_val))

            acc_train= 100 * np.sum(metrics['correct']) / np.sum(metrics['count'])

            with open(train_log_name, 'a') as f:
                f.write('Epoch {:02d}: Batch_idx {:d}: Acc_val {:.8f}: Acc_train {:.8f}: Loss {:.8f}: Time {:s}\n'.format(
                    epoch, batch_idx,  acc_val,acc_train, np.mean(metrics['loss']), time_now))

            metrics = {'loss': [], 'correct': [], 'count': []} #praznimo listu kako bismo izracunali accuracy za
            # sledecih 120_000 primera a ne za celu epohu.


    accuracy = 100 * np.sum(metrics['correct']) / np.sum(metrics['count'])

    print('Epoch: {:d}/{:d},  Loss: {:.8f}, Acc: {:.8f}'.format(epoch, num_epochs, np.mean(metrics['loss']),
                                                                accuracy))  # acc

    return metrics


def test(epoch):
    """Slicno train samo bez validation dela"""
    model.eval()
    metrics = {'correct': [], 'count': []}

    test_loader_iter = iter(test_loader)
    for _ in trange(len(test_loader_iter)):
        image, target = next(test_loader_iter)

        image = Variable(image, requires_grad=False).to(device)
        target = Variable(target, requires_grad=False).to(device)

        with torch.no_grad():
            predict = model(image)

        pred = torch.max(predict[:, :], 1)[1]
        correct = pred.eq(target.data).cpu().sum().numpy()

        metrics['correct'].append(correct)
        metrics['count'].append(target.size(0))

        accuracy = 100 * np.sum(metrics['correct']) / np.sum(metrics['count'])

    print('Testing Epoch: {:d}/{:d}, Accuracy: {:.8f} \n'.format(epoch, num_epochs, accuracy))

    return metrics


if __name__ == '__main__':
    for epoch in range(1, num_epochs + 1):

        metrics_train = train(epoch)
        if epoch > 0:
            save_name = os.path.join(save_path_model, 'model_{:02d}.pth'.format(epoch))
            torch.save(model.state_dict(), save_name) #Cuvamo model posle svake epohe

    metrics_test = test(epoch)

    acc_test = 100 * np.sum(metrics_test['correct']) / np.sum(metrics_test['count']) #preciznost za test set

    time_now = datetime.now().strftime('%H:%M:%S')

    with open(save_log_name, 'a') as f:
        f.write('Epoch {:02d}: Accuracy: {:.3f}, Time: {:s}\n'.format(
            epoch, acc_test, time_now)) #zapisujemo broj epoha, preciznost na test setu i vreme kada se testiranje zavrsilo