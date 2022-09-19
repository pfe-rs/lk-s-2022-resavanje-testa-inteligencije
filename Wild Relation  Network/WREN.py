from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import glob
import cv2
import torch
from torchvision import transforms
from torch import nn


class KNN(nn.Module):
    """KNN koji primenjujemo na context i choice panele kako bismo dobili reprezentacije istih. Sastoji se od 4 Conv2d
    sloja koji su praćeni batch normalisation-om i ReLU funkcijom aktivacije.
    """
    def __init__(self, **kwargs):
        super(KNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class TagPanelEmbeddings(nn.Module):
    """Funkcija kojom formiramo oznake za položaj svakog context i choice panela i dodajemo na 3. dimenziju ulaznog tensora.
    Ulaz je tensor oblika [batch_size, num_samples, object_size] a izlaz oblika [batch_size, num_samples, object_size + 9].
    torch.eye daje tensor u kome su sve vrednosti nule osim dijagonale čije su vrednosti 1.
    torch.expand proširuje tensor u veličinu zadatih vrednosti. Ako se prosledi veličina -1 to označava da se veličina ne menja"""
    def forward(self, panel_embeddings: torch.Tensor) -> torch.Tensor:

        batch_size = panel_embeddings.shape[0]
        tags = torch.zeros((16, 9), device=panel_embeddings.device).type_as(panel_embeddings)
        tags[:8, :8] = torch.eye(8, device=panel_embeddings.device).type_as(panel_embeddings)
        tags[8:, 8] = torch.ones(8, device=panel_embeddings.device).type_as(panel_embeddings)
        tags = tags.expand((batch_size, -1, -1))
        return torch.cat([panel_embeddings, tags], dim=2)


class GroupContextPanels(nn.Module):
    """Funkcija kojom grupišemo reprezentacije context panela u parove.  Ulaz je tensor oblika
    [batch_size, num_samples, object_size] a izlaz tensor velicina [batch_size, num_objects **2, 2 * object_size].
    .unsqueeze(1).repeat(1, num_objects, 1, 1) formira 4d tensor u kojem je prosirena prva dimenzija a
    .unsqueeze(2).repeat(1, 1, num_objects, 1) formira 4d tensor u kojem je prosirena druga dimenzija
    kada se konkateniraju ta 2 tensora po 3. dimenziji dobija se tensor koji sadrzi sve parove reprezentacija"""
    def __init__(self):
        super(GroupContextPanels, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_objects, object_size = x.size()
        return torch.cat([
            x.unsqueeze(1).repeat(1, num_objects, 1, 1),
            x.unsqueeze(2).repeat(1, 1, num_objects, 1)
        ], dim=3).view(batch_size, num_objects ** 2, 2 * object_size)


class GroupContextPanelsWith(nn.Module):
    """Funkcija kojom formiramo parove context panela i choice panela. Ulazi su tensor koji sadrzi context panele i
    tensor koji sadrzi jedan choice panel koji su oblika [batch_size, num_objects = 8, object_size] i [batch_size, object_size]
     a izlaz je tensor oblika [batch_size, num_objects, object_size * 2].
     object.unsqueeze(1).repeat(1, 8, 1) prosiruje tensor koji sadrzi jedan choice panel za svaki batch po prvoj dimenziji"""
    def forward(self, objects: torch.Tensor, object: torch.Tensor) -> torch.Tensor:
        return torch.cat([
            objects,
            object.unsqueeze(1).repeat(1, 8, 1)
        ], dim=2)



class LinearBn(nn.Module):
    """Linearni sloj sa batch normalisation-om i ReLU funkcijom aktivacije"""
    def __init__(self, in_dim: int, out_dim: int):
        super(LinearBn, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor):
        x = self.linear(x) #Primenjujemo linearni sloj
        oblik = x.shape #Cuvamo oblik tensora
        x = x.flatten(0, -2) # flatten pre batch norma
        x = self.bn(x)
        x = x.view(oblik) #vracamo prvobitni oblik tensora
        x = self.relu(x)
        return x



class DeepLinearLayerG(nn.Module):
    """Funkcija koja trazi relacije izmedju svakog para panela."""
    def __init__(self):
        super(DeepLinearLayerG, self).__init__()
        self.mlp = nn.Sequential(
            LinearBn(5202, 512), #ulaz je 5202 jer je to velicina izlaza iz KNN-a
            LinearBn(512, 512),
            LinearBn(512, 512),
            LinearBn(512, 512)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        x = x.sum(dim=1) # sumiramo vrednost po prvoj dimenziji
        return x

class DeepLinearLayerF(nn.Module):
    """Funkcija koja ocenjuje svaki choice panel"""
    def __init__(self):
        super(DeepLinearLayerF, self).__init__()
        self.mlp = nn.Sequential(
            LinearBn(512, 256),
            LinearBn(256, 256),
            #.Dropout(0.5),
            nn.Linear(256, 1)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        return x


class Wild_Relation_Network(nn.Module):
    def __init__(self):
        super(Wild_Relation_Network, self).__init__()
        self.cnn = KNN()
        self.tagovanje = TagPanelEmbeddings()
        self.group_context_panels = GroupContextPanels()
        self.group_with_answers = GroupContextPanelsWith()
        self.g_function = DeepLinearLayerG()
        self.f_function = DeepLinearLayerF()
        self.norm = nn.LayerNorm([512])

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        batch_size, num_panels, height, width = x.size() #izvlacimo dimenzije ulaznog tensora
        x = x.view(batch_size * num_panels, 1 , height, width) # menjamo velicinu tensora x kako bismo ga pustili kroz KNN
        x = self.cnn(x)
        x = x.view(batch_size, num_panels, 32 * 9 * 9) #Menjamo oblik posle KNN-a, velicina 32 oznacava num channels u KNN-u a 9*9 su dimenzije
        #interpretacija slika posle KNN-a, koje su hiperparametri
        x = self.tagovanje(x) #oznacavanje choice i context panela
        context_panels = x[:, :8, :]
        solution_panels = x[:, 8:, :]
        context_pairs = self.group_context_panels(context_panels)
        context_pairs_g_out = self.g_function(context_pairs) # dobijamo ocenu relacije izmedju svaka 2 context panela
        f_out = torch.zeros(batch_size, 8, device=x.device) #formiramo izlazni tensor
        for i in range(8):
            context_solution_pairs = self.group_with_answers(context_panels, solution_panels[:, i, :])
            context_solution_pairs_g_out = self.g_function(context_solution_pairs)# dobijamo ocenu relacije izmedju svaka 2 context i choice panela
            relations = context_pairs_g_out + context_solution_pairs_g_out #spajamo tensore sa ocenama choice-choice i choice-context
            relations = self.norm(relations)# vrsimo normalizaciju na izlazne verovatnoce
            f_out[:, i] = self.f_function(relations).squeeze() # primenjujemo funkciju f i dajemo ocenu svakom choice panelu koja predstavlja koliko je taj panel dobar izbor
        return torch.softmax(f_out, dim = 1)# Softmax po svakom redu, tj na sve odgovore u jednom test primeru



###Dataloader:

class ToTensor(object):
    """Formiranje tensora od array-a"""
    def __call__(self, sample):
        return torch.tensor(np.array(sample), dtype=torch.float32)


class dataset(Dataset):
    """Funkcija za ucitavanje batcheva"""
    def __init__(self, root, dataset_type, fig_type='*', img_size=160, transform=None, train_mode=False):
        self.transform = transform
        self.img_size = img_size
        self.train_mode = train_mode
        self.file_names = [f for f in glob.glob(os.path.join(root, fig_type, '*.npz')) if dataset_type in f]

        if self.train_mode:
            idx = list(range(len(self.file_names)))
            np.random.shuffle(idx)

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        data = np.load(self.file_names[idx])
        image = data['image'].reshape(16, 160, 160)
        target = data['target']

        del data

        resize_image = image
        if self.img_size is not None:
            resize_image = []
            for idx in range(0, 16):
                resize_image.append(
                    cv2.resize(image[idx, :], (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST))
            size_image = np.stack(resize_image)

        if self.transform:
            resize_image = self.transform(resize_image)
            targe = torch.tensor(target, dtype=torch.long)

        return resize_image, target

