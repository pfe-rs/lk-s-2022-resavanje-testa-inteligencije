from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import glob
import cv2
import torch
from torchvision import transforms
from torch import nn




class KNN(nn.Module):
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

    def forward(self, panel_embeddings: torch.Tensor) -> torch.Tensor:

        batch_size = panel_embeddings.shape[0]
        tags = torch.zeros((16, 9), device=panel_embeddings.device).type_as(panel_embeddings)
        tags[:8, :8] = torch.eye(8, device=panel_embeddings.device).type_as(panel_embeddings)
        tags[8:, 8] = torch.ones(8, device=panel_embeddings.device).type_as(panel_embeddings)
        tags = tags.expand((batch_size, -1, -1))
        return torch.cat([panel_embeddings, tags], dim=2)


class GroupContextPanels(nn.Module):
    def __init__(self):
        super(GroupContextPanels, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_objects, object_size = x.size()
        return torch.cat([
            x.unsqueeze(1).repeat(1, num_objects, 1, 1),
            x.unsqueeze(2).repeat(1, 1, num_objects, 1)
        ], dim=3).view(batch_size, num_objects ** 2, 2 * object_size)


class GroupContextPanelsWith(nn.Module):
    def forward(self, objects: torch.Tensor, object: torch.Tensor) -> torch.Tensor:
        return torch.cat([
            objects,
            object.unsqueeze(1).repeat(1, 8, 1)
        ], dim=2)



class LinearBn(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super(LinearBn, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor):
        x = self.linear(x)
        oblik = x.shape
        x = x.flatten(0, -2)
        x = self.bn(x)
        x = x.view(oblik)
        x = self.relu(x)
        return x



class DeepLinearLayerG(nn.Module):
    def __init__(self):
        super(DeepLinearLayerG, self).__init__()
        self.mlp = nn.Sequential(
            LinearBn(5202, 512),
            LinearBn(512, 512),
            LinearBn(512, 512),
            LinearBn(512, 512)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        x = x.sum(dim=1)
        return x

class DeepLinearLayerF(nn.Module):
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
        batch_size, num_panels, height, width = x.size()
        x = x.view(batch_size * num_panels, 1 , height, width)
        x = self.cnn(x)
        x = x.view(batch_size, num_panels, 32 * 9 * 9) #zasto object size 32*9*9
        x = self.tagovanje(x)
        context_panels = x[:, :8, :]
        solution_panels = x[:, 8:, :]
        context_pairs = self.group_context_panels(context_panels)
        context_pairs_g_out = self.g_function(context_pairs)
        f_out = torch.zeros(batch_size, 8, device=x.device)
        for i in range(8):
            context_solution_pairs = self.group_with_answers(context_panels, solution_panels[:, i, :])
            context_solution_pairs_g_out = self.g_function(context_solution_pairs)
            relations = context_pairs_g_out + context_solution_pairs_g_out
            relations = self.norm(relations)
            f_out[:, i] = self.f_function(relations).squeeze()
        return torch.softmax(f_out, dim = 1)



###Dataloader:

class ToTensor(object):
    def __call__(self, sample):
        return torch.tensor(np.array(sample), dtype=torch.float32)


class dataset(Dataset):
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


###Primer ucitavanja

if __name__ == "__main__":
    train_set = dataset(root='C:\\Users\\danil\\PycharmProjects\\PFE_Test_Inteligencije\\ncd\\neutral',
                        dataset_type='train',
                        fig_type="*",
                        img_size=160,
                        transform=transforms.Compose([ToTensor()]))

    train_loader = DataLoader(train_set, batch_size=16, shuffle=True, num_workers=1, pin_memory=False)

    slike, labels = next(iter(train_loader))


    print(type(slike), slike.size())
    print(labels)


    model = Wild_Relation_Network()
    preds = model(slike)

    print(preds)
