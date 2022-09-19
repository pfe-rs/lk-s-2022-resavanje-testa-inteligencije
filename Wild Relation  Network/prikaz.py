import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from WREN import *


odgovori = []
t = 0
for k in range(134973, 135500):
    data = np.load("C:\\Users\\danil\\PycharmProjects\\PFE_Test_Inteligencije\\wren\\wild-relation-network-main\\wrenLib\\"
                   "neutral\\test\\PGM_neutral_test_{}.npz".format(k))
    images = data["image"].reshape(16, 160, 160)
    target = data["target"]
    odgovori.append(target)

    meta_target = data["meta_target"]

    fig = plt.figure(figsize=(13, 10))
    rows = 7
    columns = 3


    for i in range(8):
        fig.add_subplot(rows, columns, i + 1)
        plt.imshow(images[i], cmap="gray")
        plt.axis('off')


    for i in range(10, 18):
        fig.add_subplot(rows, columns, i + 3)
        plt.title(i-10)
        plt.imshow(images[i - 2], cmap="gray")
        plt.axis('off')
    fig.tight_layout(pad=1)

    plt.show()

    t += 1
    if t % 10 ==0:
        print(odgovori)
        odgovori = []