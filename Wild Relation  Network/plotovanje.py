import matplotlib.pyplot as plt
import os

#f = open('1-2 epoha train only.txt', 'r')
epoch = []
accuracy = []
loss = []
percentage = [0.08, 0.16, 0.24, 0.32, 0.40, 0.48, 0.56, 0.64, 0.72, 0.80, 0.88, 0.96, 1, 1.08, 1.16, 1.24, 1.32, 1.40,
              1.48, 1.56, 1.64, 1.72, 1.80, 1.88, 1.96, 2]

import matplotlib.pyplot as plt


data = {'Nikola Milenić': 2,'Đorđe Marjanović': 2, 'Dragan Mićić': 3, 'Nataša Jovanović': 4, 'Luka Simić': 5}
imena = list(data.keys())
vrednosti = list(data.values())

fig = plt.figure(figsize=(10, 5))

plt.bar(imena, vrednosti, color='maroon',
        width=0.4)
plt.plot(['Nikola Milenić', 'Luka Simić'], [3.7717, 3.7717], 'c' ,lw=2, label = "WReN model nakon 5 epoha")

plt.show
plt.ylim([0, 10])
plt.xlabel("Saradnici")
plt.ylabel("Broj tačnih odgovora")
plt.title("Statistika")
plt.legend(loc="upper left")
plt.show()


data = {'Saradnici': 3.2, 'Polaznici': 3.33}
imena = list(data.keys())
vrednosti = list(data.values())

fig = plt.figure(figsize=(4, 5))

plt.bar(imena, vrednosti, color='maroon',
        width=0.6)
plt.plot(['Saradnici', 'Polaznici'], [3.12, 3.12], 'orange' ,lw=2, label = "NCD model nakon 2 epohe")
plt.plot(['Saradnici', 'Polaznici'], [3.69, 3.69], 'c' ,lw=2, label = "WReN model nakon 5 epoha")

plt.tight_layout(pad=1.08,)
plt.ylim([0, 10])
plt.xlabel("Ispitanici")
plt.ylabel("Broj tačnih odgovora")
plt.title("Statistika")
plt.legend(loc="upper left")
plt.show()

