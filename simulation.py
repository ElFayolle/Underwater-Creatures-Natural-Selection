import json
import numpy as np
import main as m

with open("creatures.json", "r", encoding="utf-8") as f:
    creatures = json.load(f)

creatures_sans_index = [liste[1:] for liste in creatures]  # Enlever l'index de chaque créature
c = np.array(creatures_sans_index, dtype=object)  # Convertir en tableau numpy d'objets
d = m.calcul_position(c)
print(d)