import json
import numpy as np
import main as m

with open("creatures.json", "r", encoding="utf-8") as f:
    creatures = json.load(f)

creatures_sans_index = [[np.array(element) for element in creature[1:]] for creature in creatures]  # Enlever l'index de chaque créature
results = [m.calcul_position(creature) for creature in creatures_sans_index]
print(results)