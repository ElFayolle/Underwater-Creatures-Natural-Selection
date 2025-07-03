import json
import numpy as np
import main_test as m

with open("creatures.json", "r", encoding="utf-8") as f:
    creatures = json.load(f)

creatures_sans_index = [[np.array(element) for element in creature[1:]] for creature in creatures]  # Enlever l'index de chaque créature
results = [m.calcul_position(creature) for creature in creatures_sans_index]
results = [[i,results[i]] for i in range(len(results))]
print(results[0])
derniere_val = [liste[1][2] for liste in results]  # Prendre la dernière valeur de chaque sous-liste
print(derniere_val)
i_max = np.argmax(derniere_val)
a = results[i_max][1][1].tolist()
print(i_max)
print(results[i_max][1][2])
with open("creature_gagnante.json", "w", encoding="utf-8") as f:
    json.dump([int(i_max), a], f, indent=2)