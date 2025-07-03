import json
import numpy as np
import physics as p



def simulation(i_simulation):
    with open(f"meilleures_creatures_{i_simulation}.json", "r", encoding="utf-8") as f:
        creatures = json.load(f)

    n = len(creatures)  # Nombre de créatures
    # liste de la forme [ [0,[positions au cours du temps,score]],[1,[positions au cours du temps,score]],... ]
    results = [[i, p.calcul_position([np.array(element) for element in creature[1:]])[1:4:2]] for i, creature in enumerate(creatures)]
    results_sorted = sorted(results, key=lambda x: x[1][-1], reverse=True)  # Trier en fonction du score décroissant
    best_results = results_sorted[:n//2]  # Prendre les n/2 meilleurs résultats
    derniere_val = [liste[1][-1] for liste in results]  # Prendre la dernière valeur de chaque sous-liste
    print(best_results[0])
    """i_max = np.argmax(derniere_val)
    a = results[i_max][1][0].tolist()
    print(i_max)
    print(results[i_max][1][1])
    with open("creature_gagnante.json", "w", encoding="utf-8") as f:
        json.dump([int(i_max), a], f, indent=2)"""
    #print(best_results)
    with open(f"meilleures_creatures_{i_simulation+1}.json", "w", encoding="utf-8") as f:
        json.dump(best_results, f, indent=2)
    with open(f"meilleures_creatures_{i_simulation+1}.txt", "w", encoding="utf-8") as f:
        for i, liste in enumerate(best_results):
            f.write(f"Creature {liste[0]}\n")
            f.write(f"Score : {liste[1][1]}\n")
            f.write("\n")

simulation(0)