import json
import numpy as np
import physics as p
import examples as e

N_SIMULATIONS = 10  # Nombre de simulations à effectuer

def simulation(i_simulation):
    with open(f"generations/meilleures_creatures_{i_simulation}.json", "r", encoding="utf-8") as f:
        creatures = json.load(f) # de la forme [ [0,position,matrice_adjacence,forces],[1,position,matrice_adjacence,forces],... ]
    
    n = len(creatures)  # Nombre de créatures
    
    results = [[i, p.calcul_position([np.array(element) for element in creature[1:]])[3]] for i, creature in enumerate(creatures)]
    # liste de la forme [ [0,score],[1,score],... ]
    results_sorted = sorted(results, key=lambda x: x[1], reverse=True)  # Trier en fonction du score décroissant
    best_results = results_sorted[:n//2]  # Prendre les n/2 meilleurs résultats
    
    with open(f"generations/meilleures_creatures_{i_simulation+1}.txt", "w", encoding="utf-8") as f:
        for i, liste in enumerate(best_results):
            f.write(f"Créature n° {liste[0]} :\n\n")
            f.write(f"Score : {liste[1]}\n\n")
            f.write(f"Positions des noeuds : {creatures[liste[0]][1]}\n\n")
            f.write(f"Matrice d'adjacence avec distances : {creatures[liste[0]][2]}\n\n")
            f.write(f"Forces par noeud en fonction du temps : {creatures[liste[0]][3]}\n\n")
            f.write("\n")
    
    creatures_nouvelles = []
    for i, liste in enumerate(best_results):
        pos = creatures[liste[0]][1]
        mat = creatures[liste[0]][2]
        forc = creatures[liste[0]][3]
        creatures_nouvelles.append([i, pos, mat, forc])
    creatures_nouvelles = mutation_toutes(creatures_nouvelles)  # Mutation des créatures sélectionnées

    with open(f"generations/meilleures_creatures_{i_simulation+1}.json", "w", encoding="utf-8") as f:
        json.dump(creatures_nouvelles, f, indent=2)

def simulation_multiple(n):
    for i in range(n):
        print(f"Simulation {i+1} en cours...")
        simulation(i)
    print("Simulations terminées.")

def mutation_toutes(creatures):
    n = len(creatures)
    i = 0
    creatures_nouvelles = creatures.copy()  # Copie de la liste des créatures
    for creature in creatures:
        #print(len(creature[1]))
        creature_modifiee = e.mutation_creature([np.array(creature[1]),np.array(creature[2]),np.array(creature[3])]) # On retire l'index de la créature
        creatures_nouvelles.append([n+i,creature_modifiee[0].tolist(),creature_modifiee[1].tolist(),creature_modifiee[2].tolist()])
        i += 1
    return creatures_nouvelles

simulation_multiple(N_SIMULATIONS)