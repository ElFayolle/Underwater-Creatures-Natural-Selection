# Underwater-Creatures-Natural-Selection
We will study the natural selection of underwater creatures #UnderWaterSophia2k26

Les créatures sont modélisées selon des noeuds reliés entre eux par des segments.
Les segments sont soumis à une force élastique obéissant à la loi de Hooke.
Chaque noeud subit une force aléatoire de manière cyclique, ce qui imprime un mouvement à la créature.
Les frottements avec l'eau permettent aux créatures de se mettre en mouvement.

Le fichier main.py contient la fenêtre Pygames permettant de visualiser les créatures dans l'eau
Les calculs des différentes forces sur chaque partie des créatures sont également contenus dans ce fichier : forces de rappel entre les noeuds selon l'axe des segments, force de frottement avec l'eau, et force exercée par la créature elle-même sur ses différents noeuds.
Les lois de Newton et le théorème de la résultante permettent ensuite de calculer le mouvement de la créature dans l'eau.


L'algorithme de examples.py permet de générer des créatures aléatoires selon plusieurs paramètres tels que le nombre de noeuds ou la longueur des segments
On génère aussi les cycles aléatoires des forces qui sont appliquées sur chaque noeud.
Les données sur les positions d'origine des créatures et les forces des noeuds sont sauvegardées dans un fichier texte creatures_texte.txt

fond.jpg : TEDLT

gen_creatures.py et fonctions.py sont inutiles mais ont servi à garder un arbre de commits plus propre.

Dans un futur proche, nous souhaitons pouvoir simuler les déplacements de plusieurs créatures afin de les comparer entre elles. Plusieurs critères sont étudiés : l'énergie dépensée par la créature (énergie cinétique de chaque noeud), la distance parcourue, la vitesse moyenne du centre de masse.
L'objectif est de modéliser plusieurs générations de créatures en gardant à chaque étape les créatures les plus performantes pour se rapprocher d'un processus de sélection naturelle. Quelles créatures seront les plus efficaces ?
