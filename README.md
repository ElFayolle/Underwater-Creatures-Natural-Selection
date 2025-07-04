# Underwater-Creatures-Natural-Selection
We will study the natural selection of underwater creatures #UnderWaterSophia2k26



L'objectif de ce projet est la simulation de l'évolution de créatures dans un milieu aquatique.



Les créatures sont modélisées selon des noeuds reliés entre eux par des segments faisant office de muscles.
Chaque noeud subit un cycle de forces aléatoires, ce qui imprime un mouvement global à la créature en conjuguant les déplacements de chaque noeud.
Les frottements avec l'eau permettent aux créatures de se déplacer dans l'eau.

STRUCTURE DU PROJET :


Nous commençons par générer une première tribu de créatures en positionnant les noeuds et les segments de manière aléatoire, et en attribuant à chaque noeud un cycle de forces aléatoires sous la forme de gaussiennes [generation.py]. Les créatures générées et leurs attributs sont enregistrés dans le dossier generations dans des fichiers textes mais aussi dans des fichiers json (le texte permettant aux utilisateurs de "voir" leurs créatures, le json étant réutilisé pour les algorithmes).

[physics.py] contient l'ensemble des lois physiques régissant le milieu aquatique dans lequel évoluent les créatures, ainsi que les fonctions de calcul des forces et des déplacements pour chaque noeud, mais aussi celles pour projeter les forces orthogonalement aux segments (les mouvements correspondant donc à des rotations des "bras")

[simulation.py] permet de simuler une ou toutes les créatures générées. On les place dans l'eau pendant une certaine durée fixée (10 secondes par exemple) et on mesure la distance parcourue et l'énergie dépensée (en sommant les énergies cinétiques de chaque noeud). Ceci permet de définir un score pour chaque créature, en donnant plus de valeur aux créatures se déplaçant loin en dépensant un minimum d'énergie.

[main.py] contient la fenêtre pygames qui permet de visualiser les créatures simulées et leur mouvement dans l'eau. On trace notamment les vecteurs force qui s'appliquent sur les noeuds et les segments mais aussi le centre de masse et sa trajectoire. S'il y a plusieurs créatures simultanément dans l'eau, on peut passer de l'une à l'autre.

[params.py] permet de stocker certaines paramètres globaux du projet

[fond.jpg] est l'image utilisée dans la fenêtre pygames por le fond bleu de l'océan
