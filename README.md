# Le Snake

Vous trouverez ci-dessous les instructions et détails sur le jeu du Snake.
Le but du jeu étant de dévorer le plus de fruits possibles et de faire ainsi grandir
son serpent en le controllant de manière à éviter de se prendre dans sa propre queue. 

Le jeu est présenté ici avec deux techniques d'IA, une recherche par A\*, où l'on essaie
de trouver le plus court chemin jusqu'au prochain fruit, ainsi qu'un réseau de neurones 
artificiels entrainé grâce à un algorithme génétique. 


## Installation

Pour installer le jeu, commencez par copier le dépot du livre ([AI-book sur github][ia-gh]),
soit en recupérant l'archive zip depuis github, soit à l'aide de l'outil git:
```
git clone https://github.com/iridia-ulb/AI-book
```

Puis, accedez au dossier du jeu:

```bash
cd Snake
```

Après avoir installé python et poetry, rendez vous dans ce dossier et installez les
dépendances du projet:

```bash
poetry install
```

## Utilisation

Vous pouvez ensuite lancer le jeu dans l'environnement virtuel nouvellement
crée, en utilsant la commande:

```bash
poetry run python main.py -p
```
Cette commande lance le jeu en mode "player" ce qui vous permet de jouer au
snake; il suffit alors d'appuyer sur la barre espace et d'utiliser les touches
directionelles du clavier.

Pour faire jouer une IA, par exemple l'IA de recherche A\*, il suffit de
lancerle jeu comme ceci:

```bash
poetry run python main.py --ai -a
```
l'option `--ai` indique au jeu de se lancer en mode IA, ensuite la 2ème option
indique le type d'IA, cela peut être `-a` pour A\*, `-s` pour le chemin en forme
de S, `-g` pour le réseau de neurone entrainé pour algorithme génétique, `-rl` pour
le réseau de neurone entrainé par reinforcement learning.

Ces dernières options (`-g` et `-rl`) requièrent l'ajout d'un argument à la commande pour indiquer
le modèle de réseau de neurone à utiliser, par exemple:

```bash
poetry run python main.py --ai -g weights/159.snake 
```
ou 
```bash
poetry run python main.py --ai -rl try10/weights_82_1_1600.h5
```
Quelques exemples de réseau de neurones pré-entrainés sont disponibles dans 
le dossier `weights`.

En résumé:
```
usage: main.py [-h] [-p | -x] [-g GENETIC | -s | -a]

Snake game.

optional arguments:
  -h, --help            show this help message and exit
  -p, --player          Player mode: the player controls the game
  -x, --ai              AI mode: the AI controls the game (requires an 'algorithm' argument)
  -g GENETIC, --genetic GENETIC
                        Genetic algorithm: plays a move based of trained neural network, please select weight file
  -s, --sshaped         S-Shaped algorithm: browses the whole grid each time in an 'S' shape. Only works if height of grid is even.
  -a, --astar           A* algorithm: classical A* algorithm, with Manhattan distance as heuristic

```

### Entrainement

Pour entrainer un nouveau réseau de neurone pour le snake, il faut lancer le
programme `train.py`, par exemple:

```bash
poetry run python train.py 
```
Les meilleurs réseau de neurones seront stockés par score dans le dossier
`weights`, ainsi, par exemple, le fichier `159.snake` contient un modèle
qui a reussit à atteindre un score de 159.

Il est aussi possible de changer certainss hyperparamètres de l'algorithme
génétique.
L'option `-p` permet de fixer le nombre de snake dans la population initiale
(par défaut à 1000).
L'option `-m` permet de fixer le taux de mutation des génomes lors d'un
changement de génération (par défaut à 0.01).
l'option `-e` permet de fixer le taux d'élitisme de l'algorithme (taux de snake
conversés entre les générations) (par défaut à 0.12).

En résumé:
```
usage: train.py [-h] [-p POPULATION] [-m MUTATION] [-e ELITISM]

Snake game, training program for neural net.

optional arguments:
  -h, --help            show this help message and exit
  -p POPULATION, --population POPULATION
                        Defines the size of the initial population (must be >20), default=1000
  -m MUTATION, --mutation MUTATION
                        Defines the mutation rate (0 < m < 1) (float), default=0.01
  -e ELITISM, --elitism ELITISM
                        Define the portion of snakes that are passed to next generation through elitism (0 < e < 1) (float), default=0.12

```

### Reinforcement Learning
Pour entainer un nouvel agent via reinforcement learning pour Snake, il faut lancer le program `train_rl.py`, par exemple:
```bash
poetry run python train_rl.py
```
L'évolution du réseau de neurones sera stocké par record dans le dossier weights_rl, ainsi par exemple,
le fichier weights_44_0.h5 correspond à un modèle qui a réussit à atteindre un record de 44.

Il est aussi possible de changer certains hyperparamètres de l'entrainement de l'agent. L'option `-e`
permet de changer le nombres d'épisodes réalisées durant l'entrainement, 2000 étant la valeur par défaut. L'option `-w` accompagnée 
d'un nom controlera le *coeur* du nom du fichier dans lequel les poids synaptiques seront enregistrés (le *weights* de *weights_44_0.h5*).
L'option `-rt` permet de relancer l'entrainement à partir d'un fichier existant avec des paramètres différents par défaut
le l'exploration sera fixée à 0 (*epsilon*) dans ce cas il faut également préciser le chemin vers le fichier de poids de départ.

Par exemple:
```bash
poetry run python train_rl.py -e 2000 -w weights -rt try10/weights_82_1_1600.h5
```

En résumé:
```
usage: train.py [-h] [-e EPISODE] [-w FILENAME] [-rt INITWEIGHTS]

Snake game, training program for neural net using reinforcement learning

optional arguments:
    -h, --help          show this help message and exit
    -e EPISODE, --episodes EPISODE
                        defines the number of episodes on which the training will be done, default 2000
    -w FILENAME, --weights FILENAME
                        define the base name for the file in which the neuronal weights will be stored periodically
    -rt INITWEIGHTS, --retrain INITWEIGHTS
                        define the path to the file used to initialise the neural network weights for retraining
```

![snake screen](../assets/img/snake.png)



[ia-gh]: https://github.com/iridia-ulb/AI-book
