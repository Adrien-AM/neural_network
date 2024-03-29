# Réseau de neurones

## Description

Implémentation en C et C++ d'un réseau de neurones.
L'objectif à terme est d'avoir un programme relativement optimisé et parallélisé pour faire des calculs sur des données importantes comme des images.

### Utilisation

  - ``make`` pour compiler tous les exécutables
  - ``make clean`` pour nettoyer le build
  - ``make test`` pour compiler et lancer le(s) test(s)
  - ``make <exe>`` pour compiler l'exécutable <exe>

Il y a 3 exécutables actuellement :
- main : simple régression sur une fonction arbitraire
- house : régression sur un dataset de prix de maisons (il faut avoir le dataset)
- mnist : OCR sur le dataset [MNIST](http://yann.lecun.com/exdb/mnist/). (Attention il faut décompresser les fichiers et les renommer correctement).

## Version C++

Couches disponibles :
- Dense : Couche de base, complètement connectée
- Input : Entrée
- Reshape : Changement de dimensions
- Flatten : X dimensions -> Vecteur
- Dropout : Désactivation aléatoire de certains neurones. - [Article](https://arxiv.org/abs/1207.0580)
- Convolutions - WIP

Activations disponibles :
- Linear
- Sigmoid
- ReLU
- Softmax

Fonctions de coût disponibles :
- Mean Absolute Error
- Mean Squared Error
- Categorical Cross Entropy
- Structural Similarity
D'autres fonctions peuvent être écrite et utilisée grâce au principe d'auto-différentiation, en utilisant l'API CompGraph.

Optimizers disponibles :
- SGD
- Adam
La taille des *batchs* et tous les hyperparamètres sont paramétrable.
La technique du *gradient clipping* peut aussi être appliquée pour éviter l'explosion du gradient.

Métriques disponibles :
- Précision

Toutes les données sont manipulées avec la structure de données personnalisée Tensor<T>.

## Version C

### État actuel

Réseau de neurones à couches denses uniquement.

Le réseau prend en entrée un tableau de vecteurs de flottants et renvoie un vecteur de flottants. \
Chaque neurone a un poids et un biais. \
L'apprentissage utilise par défaut l'algorithme du momentum pour le calcul du gradient. \
Les poids et biais initiaux des neurones sont initialisés selon une distribution normale de moyenne et écart-type paramétrables. \
Pour éviter les problèmes d'explosion du gradient, il est possible de le limiter lors de l'entraînement avec la technique du *gradient clipping*. 

Les fonctions d'activation disponibles sont :
  - Linear
  - ReLU
  - Sigmoid

Les fonctions de coût disponibles sont :
  - Mean Absolute Error (MAE)
  - Mean Squared Error (MSE)
  - Categorical Cross-Entropy

Une couche Softmax est aussi possible pour les problèmes de classification.

Les interfaces ``neural_network.h`` et ``layer.h`` documentent la façon d'utiliser un réseau de neurones.

!! L'entraînement par batchs n'est pas encore implémenté.
