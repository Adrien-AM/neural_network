# Réseau de neurones

## Description

Implémentation en C d'un réseau de neurones dense.
L'objectif à terme est d'avoir un programme relativement optimisé et parallélisé pour faire des calculs sur des données importantes comme des images.

## État actuel

Réseau de neurones à couches dense uniquement.
Le réseau prend en entrée un tableau de vecteurs de flottants et renvoie un vecteur de flottants.
Chaque neurone a un poids et un biais.
L'apprentissage utilise par défaut l'algorithme du momentum pour l'apprentissage.

Les fonctions d'activation disponibles sont :
  - Linear
  - ReLU
  - Sigmoid

## Utilisation

  - ``make`` pour compiler les sources
  - ``make clean`` pour nettoyer le build
  - ``./nn`` pour lancer l'exécution
