# TP Deep Learning : MLP sur Bike Sharing

## Objet du TP

Je prédis `cnt`, le nombre de vélos loués, avec des perceptrons multi-couches (Keras / TensorFlow), à partir du calendrier et de la météo. La métrique de comparaison est la MAE (erreur absolue moyenne), en nombre de locations.

Le notebook suit trois actes :

1. **Jeu journalier** (`day.csv`) : une référence naïve et trois MLP, pour montrer le sur-apprentissage sur un petit jeu.
2. **Jeu horaire** (`hour.csv`) : tout le protocole. Je change un seul paramètre à la fois : variables d'entrée et encodage de l'heure, profondeur, largeur, fonction de coût, régularisation. Puis une recherche Keras Tuner (HyperBand), le test final et un essai de contrôle en découpage aléatoire.
3. **Comparaison** des deux modèles retenus sur le total journalier, sur les mêmes jours de test.

Le découpage est chronologique et identique pour les deux jeux : entraînement de janvier 2011 à juin 2012, validation de juillet à septembre 2012, test d'octobre à décembre 2012.

## Contenu du dépôt

| Fichier | Rôle |
| --- | --- |
| `exercice_mlp.ipynb` | Le notebook, exécuté de haut en bas, avec toutes les sorties enregistrées |
| `day.csv`, `hour.csv` | Le jeu Bike Sharing (UCI), versions journalière et horaire |
| `requirements.txt` | Les versions des bibliothèques utilisées |

L'exécution crée deux dossiers, ignorés par git : `logs/` (journaux TensorBoard) et `kt_dir/` (essais Keras Tuner).

## Lancer le notebook en local

Avec Python 3.13 (version utilisée pour l'exécution enregistrée) :

```bash
python -m venv .venv
```

```bash
.venv/Scripts/activate
```

Sous macOS ou Linux, la commande d'activation est `source .venv/bin/activate`.

```bash
pip install -r requirements.txt jupyter
```

```bash
jupyter notebook exercice_mlp.ipynb
```

Puis « Exécuter tout ». Le notebook complet prend environ 10 minutes sur un processeur, sans GPU.

## Lancer le notebook sur Google Colab

1. Ouvrir [colab.research.google.com](https://colab.research.google.com), puis « Fichier > Importer le notebook » et choisir `exercice_mlp.ipynb` (onglet GitHub ou import du fichier).
2. « Exécution > Tout exécuter ».

Rien à installer à la main : si `day.csv` et `hour.csv` ne sont pas présents, le notebook les télécharge depuis l'UCI, et il installe Keras Tuner s'il manque. Les chiffres peuvent varier légèrement par rapport à l'exécution enregistrée (autres versions, autre matériel).

## Ouvrir TensorBoard

Chaque essai écrit ses courbes dans `logs/<identifiant de l'essai>`.

- **Sur Colab** : l'avant-dernière cellule de code ouvre TensorBoard directement dans le notebook.
- **En local** : après l'exécution, dans le dossier du projet :

```bash
tensorboard --logdir logs
```

Puis ouvrir http://localhost:6006. Les courbes `mae_cnt` et `val_mae_cnt` sont les MAE en nombre de locations.

## Résultat principal

Modèle horaire retenu : **R_ES**, un MLP de deux couches cachées de 64 neurones (8 001 paramètres), avec l'heure en one-hot, la perte MAE et l'arrêt précoce.

| Modèle | MAE de test |
| --- | --- |
| **MLP retenu (R_ES)** | **45,1 locations par heure** |
| Référence : moyenne par heure | 99,9 locations par heure |
| Référence : médiane constante | 161,1 locations par heure |

Sur le total journalier des 92 jours de test, le modèle horaire sommé par jour (670 locations par jour d'erreur) fait mieux que le modèle journalier (803).
