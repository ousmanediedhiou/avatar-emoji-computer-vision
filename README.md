# Avatar Emoji — Vision numérique

Projet académique de vision par ordinateur réalisé à l’Université Laval.

## Présentation

Application Python utilisant une webcam pour détecter des visages et analyser certains mouvements et expressions du visage, afin d’animer un avatar emoji. Le projet utilise OpenCV et MediaPipe.

## Technologies

- Python
- OpenCV (détection DNN)
- MediaPipe
- NumPy

## Contenu du dépôt

- `avatar_mediapipe.py` : programme principal
- `deploy.prototxt` et `res10_300x300_ssd_iter_140000.caffemodel` : fichiers du détecteur DNN OpenCV
- `background.png` : ressource graphique
- `demo_original.txt` : notes d'installation d'origine (le nom du script mentionné dans ce fichier est ancien)

## Exécution

Prérequis : Python 3.10 et une webcam. Dans le dossier du projet :

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python avatar_mediapipe.py
```

> Les versions exactes des dépendances et le fonctionnement sur une autre machine n'ont pas été validés dans cette préparation du portfolio.

## Démonstration vidéo

Une vidéo de démonstration sera ajoutée dans les [Releases](../../releases) après création du dépôt. Elle peut montrer le visage de l'auteur : sa publication est volontaire.

## Contexte

Projet académique de vision numérique. Les fichiers de travail d'IDE et les enregistrements bruts ne font pas partie du code source distribué ici.
