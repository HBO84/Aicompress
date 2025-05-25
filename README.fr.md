# AICompress : Compresseur de Fichiers Intelligent

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
**Langues :** [English](./README.en.md) | **Français**
AICompress est une application de compression de fichiers expérimentale qui utilise une approche basée sur l'intelligence artificielle (IA "Chef d'Orchestre") pour choisir la méthode de compression la plus adaptée à chaque type de fichier. Elle vise à offrir un bon équilibre entre taux de compression et vitesse, tout en explorant l'utilisation de techniques neuronales pour des cas spécifiques.

## Fonctionnalités Actuelles

* **Compression Intelligente :** Une IA "Chef d'Orchestre" (basée sur un modèle RandomForest) analyse les fichiers et choisit parmi plusieurs algorithmes de compression classiques.
* **Algorithmes Supportés par l'IA :**
    * STORED (aucune compression)
    * DEFLATE (niveaux 1, 6, 9 via `zlib`)
    * BZIP2 (niveau 9 via `bz2`)
    * LZMA (presets 0, 6, 9 via `lzma` et format XZ)
    * Zstandard (niveaux 1, 3, 9, 15 via `zstandard`)
    * Brotli (qualités 1, 6, 11 via `brotli`)
* **Moteur Autoencodeur (Expérimental) :** Un autoencodeur neuronal (Keras/TensorFlow) pour la compression avec perte d'images couleur de petite/moyenne taille (actuellement adapté pour du 32x32 pixels type CIFAR-10).
* **Format d'Archive Personnalisé `.aic` :** Basé sur le format ZIP standard, contenant les fichiers traités et un fichier de métadonnées `aicompress_metadata.json`.
* **Chiffrement :** Protection optionnelle des archives `.aic` par mot de passe utilisant AES-256 en mode GCM.
* **Décompression Multi-formats :**
    * Archives `.aic` créées par l'application.
    * Archives `.zip` standard.
    * Archives `.rar` (nécessite l'outil `unrar` externe).
    * Archives `.7z` (via la bibliothèque Python `py7zr`).
* **Interface Utilisateur Graphique (GUI) :** Développée avec Tkinter.
    * Ajout de fichiers et de dossiers.
    * Sélection du fichier de sortie et du dossier de destination.
    * Option de chiffrement avec mot de passe.
    * Barres de progression détaillées pour la compression et la décompression (par fichier/item pour `.aic`, `.zip`, `.rar`; indéterminée pour `.7z`).
    * Boutons "Annuler" pour les opérations longues.
    * Affichage des logs des opérations.
* **Mises à Jour OTA (Over-The-Air) :** Infrastructure de base pour la mise à jour des modèles d'IA (fonctionnalité à tester et finaliser).

## Statut du Projet

**Version Alpha.**
Ce projet est en développement actif et est principalement un terrain d'expérimentation et d'apprentissage. Des bugs peuvent être présents et des changements majeurs peuvent survenir.

## Installation

1.  **Prérequis :**
    * Python 3.10 ou supérieur recommandé.
    * `pip` (gestionnaire de paquets Python).
    * Pour la décompression des archives `.rar`, l'outil en ligne de commande `unrar` doit être installé sur votre système et accessible dans le PATH de votre système.
    * Sur certains systèmes Linux, pour la détection de type de fichier avec `python-magic`, l'installation de `libmagic1` (ou un paquet similaire) peut être nécessaire :
        ```bash
        sudo apt-get update && sudo apt-get install libmagic1
        ```

2.  **Cloner le Dépôt (si le dépôt est public) :**
    ```bash
    git clone [https://github.com/hbo84/AICompressProject.git](https://github.com/hbo84/AICompressProject.git)
    cd AICompressProject
    ```

3.  **Créer un Environnement Virtuel (Fortement Recommandé) :**
    ```bash
    python3 -m venv aic_env
    source aic_env/bin/activate  # Sur Linux/macOS
    # ou aic_env\Scripts\activate   # Sur Windows
    ```

4.  **Installer les Dépendances :**
    À la racine du projet, vous trouverez un fichier `requirements.txt`. Installez les dépendances avec :
    ```bash
    pip install -r requirements.txt
    ```

## Utilisation

1.  Activez votre environnement virtuel :
    ```bash
    source aic_env/bin/activate # ou équivalent Windows
    ```
2.  Lancez l'interface graphique depuis la racine du projet :
    ```bash
    python aicompress_gui.py
    ```
3.  Utilisez l'interface pour :
    * Ajouter des fichiers ou des dossiers à la liste de compression.
    * Spécifier le nom et l'emplacement de l'archive `.aic` de sortie.
    * Choisir d'activer le chiffrement par mot de passe.
    * Lancer la compression.
    * Sélectionner une archive (`.aic`, `.zip`, `.rar`, `.7z`) et un dossier de destination pour la décompression.

## Fonctionnement de l'IA (Aperçu)

* **Analyseur d'IA (`aicompress/ai_analyzer.py`) :** Détermine le type de fichier (texte, image, binaire, script Python, etc.) et extrait plusieurs caractéristiques numériques : taille, entropie normalisée, et un "ratio de compressibilité rapide".
* **Chef d'Orchestre (`aicompress/orchestrator.py`) :** Un modèle de Machine Learning (RandomForest) est entraîné sur un jeu de données pré-calculé. Il utilise les caractéristiques extraites par l'analyseur pour prédire la "meilleure" méthode de compression (et son niveau/preset) parmi celles disponibles (STORED, DEFLATE, BZIP2, LZMA, Zstandard, Brotli, Moteur AE).
* **Moteur Autoencodeur (`aicompress/ae_engine.py`) :** Un modèle de type autoencodeur convolutif (entraîné sur CIFAR-10) est utilisé pour la compression avec perte de petites images couleur. Le "Chef d'Orchestre" peut décider d'utiliser ce moteur pour les images compatibles.

## Feuille de Route / Fonctionnalités Futures (Exemples)

* Amélioration continue de la précision et de la pertinence du "Chef d'Orchestre" IA.
* Gestion récursive des archives en entrée (décompresser puis recompresser intelligemment).
* Améliorations GUI : redimensionnement de la fenêtre, glisser-déposer.
* Finalisation et tests des mises à jour OTA pour les modèles d'IA.
* Packaging de l'application pour une distribution plus facile (exécutables Windows/Linux).
* Exploration de nouveaux moteurs de compression neuronaux (avec ou sans perte).

## Contribuer

Ce projet est actuellement un développement personnel. Les suggestions et retours sont les bienvenus via les "Issues" GitHub.

## Licence

Ce projet est distribué sous les termes de la **licence MIT**.
Voir le fichier `LICENSE` à la racine du projet pour plus de détails.