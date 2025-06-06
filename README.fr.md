# AICompress : Compresseur de Fichiers Intelligent v1.3.0

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Langues :** [English](./README.md) | **Français**
AICompress est un projet open source expérimental qui utilise une IA "Chef d'Orchestre" pour sélectionner intelligemment la méthode de compression la plus adaptée à chaque fichier. Il vise à offrir un équilibre optimal entre taux de compression et vitesse en s'appuyant sur une gamme d'algorithmes classiques. Une fonctionnalité clé est sa capacité à optimiser récursivement les fichiers contenus dans des archives existantes.

## Fonctionnalités Principales

* **Sélection Intelligente de Compression :** Une IA "Chef d'Orchestre" (modèle RandomForest) analyse les fichiers (type, taille, entropie, ratio de compressibilité rapide) pour choisir le meilleur algorithme et ses paramètres.
* **Palette d'Algorithmes Complets :**
    * STORED (aucune compression)
    * DEFLATE (niveaux 1, 6, 9 via `zlib`)
    * BZIP2 (niveau 9 via `bz2`)
    * LZMA (presets 0, 6, 9 via `lzma` et format XZ)
    * Zstandard (niveaux 1, 3, 9, 15 via `zstandard`)
    * Brotli (qualités 1, 6, 11 via `brotli`)
* **Optimisation Récursive des Archives :** Lors de la compression, si l'option est activée, AICompress peut détecter les archives (ZIP, RAR, 7z, AIC) dans votre sélection, les extraire, et recompresser intelligemment leur contenu individuellement dans l'archive `.aic` finale.
* **Format d'Archive Personnalisé (`.aic`) :** Basé sur le format ZIP, avec un fichier de métadonnées `aicompress_metadata.json` détaillant les opérations.
* **Chiffrement :** Protection optionnelle des archives `.aic` par mot de passe (AES-256 GCM).
* **Décompression Multi-formats :**
    * Archives `.aic` créées par l'application.
    * Archives `.zip` standard.
    * Archives `.rar` (nécessite l'outil `unrar` externe).
    * Archives `.7z` (via la bibliothèque Python `py7zr`).
* **Interface Utilisateur Graphique (GUI) :** Développée avec Tkinter, incluant :
    * Ajout de fichiers/dossiers.
    * Sélection des fichiers de sortie/destination.
    * Option de chiffrement.
    * Barres de progression (détaillée pour compression, .aic, .zip, .rar ; indéterminée pour .7z en décompression).
    * Boutons "Annuler".
    * Affichage des logs.
* **Mises à Jour OTA (Over-The-Air) des Modèles IA :** Infrastructure de base présente (à finaliser et tester).

## Statut du Projet

**Version 1.3.0 - Alpha.**
Projet en développement actif, principalement à des fins d'expérimentation et d'apprentissage. Des bugs sont possibles.

## Installation

1.  **Prérequis :**
    * Python 3.10 - 3.12 recommandé (développé et testé avec 3.12).
    * `pip` (gestionnaire de paquets Python).
    * Pour la décompression `.rar`, l'outil `unrar` doit être installé et dans le PATH.
    * Sous Linux, pour `python-magic`, `libmagic1` peut être requis (`sudo apt-get install libmagic1`).
    * Sous Windows (si exécution depuis les sources), `python-magic-bin` est recommandé.

2.  **Cloner le Dépôt (si public) :**
    ```bash
    git clone [https://github.com/VOTRE_NOM_UTILISATEUR/AICompressProject.git](https://github.com/VOTRE_NOM_UTILISATEUR/AICompressProject.git)
    cd AICompressProject
    ```

3.  **Créer un Environnement Virtuel (Fortement Recommandé) :**
    ```bash
    python3 -m venv aic_env
    source aic_env/bin/activate  # Linux/macOS
    # ou aic_env\Scripts\activate   # Windows
    ```

4.  **Installer les Dépendances :**
    ```bash
    pip install -r requirements.txt
    ```

## Utilisation

1.  Activez votre environnement virtuel.
2.  Lancez la GUI depuis la racine du projet :
    ```bash
    python aicompress_gui.py
    ```
3.  Utilisez l'interface pour ajouter des fichiers/dossiers, choisir les options (notamment "Optimiser les archives incluses" pour la nouvelle fonctionnalité), et lancer les opérations.

## Fonctionnement de l'IA (Aperçu)

* **Analyseur d'IA (`aicompress/ai_analyzer.py`) :** Détermine le type de fichier et extrait des caractéristiques (taille, entropie, ratio de compressibilité rapide).
* **Chef d'Orchestre (`aicompress/orchestrator.py`) :** Un modèle RandomForest utilise ces caractéristiques pour prédire la meilleure méthode de compression classique (STORED, DEFLATE, BZIP2, LZMA, Zstandard, Brotli) et son niveau/qualité.

## Feuille de Route / Fonctionnalités Futures

* Amélioration continue de l'IA "Chef d'Orchestre".
* Améliorations GUI : redimensionnement stable de la fenêtre, glisser-déposer.
* Affinement de la barre de progression pour les opérations récursives.
* Finalisation des mises à jour OTA.
* Packaging pour une distribution facilitée (exécutables).

## Contribuer


Ce projet est un développement personnel. Suggestions et retours bienvenus via les "Issues" GitHub.

## Licence

Ce projet est distribué sous les termes de la **licence MIT**.
Voir le fichier `LICENSE` pour plus de détails.