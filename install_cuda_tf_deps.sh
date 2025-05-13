#!/bin/bash

# Script pour installer CUDA Toolkit 12.5 et cuDNN 9.3 pour TensorFlow 2.19 sur Ubuntu
# VERSION MODIFIÉE POUR UTILISER UN FICHIER cuda-keyring LOCAL TÉLÉCHARGÉ MANUELLEMENT.

echo "------------------------------------------------------------------------------------"
echo "Installation des dépendances CUDA Toolkit 12.5 et cuDNN 9.3 pour TensorFlow"
echo "UTILISATION D'UN FICHIER cuda-keyring LOCAL."
echo "ATTENTION : Ce script nécessite sudo et modifiera votre système."
echo "Une connexion internet est requise pour les téléchargements via apt."
echo "------------------------------------------------------------------------------------"
echo ""
read -p "Assurez-vous que 'cuda-keyring_1.1-1_all.deb' est dans ce dossier. Appuyez sur Entrée pour continuer, ou Ctrl+C pour annuler..."

# Arrêter le script en cas d'erreur
set -e

# Fonction pour afficher les en-têtes de section
echo_section_header() {
    echo ""
    echo "--------------------------------------------------"
    echo "$1"
    echo "--------------------------------------------------"
}

KEYRING_FILE_LOCAL="cuda-keyring_1.1-1_all.deb" # Nom standard du fichier attendu

# --- Étape 1: Prérequis et configuration du dépôt NVIDIA avec le keyring local ---
echo_section_header "Étape 1: Installation des prérequis et configuration du dépôt NVIDIA (via keyring local)"

sudo apt update
sudo apt install -y software-properties-common gpg # wget n'est plus nécessaire pour le keyring

echo "Configuration du dépôt NVIDIA CUDA avec le fichier keyring local..."
# Nettoyer les anciennes configurations de dépôt cuda au cas où
sudo rm -f /etc/apt/sources.list.d/cuda*.list /etc/apt/trusted.gpg.d/cuda*.gpg

if [ -f "$KEYRING_FILE_LOCAL" ]; then
    echo "Utilisation du fichier keyring local : $KEYRING_FILE_LOCAL"
    sudo dpkg -i "$KEYRING_FILE_LOCAL"
    sudo apt update # Mettre à jour la liste des paquets après avoir ajouté le dépôt
    echo "Dépôt NVIDIA configuré avec succès via le keyring local."
else
    echo "ERREUR : Le fichier '$KEYRING_FILE_LOCAL' n'a pas été trouvé dans le dossier actuel ($(pwd))."
    echo "Veuillez télécharger le fichier .deb du keyring NVIDIA (généralement pour Ubuntu 22.04 s'il n'y a pas pour 24.04),"
    echo "placez-le ici, assurez-vous qu'il est nommé '$KEYRING_FILE_LOCAL', puis relancez le script."
    exit 1
fi

# --- Étape 2: Installation du CUDA Toolkit 12.5 ---
echo_section_header "Étape 2: Installation du CUDA Toolkit 12.5"
echo "Installation de cuda-toolkit-12-5... Cela peut prendre du temps et télécharger beaucoup de données."
# Ce paquet devrait maintenant être disponible grâce au dépôt configuré.
sudo apt install -y cuda-toolkit-12-5
echo "CUDA Toolkit 12.5 devrait être installé."

# --- Étape 3: Installation de cuDNN 9.3 pour CUDA 12.5 ---
echo_section_header "Étape 3: Installation de cuDNN 9.3 pour CUDA 12.5"
echo "Installation de libcudnn9-dev (pour CUDA 12.x)..."
# Le paquet libcudnn9-dev devrait tirer libcudnn9 comme dépendance.
# Le dépôt NVIDIA devrait fournir une version compatible avec CUDA 12.5 (cuDNN 9.3 ou proche).
sudo apt install -y libcudnn9-dev
echo "cuDNN 9.x (espérons 9.3 compatible avec CUDA 12.5) devrait être installé."

# --- Étape 4: Configuration des variables d'environnement ---
echo_section_header "Étape 4: Configuration des variables d'environnement pour CUDA 12.5"
BASHRC_FILE="$HOME/.bashrc"

echo "Configuration de CUDA_HOME, PATH, et LD_LIBRARY_PATH pour CUDA 12.5 dans $BASHRC_FILE..."

# Vérifier si les lignes existent déjà pour éviter les doublons (méthode simple)
if ! grep -q "CUDA Toolkit 12.5 Configuration for AICompress" "$BASHRC_FILE"; then
    {
        echo ""
        echo "# CUDA Toolkit 12.5 Configuration for AICompress"
        echo "export CUDA_HOME=/usr/local/cuda-12.5"
        echo 'export PATH=$CUDA_HOME/bin:$PATH'
        echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH'
        echo 'export LD_LIBRARY_PATH=$CUDA_HOME/extras/CUPTI/lib64:$LD_LIBRARY_PATH' # Pour CUPTI
    } >> "$BASHRC_FILE"
    echo "Variables d'environnement ajoutées à votre $BASHRC_FILE."
else
    echo "Les variables d'environnement pour CUDA 12.5 semblent déjà exister dans $BASHRC_FILE."
fi

echo "Vous devrez exécuter 'source $BASHRC_FILE' ou ouvrir un nouveau terminal pour qu'elles prennent effet."
echo ""
echo "Pour que les variables soient disponibles immédiatement dans CE SCRIPT pour vérification (nvcc):"
export CUDA_HOME=/usr/local/cuda-12.5
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CUDA_HOME/extras/CUPTI/lib64:$LD_LIBRARY_PATH

# --- Étape 5: Vérification ---
echo_section_header "Étape 5: Vérification de l'installation"

echo "Vérification avec nvcc --version (devrait afficher 12.5) :"
if command -v nvcc &> /dev/null
then
    nvcc --version
else
    echo "nvcc non trouvé. L'installation de CUDA Toolkit a peut-être échoué ou PATH n'est pas encore à jour."
    echo "Essayez d'ouvrir un nouveau terminal après ce script."
fi
echo ""

echo "Vérification avec nvidia-smi (votre pilote devrait toujours être le même) :"
if command -v nvidia-smi &> /dev/null
then
    nvidia-smi
else
    echo "nvidia-smi non trouvé. Problème avec les pilotes NVIDIA ou PATH."
fi
echo ""

echo "Vérification de la présence de libdevice (important pour TensorFlow) :"
LIBDEVICE_PATH_PATTERN="/usr/local/cuda-12.5/nvvm/libdevice/libdevice*.10.bc"
# Utiliser find pour gérer les variations mineures de nom de fichier
FOUND_LIBDEVICE=$(find /usr/local/cuda-12.5/nvvm/libdevice/ -name 'libdevice*.10.bc' 2>/dev/null | head -n 1)

if [ -n "$FOUND_LIBDEVICE" ] && [ -f "$FOUND_LIBDEVICE" ]; then
    echo "libdevice trouvé à : $FOUND_LIBDEVICE"
else
    echo "ATTENTION : libdevice NON TROUVÉ au chemin attendu ($LIBDEVICE_PATH_PATTERN)."
    echo "Cela pourrait causer des problèmes avec TensorFlow. Vérifiez votre installation CUDA."
    echo "Vous pouvez chercher manuellement avec: sudo find /usr/local/cuda-12.5 -name 'libdevice*.10.bc'"
fi
echo ""

echo "Vérification de la présence de ptxas (important pour TensorFlow) :"
if command -v ptxas &> /dev/null
then
    echo "ptxas trouvé ici :"
    which ptxas
    ptxas --version || echo "ptxas --version a échoué, mais ptxas est dans le PATH."
else
    echo "ATTENTION : ptxas NON TROUVÉ dans le PATH."
    echo "Cela causera des problèmes avec TensorFlow. Vérifiez votre installation CUDA et le PATH."
fi
echo ""

echo_section_header "Installation terminée (ou du moins tentée !)."
echo "------------------------------------------------------------------------------------"
echo "ACTIONS IMPORTANTES POST-SCRIPT :"
echo "1. Ouvrez un NOUVEAU terminal pour que les variables d'environnement soient chargées."
echo "   (Ou exécutez : source $BASHRC_FILE)"
echo "2. Dans le nouveau terminal, vérifiez à nouveau :"
echo "   nvcc --version   (doit afficher 12.5.x)"
echo "   echo \$PATH       (doit inclure /usr/local/cuda-12.5/bin)"
echo "   echo \$LD_LIBRARY_PATH (doit inclure /usr/local/cuda-12.5/lib64)"
echo "3. Activez votre environnement virtuel Python pour AICompress ('aic_env')."
echo "4. Si TensorFlow était déjà installé, il est fortement recommandé de le réinstaller pour qu'il se lie correctement aux nouvelles bibliothèques CUDA :"
echo "   pip uninstall tensorflow  # (Ignorez si pas déjà installé dans le venv)"
echo "   pip install \"tensorflow[and-cuda]==2.19.0\"  # Tente d'installer les composants CUDA liés à TF"
echo "   # OU alternativement: pip install tensorflow==2.19.0"
echo "5. Testez TensorFlow avec GPU en Python :"
echo "   python -c \"import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))\""
echo "   Cela devrait lister votre GPU s'il est correctement configuré."
echo "6. Si vous rencontrez toujours des problèmes, un redémarrage complet du système est parfois nécessaire après l'installation de CUDA."
echo "------------------------------------------------------------------------------------"

exit 0