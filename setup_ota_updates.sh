#!/bin/bash

echo "--------------------------------------------------------------------------"
echo "Script de configuration pour les Mises à Jour OTA des Modèles IA"
echo "Ce script va installer 'requests', mettre à jour 'ai_analyzer.py',"
echo "créer 'ota_updater.py', et mettre à jour 'aicompress_gui.py'."
echo ""
echo "ATTENTION : Une sauvegarde de votre travail est fortement recommandée."
echo "Assurez-vous que votre environnement virtuel Python est activé."
echo "--------------------------------------------------------------------------"
echo ""
read -p "Appuyez sur Entrée pour continuer, ou Ctrl+C pour annuler MAINTENANT..."

# Vérifier si nous sommes (probablement) dans AICompressProject
if [ ! -d "aicompress" ] || [ ! -f "main.py" ]; then
    echo "ERREUR : Ce script doit être exécuté depuis la racine de AICompressProject."
    exit 1
fi

# Fonction pour afficher les en-têtes de section
echo_section_header() {
    echo ""
    echo "--------------------------------------------------"
    echo "$1"
    echo "--------------------------------------------------"
}

# Étape 1: Installation de la dépendance 'requests'
echo_section_header "Étape 1: Installation de 'requests'"
if pip install requests; then
    echo "'requests' installé/mis à jour avec succès."
else
    echo "ERREUR : L'installation de 'requests' a échoué."
    echo "Veuillez l'installer manuellement : pip install requests"
    exit 1
fi

# Étape 2: Mise à jour de aicompress/ai_analyzer.py
echo_section_header "Étape 2: Mise à jour de aicompress/ai_analyzer.py"
cat << 'EOF' > aicompress/ai_analyzer.py
# aicompress/ai_analyzer.py
import os
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import numpy as np
import joblib

try:
    import magic
    PYTHON_MAGIC_AVAILABLE = True
except ImportError:
    PYTHON_MAGIC_AVAILABLE = False
    print("AVERTISSEMENT (ai_analyzer.py): 'python-magic' non trouvé. Identification binaire limitée.")

MODEL_BASE_DIR = os.path.join(os.path.expanduser("~"), ".aicompress")
MODELS_SPECIFIC_DIR = os.path.join(MODEL_BASE_DIR, "models")
TEXT_CLASSIFIER_MODEL_NAME = "text_classifier.joblib"
TEXT_CLASSIFIER_CONFIG_NAME = "text_classifier_config.json"

TEXT_CLASSIFIER_MODEL_PATH = os.path.join(MODELS_SPECIFIC_DIR, TEXT_CLASSIFIER_MODEL_NAME)
TEXT_CLASSIFIER_CONFIG_PATH = os.path.join(MODELS_SPECIFIC_DIR, TEXT_CLASSIFIER_CONFIG_NAME)

DEFAULT_TEXT_CLASSIFIER_VERSION = "1.0" 

text_classifier_model = None
current_text_classifier_version = DEFAULT_TEXT_CLASSIFIER_VERSION # Initialisé à défaut

def ensure_model_dir_exists():
    os.makedirs(MODELS_SPECIFIC_DIR, exist_ok=True)

def get_local_text_classifier_version():
    global current_text_classifier_version
    ensure_model_dir_exists()
    if os.path.exists(TEXT_CLASSIFIER_CONFIG_PATH):
        try:
            with open(TEXT_CLASSIFIER_CONFIG_PATH, 'r') as f:
                config = json.load(f)
                current_text_classifier_version = config.get("version", DEFAULT_TEXT_CLASSIFIER_VERSION)
        except Exception as e:
            print(f"[AI_ANALYZER] Erreur lecture config version: {e}. Utilisation défaut {DEFAULT_TEXT_CLASSIFIER_VERSION}.")
            current_text_classifier_version = DEFAULT_TEXT_CLASSIFIER_VERSION
    else: # Pas de fichier config, on est à la version par défaut (ou le modèle n'existe pas encore)
        current_text_classifier_version = DEFAULT_TEXT_CLASSIFIER_VERSION
    return current_text_classifier_version

def save_local_text_classifier_version(version_str):
    global current_text_classifier_version
    ensure_model_dir_exists()
    try:
        with open(TEXT_CLASSIFIER_CONFIG_PATH, 'w') as f:
            json.dump({"version": version_str}, f)
        current_text_classifier_version = version_str
        print(f"[AI_ANALYZER] Version modèle texte local MàJ: {version_str}")
    except Exception as e:
        print(f"[AI_ANALYZER] Erreur sauvegarde config version: {e}")

def train_and_save_text_classifier(version_to_save=DEFAULT_TEXT_CLASSIFIER_VERSION):
    global text_classifier_model
    ensure_model_dir_exists()
    print(f"[AI_ANALYZER] Entraînement nouveau modèle texte (v {version_to_save})...")
    training_data = [
        ("def class import def for if else elif try except finally with as pass return yield", "python_script"),
        ("lambda def __init__ self cls args kwargs decorator", "python_script"),
        ("import os sys json re math numpy pandas sklearn", "python_script"),
        ("the a is are was were will be and or but for of to in on at", "english_text"),
        ("sentence paragraph word letter text document write read speak", "english_text"),
        ("hello world good morning example another however therefore because", "english_text"),
    ]
    texts, labels = zip(*training_data)
    model = make_pipeline(CountVectorizer(ngram_range=(1, 2)), MultinomialNB())
    model.fit(texts, labels)
    try:
        joblib.dump(model, TEXT_CLASSIFIER_MODEL_PATH)
        print(f"[AI_ANALYZER] Modèle texte sauvegardé: {TEXT_CLASSIFIER_MODEL_PATH}")
        save_local_text_classifier_version(version_to_save)
        text_classifier_model = model
    except Exception as e:
        print(f"[AI_ANALYZER] Erreur sauvegarde modèle texte: {e}")
        text_classifier_model = model # Utiliser en mémoire
    return text_classifier_model # Retourner le modèle entraîné

def load_text_classifier(force_retrain=False):
    global text_classifier_model, current_text_classifier_version
    ensure_model_dir_exists()
    
    current_text_classifier_version = get_local_text_classifier_version() # Met à jour la variable globale
    print(f"[AI_ANALYZER] Version locale modèle texte (au chargement): {current_text_classifier_version}")

    if not force_retrain and os.path.exists(TEXT_CLASSIFIER_MODEL_PATH):
        try:
            print(f"[AI_ANALYZER] Chargement modèle texte: {TEXT_CLASSIFIER_MODEL_PATH}")
            text_classifier_model = joblib.load(TEXT_CLASSIFIER_MODEL_PATH)
            print("[AI_ANALYZER] Modèle texte chargé.")
            return text_classifier_model
        except Exception as e:
            print(f"[AI_ANALYZER] Erreur chargement modèle texte: {e}. Réentraînement.")
    
    return train_and_save_text_classifier(current_text_classifier_version) # Entraîne avec la version actuelle ou par défaut

text_classifier_model = load_text_classifier() # Charger au moment de l'import

def predict_text_content_type(text_content_str):
    global text_classifier_model
    if text_classifier_model is None:
        print("[AI_ANALYZER] Modèle texte non dispo pour prédiction. Tentative de chargement/entraînement.")
        load_text_classifier() # S'assure que le modèle est chargé ou entraîné
        if text_classifier_model is None: return "unknown_text_model_unavailable"
    if not isinstance(text_content_str, str) or not text_content_str.strip(): return "unknown_text_or_empty"    
    try:
        return text_classifier_model.predict([text_content_str])[0]
    except Exception as e:
        print(f"[AI_ANALYZER] Erreur prédiction type texte: {e}"); return "unknown_text_prediction_error"

def analyze_file_content(file_path):
    if not os.path.exists(file_path): return "file_not_found"
    if os.path.getsize(file_path) == 0: return "empty_file"
    file_type_by_magic = "unknown"
    if PYTHON_MAGIC_AVAILABLE:
        try:
            mime_type = magic.from_file(file_path, mime=True)
            if mime_type: # S'assurer que mime_type n'est pas None
                if mime_type.startswith('image/jpeg'): file_type_by_magic = "jpeg_image"
                elif mime_type.startswith('image/png'): file_type_by_magic = "png_image"
                elif mime_type.startswith('image/gif'): file_type_by_magic = "gif_image"
                elif mime_type.startswith('image/webp'): file_type_by_magic = "webp_image"
                elif mime_type.startswith('application/pdf'): file_type_by_magic = "pdf_document"
                elif mime_type.startswith('application/zip'): file_type_by_magic = "zip_archive"
                elif mime_type.startswith('application/x-rar-compressed'): file_type_by_magic = "rar_archive"
                elif mime_type.startswith('application/x-7z-compressed'): file_type_by_magic = "7z_archive"
                elif mime_type.startswith('application/x-tar'): file_type_by_magic = "tar_archive"
                elif mime_type.startswith('application/gzip') or mime_type.startswith('application/x-gzip'): file_type_by_magic = "gzip_archive"
                elif mime_type.startswith('application/x-bzip2'): file_type_by_magic = "bzip2_archive"
                elif mime_type.startswith('application/x-xz'): file_type_by_magic = "xz_archive"
                elif mime_type.startswith('audio/mpeg'): file_type_by_magic = "mp3_audio"
                elif mime_type.startswith('audio/'): file_type_by_magic = mime_type.replace('audio/', '').split(';')[0] + "_audio"
                elif mime_type.startswith('video/'): file_type_by_magic = mime_type.replace('video/', '').split(';')[0] + "_video"
                elif mime_type.startswith('text/x-python'): file_type_by_magic = "python_script"
                elif mime_type.startswith('text/html'): file_type_by_magic = "html_document"
                elif mime_type.startswith('text/xml'): file_type_by_magic = "xml_document"
                elif mime_type.startswith('application/json'): file_type_by_magic = "json_data"
                elif mime_type.startswith('text/css'): file_type_by_magic = "css_stylesheet"
                elif mime_type.startswith('text/plain'): file_type_by_magic = "plain_text"
                elif mime_type.startswith('text/'): file_type_by_magic = "generic_text"
                elif mime_type.startswith('application/octet-stream'): file_type_by_magic = "generic_binary"
                else: file_type_by_magic = mime_type.replace('/', '_').split(';')[0]
            else: # mime_type est None
                 file_type_by_magic = "unknown_mime_none"

            text_like_types = ["plain_text", "python_script", "generic_text", "html_document", "xml_document", "json_data", "css_stylesheet"]
            if file_type_by_magic in text_like_types:
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f: content_sample = f.read(2048) 
                    if file_type_by_magic == "python_script": return "python_script"
                    refined_text_type = predict_text_content_type(content_sample)
                    return refined_text_type if refined_text_type and "unknown" not in refined_text_type else file_type_by_magic
                except Exception: return file_type_by_magic 
            return file_type_by_magic
        except Exception as e: print(f"[AI_ANALYZER] Avertissement: Erreur python-magic {file_path}: {e}")
    try: # Fallback
        with open(file_path, 'rb') as f_rb: content_sample_bytes = f_rb.read(512)
        if b'\0' in content_sample_bytes: return "unknown_binary_heuristic_null"
        try:
            content_sample_text = content_sample_bytes.decode('utf-8', errors='ignore') # Ignore errors for heuristic
            non_printable = sum(1 for char_code in content_sample_bytes[:100] if not (32 <= char_code <= 126 or char_code in [9,10,13]))
            if non_printable > 10: return "unknown_binary_heuristic_nonprint"
            text_type = predict_text_content_type(content_sample_text)
            return text_type if text_type and "unknown" not in text_type else "generic_text_heuristic"
        except UnicodeDecodeError: return "unknown_binary_read_error"
    except Exception: return "error_during_fallback_analysis"

if __name__ == '__main__':
    print(f"[AI_ANALYZER TEST] python-magic: {PYTHON_MAGIC_AVAILABLE}")
    print(f"[AI_ANALYZER TEST] Modèle texte initial (v {get_local_text_classifier_version()}): {'Oui' if text_classifier_model else 'Non'}")
    test_py_content = "def my_func():\n import sys\n print('hello')"; temp_file = "temp_test_code.py"
    with open(temp_file, "w") as f: f.write(test_py_content)
    print(f"[AI_ANALYZER TEST] Analyse {temp_file}: {analyze_file_content(temp_file)}")
    if os.path.exists(temp_file): os.remove(temp_file)
EOF
echo "aicompress/ai_analyzer.py mis à jour."

# Étape 3: Création de aicompress/ota_updater.py
echo_section_header "Étape 3: Création de aicompress/ota_updater.py"
mkdir -p aicompress # S'assurer que le dossier existe
cat << 'EOF' > aicompress/ota_updater.py
# aicompress/ota_updater.py
import os
import json
import requests
import hashlib
import shutil

# Importer les configurations et fonctions nécessaires depuis ai_analyzer
# Cela crée une dépendance, mais c'est pour gérer le modèle spécifique
try:
    from .ai_analyzer import (
        MODELS_SPECIFIC_DIR, 
        TEXT_CLASSIFIER_MODEL_NAME, 
        TEXT_CLASSIFIER_CONFIG_PATH, # Utilisé pour savoir où est la config de version
        TEXT_CLASSIFIER_MODEL_PATH,   # Utilisé pour savoir où sauvegarder le modèle
        save_local_text_classifier_version,
        load_text_classifier, # Pour recharger après mise à jour
        get_local_text_classifier_version,
        ensure_model_dir_exists,
        _default_log # Utiliser la même fonction de log par défaut
    )
    AI_ANALYZER_IMPORTS_OK = True
except ImportError as e:
    print(f"ERREUR (ota_updater.py): Impossible d'importer depuis ai_analyzer: {e}")
    AI_ANALYZER_IMPORTS_OK = False
    # Définir des fallbacks pour que le reste du module puisse être importé sans planter
    MODELS_SPECIFIC_DIR = os.path.join(os.path.expanduser("~"), ".aicompress_fallback", "models")
    TEXT_CLASSIFIER_MODEL_NAME = "text_classifier.joblib"
    TEXT_CLASSIFIER_CONFIG_PATH = os.path.join(MODELS_SPECIFIC_DIR, "text_classifier_config.json")
    TEXT_CLASSIFIER_MODEL_PATH = os.path.join(MODELS_SPECIFIC_DIR, TEXT_CLASSIFIER_MODEL_NAME)
    def _default_log(message): print(message)
    def ensure_model_dir_exists(): os.makedirs(MODELS_SPECIFIC_DIR, exist_ok=True)
    def save_local_text_classifier_version(v): _default_log(f"Fallback save_local_text_classifier_version {v}")
    def load_text_classifier(force_retrain=False): _default_log("Fallback load_text_classifier"); return None
    def get_local_text_classifier_version(): _default_log("Fallback get_local_text_classifier_version"); return "0.0"


# !!! IMPORTANT: REMPLACEZ CETTE URL PAR CELLE DE VOTRE FICHIER model_info.json HÉBERGÉ !!!
MODEL_INFO_URL = "https://gist.githubusercontent.com/VotreNomUser/IDdeVotreGist/raw/xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx/model_info.json"
# Exemple: MODEL_INFO_URL = "https://gist.githubusercontent.com/anonymous/some_gist_id/raw/model_info.json"

def _calculate_sha256(filepath, log_callback=_default_log):
    sha256_hash = hashlib.sha256()
    try:
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    except IOError as e:
        log_callback(f"[OTA_UPDATER] Erreur lecture fichier pour checksum: {e}")
        return None

def check_for_model_updates(log_callback=_default_log):
    if not AI_ANALYZER_IMPORTS_OK:
        log_callback("[OTA_UPDATER] ERREUR: Dépendances internes non chargées. Mise à jour impossible.")
        return {}
        
    log_callback("[OTA_UPDATER] Vérification des mises à jour modèles IA...")
    if "URL_DE_VOTRE_MODEL_INFO.json" in MODEL_INFO_URL or "VotreNomUser/IDdeVotreGist" in MODEL_INFO_URL : # Vérif placeholder
        log_callback(f"[OTA_UPDATER] ERREUR: MODEL_INFO_URL ('{MODEL_INFO_URL}') non configurée. Veuillez la modifier dans aicompress/ota_updater.py.")
        return {}
        
    try:
        response = requests.get(MODEL_INFO_URL, timeout=15)
        response.raise_for_status()
        server_model_info = response.json()
        log_callback("[OTA_UPDATER] Informations modèles récupérées du serveur.")
    except requests.exceptions.Timeout:
        log_callback(f"[OTA_UPDATER] Timeout lors de la récupération des infos modèles depuis {MODEL_INFO_URL}.")
        return {}
    except requests.exceptions.RequestException as e:
        log_callback(f"[OTA_UPDATER] Erreur récupération infos modèles: {e}")
        return {}
    except json.JSONDecodeError as e:
        log_callback(f"[OTA_UPDATER] Erreur décodage JSON infos modèles: {e}")
        return {}

    updates_available = {}
    if "text_classifier" in server_model_info:
        tc_server_info = server_model_info["text_classifier"]
        server_version = tc_server_info.get("latest_version")
        local_version = get_local_text_classifier_version()
        
        log_callback(f"[OTA_UPDATER] Text Classifier - Local: v{local_version}, Serveur: v{server_version}")
        
        if server_version and (local_version is None or server_version > local_version): # Comparaison simple
            log_callback(f"[OTA_UPDATER] Nouvelle version disponible pour Text Classifier: {server_version}")
            updates_available["text_classifier"] = tc_server_info
        else:
            log_callback("[OTA_UPDATER] Text Classifier est à jour.")
            
    # Ajouter ici des vérifications pour d'autres modèles (mnist_encoder, etc.)
    # Exemple:
    # if "mnist_encoder" in server_model_info:
    #     # ... logique similaire ...

    return updates_available

def download_and_install_model(model_name, model_info, log_callback=_default_log):
    if not AI_ANALYZER_IMPORTS_OK:
        log_callback("[OTA_UPDATER] ERREUR: Dépendances internes non chargées. Installation modèle impossible.")
        return False

    ensure_model_dir_exists()
    model_url = model_info.get("url")
    expected_checksum = model_info.get("checksum_sha256")
    new_version = model_info.get("latest_version")

    if not model_url:
        log_callback(f"[OTA_UPDATER] ERREUR: URL manquante pour modèle {model_name}.")
        return False

    local_model_file_path = None
    config_update_function = None

    if model_name == "text_classifier":
        local_model_file_path = TEXT_CLASSIFIER_MODEL_PATH
        config_update_function = save_local_text_classifier_version
        reload_function = lambda: load_text_classifier(force_retrain=False) # Pour recharger
    # elif model_name == "mnist_encoder":
    #     local_model_file_path = os.path.join(MODELS_SPECIFIC_DIR, "mnist_encoder.keras") # Exemple
    #     # config_update_function = save_local_mnist_encoder_version # Une fonction similaire
    #     # reload_function = load_mnist_ae_models # Qui recharge l'AE
    else:
        log_callback(f"[OTA_UPDATER] ERREUR: Nom de modèle inconnu pour téléchargement: {model_name}")
        return False
    
    download_path_temp = local_model_file_path + ".tmp"

    log_callback(f"[OTA_UPDATER] Téléchargement {model_name} v{new_version} depuis {model_url}...")
    try:
        with requests.get(model_url, stream=True, timeout=120) as r: # Timeout plus long pour les modèles
            r.raise_for_status()
            with open(download_path_temp, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192 * 16): f.write(chunk) # Plus gros chunk
        log_callback(f"[OTA_UPDATER] {model_name} téléchargé: {download_path_temp}")

        if expected_checksum:
            log_callback(f"[OTA_UPDATER] Vérification checksum pour {model_name}...")
            downloaded_checksum = _calculate_sha256(download_path_temp, log_callback)
            if downloaded_checksum != expected_checksum:
                log_callback(f"[OTA_UPDATER] ERREUR: Checksum invalide. Attendu: {expected_checksum}, Obtenu: {downloaded_checksum}")
                os.remove(download_path_temp); return False
            log_callback(f"[OTA_UPDATER] Checksum vérifié.")

        # Sauvegarde de l'ancien modèle si existant
        if os.path.exists(local_model_file_path):
            backup_path = local_model_file_path + ".bak"
            if os.path.exists(backup_path): os.remove(backup_path) # Supprimer ancienne sauvegarde
            shutil.move(local_model_file_path, backup_path)
            log_callback(f"[OTA_UPDATER] Ancien modèle sauvegardé en .bak: {backup_path}")
        
        shutil.move(download_path_temp, local_model_file_path)
        log_callback(f"[OTA_UPDATER] {model_name} installé: {local_model_file_path}")

        if config_update_function: config_update_function(new_version)
        if reload_function: reload_function() # Recharger le modèle en mémoire
        
        log_callback(f"[OTA_UPDATER] Modèle {model_name} (v{new_version}) mis à jour et rechargé.")
        return True
    except Exception as e:
        log_callback(f"[OTA_UPDATER] Erreur installation {model_name}: {e}")
        import traceback; log_callback(traceback.format_exc())
    if os.path.exists(download_path_temp): os.remove(download_path_temp)
    return False

if __name__ == '__main__':
    print("--- Test OTA Updater (nécessite configuration MODEL_INFO_URL et serveur de test) ---")
    if "URL_DE_VOTRE_MODEL_INFO.json" in MODEL_INFO_URL or "VotreNomUser/IDdeVotreGist" in MODEL_INFO_URL:
        print("MODIFIEZ MODEL_INFO_URL dans aicompress/ota_updater.py pour tester.")
    elif not AI_ANALYZER_IMPORTS_OK:
        print("Imports depuis ai_analyzer ont échoué. Vérifiez les erreurs.")
    else:
        ensure_model_dir_exists() # S'assurer que le dossier existe pour les tests
        # Simuler une version locale plus ancienne pour le test
        save_local_text_classifier_version("0.9") 
        print(f"Version locale (avant test): {get_local_text_classifier_version()}")
        
        updates = check_for_model_updates()
        if updates:
            print("Mises à jour disponibles:", updates)
            # Pour le test, on tente de mettre à jour text_classifier s'il est listé
            if "text_classifier" in updates:
                info = updates["text_classifier"]
                if input(f"Test: Télécharger {info.get('latest_version')}? (o/N): ").lower() == 'o':
                    download_and_install_model("text_classifier", info)
                    print(f"Version locale (après test): {get_local_text_classifier_version()}")
        else:
            print("Aucune mise à jour ou erreur vérification.")
EOF
echo "aicompress/ota_updater.py créé."

# Étape 4: Mise à jour de aicompress_gui.py
echo_section_header "Étape 4: Mise à jour de aicompress_gui.py pour le bouton OTA"
cat << 'EOF' > aicompress_gui.py
# aicompress_gui.py
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, simpledialog
import os
import threading 

from aicompress.core import (compress_to_aic, extract_archive, 
                             AI_ANALYZER_AVAILABLE, RARFILE_AVAILABLE, DEFAULT_AIC_EXTENSION,
                             CRYPTOGRAPHY_AVAILABLE)
# Importer les nouvelles fonctions OTA
try:
    from aicompress.ota_updater import check_for_model_updates, download_and_install_model
    OTA_AVAILABLE = True
except ImportError as e_ota:
    print(f"AVERTISSEMENT (GUI): ota_updater.py non trouvé ou erreur import: {e_ota}")
    OTA_AVAILABLE = False
    def check_for_model_updates(log_callback): log_callback("[GUI] Module OTA non dispo."); return {}
    def download_and_install_model(name, info, log_callback): log_callback("[GUI] Module OTA non dispo."); return False


class AICompressGUI:
    def __init__(self, root_window):
        self.root = root_window
        self.root.title(f"AICompress Alpha (OTA Ready) - {DEFAULT_AIC_EXTENSION}")
        self.root.geometry("780x680") # Un peu plus haut pour le nouveau bouton

        self.files_to_compress = [] 

        main_frame = tk.Frame(self.root, padx=10, pady=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # ... (Section Compression et Décompression comme dans la version précédente avec mots de passe) ...
        # (Je copie les sections pour la complétude, elles sont identiques à votre dernière version fonctionnelle)
        compress_frame = tk.LabelFrame(main_frame, text=f"Compresser vers {DEFAULT_AIC_EXTENSION}", padx=10, pady=10)
        compress_frame.pack(fill=tk.X, pady=5)
        files_selection_controls_frame = tk.Frame(compress_frame); files_selection_controls_frame.pack(fill=tk.X, pady=(0,5))
        btn_add_files = tk.Button(files_selection_controls_frame, text="Ajouter Fichier(s)", command=self.add_files); btn_add_files.pack(side=tk.LEFT, padx=5)
        btn_add_folder = tk.Button(files_selection_controls_frame, text="Ajouter Dossier", command=self.add_folder); btn_add_folder.pack(side=tk.LEFT, padx=5)
        btn_clear_list = tk.Button(files_selection_controls_frame, text="Vider Liste", command=self.clear_file_list); btn_clear_list.pack(side=tk.LEFT, padx=5)
        listbox_frame = tk.Frame(compress_frame); listbox_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        self.listbox_files = tk.Listbox(listbox_frame, selectmode=tk.EXTENDED, height=6)
        self.listbox_files_scrollbar_y = tk.Scrollbar(listbox_frame, orient=tk.VERTICAL, command=self.listbox_files.yview)
        self.listbox_files_scrollbar_x = tk.Scrollbar(listbox_frame, orient=tk.HORIZONTAL, command=self.listbox_files.xview)
        self.listbox_files.config(yscrollcommand=self.listbox_files_scrollbar_y.set, xscrollcommand=self.listbox_files_scrollbar_x.set)
        self.listbox_files_scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y); self.listbox_files_scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)
        self.listbox_files.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        output_frame = tk.Frame(compress_frame); output_frame.pack(fill=tk.X, pady=5) 
        lbl_output_aic = tk.Label(output_frame, text=f"Fichier {DEFAULT_AIC_EXTENSION}:"); lbl_output_aic.pack(side=tk.LEFT)
        self.entry_output_aic_path = tk.Entry(output_frame, width=35); self.entry_output_aic_path.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        btn_browse_output_aic = tk.Button(output_frame, text="Parcourir...", command=self.browse_output_aic); btn_browse_output_aic.pack(side=tk.LEFT, padx=5)
        compress_options_frame = tk.Frame(compress_frame); compress_options_frame.pack(fill=tk.X, pady=(5,0))
        self.var_encrypt_aic = tk.BooleanVar()
        self.chk_encrypt_aic = tk.Checkbutton(compress_options_frame, text="Protéger par mot de passe:", variable=self.var_encrypt_aic, command=self.toggle_password_entry_compress, state=tk.NORMAL if CRYPTOGRAPHY_AVAILABLE else tk.DISABLED)
        self.chk_encrypt_aic.pack(side=tk.LEFT)
        self.entry_password_compress = tk.Entry(compress_options_frame, show="*", width=25, state=tk.DISABLED); self.entry_password_compress.pack(side=tk.LEFT, padx=5)
        if not CRYPTOGRAPHY_AVAILABLE: tk.Label(compress_options_frame, text="(Crypto non dispo)", fg="red").pack(side=tk.LEFT)
        self.btn_compress_action = tk.Button(compress_frame, text="COMPRESSER", command=self.start_compression_thread, bg="lightblue", relief=tk.RAISED, width=15); self.btn_compress_action.pack(pady=10)
        decompress_frame = tk.LabelFrame(main_frame, text="Décompresser Archive", padx=10, pady=10); decompress_frame.pack(fill=tk.X, pady=10)
        source_archive_frame = tk.Frame(decompress_frame); source_archive_frame.pack(fill=tk.X, pady=2)
        lbl_source_archive = tk.Label(source_archive_frame, text="Archive source:"); lbl_source_archive.pack(side=tk.LEFT)
        self.entry_source_archive_path = tk.Entry(source_archive_frame, width=35); self.entry_source_archive_path.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        btn_browse_source_archive = tk.Button(source_archive_frame, text="Parcourir...", command=self.browse_source_archive); btn_browse_source_archive.pack(side=tk.LEFT, padx=5)
        dest_folder_frame = tk.Frame(decompress_frame); dest_folder_frame.pack(fill=tk.X, pady=2)
        lbl_dest_folder = tk.Label(dest_folder_frame, text="Dossier destination:"); lbl_dest_folder.pack(side=tk.LEFT)
        self.entry_dest_folder_path = tk.Entry(dest_folder_frame, width=35); self.entry_dest_folder_path.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        btn_browse_dest_folder = tk.Button(dest_folder_frame, text="Parcourir...", command=self.browse_dest_folder); btn_browse_dest_folder.pack(side=tk.LEFT, padx=5)
        password_frame_decompress = tk.Frame(decompress_frame); password_frame_decompress.pack(fill=tk.X, pady=2)
        lbl_password_decompress = tk.Label(password_frame_decompress, text="Mot de passe (si requis):"); lbl_password_decompress.pack(side=tk.LEFT)
        self.entry_password_decompress = tk.Entry(password_frame_decompress, show="*", width=30); self.entry_password_decompress.pack(side=tk.LEFT, padx=5)
        self.btn_decompress_action = tk.Button(decompress_frame, text="DÉCOMPRESSER", command=self.start_decompression_thread, bg="lightgreen", relief=tk.RAISED, width=15); self.btn_decompress_action.pack(pady=10)
        
        # --- NOUVEAU: Cadre pour les actions supplémentaires (OTA) ---
        extra_actions_frame = tk.LabelFrame(main_frame, text="Utilitaires", padx=10, pady=10)
        extra_actions_frame.pack(fill=tk.X, pady=5) # Placé avant les logs

        self.btn_check_ota_updates = tk.Button(extra_actions_frame, text="Vérifier MàJ Modèles IA", 
                                               command=self.run_check_ota_updates,
                                               state=tk.NORMAL if OTA_AVAILABLE else tk.DISABLED)
        self.btn_check_ota_updates.pack(side=tk.LEFT, padx=5)
        if not OTA_AVAILABLE:
            ota_unavailable_label = tk.Label(extra_actions_frame, text="(Module OTA non dispo)", fg="red")
            ota_unavailable_label.pack(side=tk.LEFT)
        # --- FIN NOUVEAU ---

        log_frame = tk.LabelFrame(main_frame, text="Logs et Messages", padx=10, pady=10)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        self.log_text = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, height=8, state=tk.DISABLED) # Hauteur un peu réduite
        self.log_text.pack(fill=tk.BOTH, expand=True)

        self.log_message(f"AICompress GUI (v0.6 OTA) Initialisé. Crypto: {'OK' if CRYPTOGRAPHY_AVAILABLE else 'NON DISPO'}. IA Analyzer: {'OK' if AI_ANALYZER_AVAILABLE else 'NON DISPO'}. OTA: {'OK' if OTA_AVAILABLE else 'NON DISPO'}.")

    def toggle_password_entry_compress(self): # ... (comme avant)
        if self.var_encrypt_aic.get() and CRYPTOGRAPHY_AVAILABLE: self.entry_password_compress.config(state=tk.NORMAL)
        else: self.entry_password_compress.config(state=tk.DISABLED); self.entry_password_compress.delete(0, tk.END)

    def log_message(self, message): # ... (comme avant)
        def _update_log():
            if self.log_text.winfo_exists(): self.log_text.config(state=tk.NORMAL); self.log_text.insert(tk.END, str(message) + "\n"); self.log_text.see(tk.END); self.log_text.config(state=tk.DISABLED)
        if self.root.winfo_exists(): self.root.after(0, _update_log)
        print(message)
    
    # ... (add_files, add_folder, clear_file_list, browse_*, _set_buttons_state comme avant) ...
    # (Je copie les versions courtes pour la concision du script bash)
    def add_files(self): filepaths = filedialog.askopenfilenames(title="Sélectionner fichier(s)"); \
        if filepaths: c=0; [(self.files_to_compress.append(fp), self.listbox_files.insert(tk.END, fp), globals().update(c=globals().get("c",0)+1)) for fp in filepaths if fp not in self.files_to_compress]; \
        if globals().get("c",0)>0: self.log_message(f"[GUI] {globals().get('c',0)} fichier(s) ajouté(s)."); self.listbox_files.see(tk.END); globals()["c"]=0
    def add_folder(self): folderpath = filedialog.askdirectory(title="Sélectionner dossier"); \
        if folderpath: \
            if folderpath not in self.files_to_compress: self.files_to_compress.append(folderpath); self.listbox_files.insert(tk.END, folderpath); self.log_message(f"[GUI] Dossier '{folderpath}' ajouté.") ; \
            else: self.log_message(f"[GUI] Dossier '{folderpath}' déjà listé."); \
            self.listbox_files.see(tk.END)
    def clear_file_list(self): self.files_to_compress = []; self.listbox_files.delete(0, tk.END); self.log_message("[GUI] Liste vidée.")
    def browse_output_aic(self): filepath = filedialog.asksaveasfilename(title="Enregistrer sous...", defaultextension=DEFAULT_AIC_EXTENSION, filetypes=[(f"AICompress (*{DEFAULT_AIC_EXTENSION})", f"*{DEFAULT_AIC_EXTENSION}"), ("Tous", "*.*")]); \
        if filepath: base, ext = os.path.splitext(filepath); \
        if ext.lower() != DEFAULT_AIC_EXTENSION.lower(): filepath = base + DEFAULT_AIC_EXTENSION; \
        self.entry_output_aic_path.delete(0, tk.END); self.entry_output_aic_path.insert(0, filepath); self.log_message(f"[GUI] Sortie AIC: {filepath}")
    def browse_source_archive(self): filepath = filedialog.askopenfilename(title="Sélectionner archive", filetypes=[("Archives", f"*{DEFAULT_AIC_EXTENSION} *.zip *.rar *.7z"), ("Tous", "*.*")]); \
        if filepath: self.entry_source_archive_path.delete(0,tk.END); self.entry_source_archive_path.insert(0,filepath); self.log_message(f"[GUI] Archive source: {filepath}")
    def browse_dest_folder(self): folderpath = filedialog.askdirectory(title="Sélectionner dossier destination"); \
        if folderpath: self.entry_dest_folder_path.delete(0,tk.END); self.entry_dest_folder_path.insert(0,folderpath); self.log_message(f"[GUI] Dossier destination: {folderpath}")
    def _set_buttons_state(self, state): # ... (version complète avec gestion des textes)
        target_buttons = [self.btn_compress_action, self.btn_decompress_action, self.btn_check_ota_updates]
        default_texts = {"COMPRESSER": "COMPRESSER", "DÉCOMPRESSER": "DÉCOMPRESSER", "Vérifier MàJ Modèles IA": "Vérifier MàJ Modèles IA"}
        busy_texts = {"COMPRESSER": "Compression...", "DÉCOMPRESSER": "Décompression...", "Vérifier MàJ Modèles IA": "Vérification..."}

        for btn in target_buttons:
            if btn and btn.winfo_exists(): # Vérifier si le bouton existe
                # Trouver la clé du texte par défaut pour ce bouton
                btn_key = None
                for k, v_default in default_texts.items():
                    # Comparer avec le texte actuel ou le texte occupé si le bouton est déjà occupé
                    if btn.cget("text") == v_default or btn.cget("text") == busy_texts.get(k):
                        btn_key = k
                        break
                
                if btn_key: # Si on a trouvé une clé correspondante
                    new_text = busy_texts[btn_key] if state == tk.DISABLED else default_texts[btn_key]
                    btn.config(state=state, text=new_text)


    # --- Méthodes pour la compression et décompression (avec password_compress pour run_compression) ---
    # (Elles sont identiques à votre dernière version fonctionnelle, je les inclus pour la complétude)
    def start_compression_thread(self):
        if not self.files_to_compress: messagebox.showerror("Erreur", "Aucun fichier/dossier."); return
        output_path = self.entry_output_aic_path.get()
        if not output_path: messagebox.showerror("Erreur", "Chemin de sortie requis."); return
        base, ext = os.path.splitext(output_path)
        if ext.lower() != DEFAULT_AIC_EXTENSION.lower(): output_path = base + DEFAULT_AIC_EXTENSION; self.log_message(f"[GUI] Extension {DEFAULT_AIC_EXTENSION} appliquée. Sortie: {output_path}"); self.entry_output_aic_path.delete(0, tk.END); self.entry_output_aic_path.insert(0, output_path)
        password_for_compression = None
        if self.var_encrypt_aic.get():
            if not CRYPTOGRAPHY_AVAILABLE: messagebox.showerror("Erreur", "Crypto non dispo."); return
            password_for_compression = self.entry_password_compress.get()
            if not password_for_compression: messagebox.showerror("Erreur", "Mdp requis si chiffré."); return
            if len(password_for_compression) < 4: messagebox.showwarning("Attention", "Mdp court.")
        self._set_buttons_state(tk.DISABLED)
        self.log_message(f"[GUI] Compression vers {output_path} (Chiffré: {'Oui' if password_for_compression else 'Non'})...")
        thread = threading.Thread(target=self.run_compression, args=(list(self.files_to_compress), output_path, password_for_compression), daemon=True); thread.start()
    def run_compression(self, files_list, output_aic_path, password_compress):
        try:
            success, status_msg = compress_to_aic(files_list, output_aic_path, password_compress=password_compress, log_callback=self.log_message)
            if success:
                if self.root.winfo_exists(): self.root.after(0, lambda: messagebox.showinfo("Succès", f"Compression terminée !\nSauvegardé: {output_aic_path}"))
            else:
                if self.root.winfo_exists(): self.root.after(0, lambda: messagebox.showerror("Échec", f"Compression échouée.\nMotif: {status_msg}"))
        except Exception as e: self.log_message(f"[GUI] Erreur majeure compression: {e}"); import traceback; self.log_message(f"[GUI] Traceback: {traceback.format_exc()}"); \
            if self.root.winfo_exists(): self.root.after(0, lambda: messagebox.showerror("Erreur Fatale", f"Erreur: {e}"))
        finally:
            if self.root.winfo_exists(): self.root.after(0, self._set_buttons_state, tk.NORMAL)
    def start_decompression_thread(self):
        source_archive = self.entry_source_archive_path.get(); dest_folder = self.entry_dest_folder_path.get(); password = self.entry_password_decompress.get()
        if not source_archive or not dest_folder: messagebox.showerror("Erreur", "Archive et destination requises."); return
        if not os.path.exists(source_archive): messagebox.showerror("Erreur", f"Archive '{source_archive}' non trouvée."); return
        self._set_buttons_state(tk.DISABLED)
        self.log_message(f"[GUI] Décompression de {source_archive} (Mdp: {'Oui' if password else 'Non'})...")
        thread = threading.Thread(target=self.run_decompression, args=(source_archive, dest_folder, password if password else None), daemon=True); thread.start()
    def run_decompression(self, archive_path, output_dir, password):
        try:
            success, status_code = extract_archive(archive_path, output_dir, password=password, log_callback=self.log_message)
            if success:
                if self.root.winfo_exists(): self.root.after(0, lambda: messagebox.showinfo("Succès", f"Décompression terminée: {output_dir}"))
            else:
                password_errors = ["PasswordError", "PasswordNeeded", "BadRarFileOrPassword"] # Unifier les codes d'erreur mdp
                is_pwd_error = any(err_code_part in status_code for err_code_part in password_errors) if isinstance(status_code, str) else False
                if is_pwd_error:
                    self.log_message(f"[GUI] Échec décompression: Mdp requis/incorrect pour {archive_path}.")
                    if self.root.winfo_exists(): self.root.after(0, self.prompt_for_password_and_retry_decompression, archive_path, output_dir)
                else:
                    self.log_message(f"[GUI] Échec décompression '{archive_path}'. Status: {status_code}")
                    if self.root.winfo_exists(): self.root.after(0, lambda: messagebox.showerror("Échec", f"Décompression échouée '{archive_path}'.\nMotif: {status_code}"))
        except Exception as e: self.log_message(f"[GUI] Erreur majeure décompression: {e}"); import traceback; self.log_message(f"[GUI] Traceback: {traceback.format_exc()}"); \
            if self.root.winfo_exists(): self.root.after(0, lambda: messagebox.showerror("Erreur Fatale", f"Erreur: {e}"))
        finally:
            if self.root.winfo_exists(): self.root.after(0, self._set_buttons_state, tk.NORMAL)
    def prompt_for_password_and_retry_decompression(self, archive_path, output_dir):
        if not self.root.winfo_exists(): return
        new_password = simpledialog.askstring("Mot de Passe Requis", f"L'archive '{os.path.basename(archive_path)}' est protégée ou mdp incorrect.\nNouveau mot de passe :", show='*')
        if new_password is not None: 
            self.log_message(f"[GUI] Nouvelle tentative décompression {archive_path} avec mdp.")
            self.entry_password_decompress.delete(0, tk.END); self.entry_password_decompress.insert(0, new_password)
            self._set_buttons_state(tk.DISABLED)
            thread = threading.Thread(target=self.run_decompression, args=(archive_path, output_dir, new_password), daemon=True); thread.start()
        else:
            self.log_message("[GUI] Décompression annulée (pas de nouveau mdp fourni).")
            if self.root.winfo_exists(): self.root.after(0, self._set_buttons_state, tk.NORMAL)

    # --- NOUVELLES MÉTHODES POUR OTA ---
    def run_check_ota_updates(self):
        if not OTA_AVAILABLE:
            self.log_message("[GUI] Fonctionnalité de mise à jour OTA non disponible (module manquant).")
            messagebox.showwarning("OTA Indisponible", "Le module de mise à jour OTA n'a pas pu être chargé.")
            return

        self.log_message("[GUI] Vérification des mises à jour des modèles IA...")
        self._set_buttons_state(tk.DISABLED) # Désactiver tous les boutons d'action
        self.btn_check_ota_updates.config(text="Vérification...") # Spécifique à ce bouton

        thread = threading.Thread(target=self._perform_ota_check_and_prompt, daemon=True)
        thread.start()

    def _perform_ota_check_and_prompt(self):
        # S'assurer que les opérations GUI se font via self.root.after si elles modifient l'UI
        # et viennent d'un thread. messagebox est problématique à appeler directement depuis un thread
        # sans bloquer le thread GUI ou risquer des erreurs.
        # Pour l'instant, on logue principalement et on fera un simple messagebox à la fin.
        
        updates_info_for_gui = [] # Sera une liste de strings à afficher
        models_to_update_actions = [] # Sera une liste de (nom_modele, info_serveur)

        try:
            updates_available = check_for_model_updates(log_callback=self.log_message)
            
            if not updates_available:
                updates_info_for_gui.append("Aucune nouvelle mise à jour de modèle IA disponible.")
            else:
                updates_info_for_gui.append("Mises à jour de modèles IA disponibles:")
                for model_name, server_info in updates_available.items():
                    info_str = (f"  - {model_name}: v{server_info.get('latest_version')} "
                                f"({server_info.get('description', 'N/A')})")
                    updates_info_for_gui.append(info_str)
                    models_to_update_actions.append((model_name, server_info))
                updates_info_for_gui.append("\nSouhaitez-vous installer ces mises à jour ?")
            
            # Afficher les résultats et demander confirmation dans le thread principal
            if self.root.winfo_exists():
                self.root.after(0, self._show_ota_results_and_prompt_install, updates_info_for_gui, models_to_update_actions)

        except Exception as e:
            self.log_message(f"[GUI] Erreur pendant la vérification OTA: {e}")
            import traceback
            self.log_message(f"[GUI] Traceback OTA: {traceback.format_exc()}")
            if self.root.winfo_exists():
                self.root.after(0, lambda: messagebox.showerror("Erreur OTA", f"Erreur pendant la vérification des MàJ: {e}"))
        finally:
            # Réactiver le bouton de vérification, les autres boutons sont gérés par leur propre logique
             if self.root.winfo_exists():
                self.root.after(0, lambda: {
                    self.btn_check_ota_updates.config(state=tk.NORMAL, text="Vérifier MàJ Modèles IA"),
                    # Ne pas réactiver les autres boutons ici, ils ont leur propre cycle
                    # self._set_buttons_state(tk.NORMAL) # Non, car on ne veut pas interrompre une autre op
                })


    def _show_ota_results_and_prompt_install(self, info_messages, models_to_install):
        if not self.root.winfo_exists(): return

        full_message = "\n".join(info_messages)
        if models_to_install: # S'il y a des actions à faire (des modèles à mettre à jour)
            user_response = messagebox.askyesno("Mises à Jour Modèles IA", full_message)
            if user_response:
                self.log_message("[GUI] L'utilisateur a accepté d'installer les mises à jour.")
                self._set_buttons_state(tk.DISABLED) # Désactiver pendant l'installation
                self.btn_check_ota_updates.config(text="Installation MàJ...")
                
                # Lancer l'installation dans un thread
                thread = threading.Thread(target=self._perform_model_installations, 
                                          args=(models_to_install,), daemon=True)
                thread.start()
            else:
                self.log_message("[GUI] L'utilisateur a refusé d'installer les mises à jour.")
                self._set_buttons_state(tk.NORMAL) # Réactiver si refus
        else: # Juste un message informatif (pas de MàJ)
            messagebox.showinfo("Mises à Jour Modèles IA", full_message)
            self._set_buttons_state(tk.NORMAL) # Réactiver


    def _perform_model_installations(self, models_to_install):
        all_successful = True
        for model_name, server_info in models_to_install:
            self.log_message(f"[GUI] Installation de la mise à jour pour {model_name}...")
            success = download_and_install_model(model_name, server_info, log_callback=self.log_message)
            if success:
                self.log_message(f"[GUI] Mise à jour de {model_name} réussie !")
            else:
                self.log_message(f"[GUI] ÉCHEC de la mise à jour de {model_name}.")
                all_successful = False
        
        # Message final après toutes les tentatives
        if self.root.winfo_exists():
            if all_successful and models_to_install: # S'il y avait qqc à installer et que tout a réussi
                self.root.after(0, lambda: messagebox.showinfo("Mises à Jour IA", "Tous les modèles sélectionnés ont été mis à jour avec succès !"))
            elif models_to_install : # S'il y avait qqc à installer mais au moins un échec
                 self.root.after(0, lambda: messagebox.showwarning("Mises à Jour IA", "Certaines mises à jour de modèles ont échoué. Consultez les logs."))
            # Si models_to_install était vide, aucun message n'est nécessaire ici car _show_ota_results l'a déjà fait.
        
        if self.root.winfo_exists():
            self.root.after(0, self._set_buttons_state, tk.NORMAL) # Réactiver tous les boutons

if __name__ == '__main__':
    app_window = tk.Tk()
    gui_app = AICompressGUI(app_window)
    app_window.mainloop()
EOF
echo "aicompress_gui.py mis à jour."
echo ""

echo "--------------------------------------------------------------------------"
echo "Configuration pour les Mises à Jour OTA terminée."
echo "ACTIONS IMPORTANTES POUR TESTER :"
echo "1. MODIFIEZ L'URL 'MODEL_INFO_URL' dans 'aicompress/ota_updater.py'."
echo "   Elle doit pointer vers un fichier 'model_info.json' que vous hébergez."
echo "2. Créez et hébergez votre fichier 'model_info.json'. Exemple de contenu :"
echo '   {'
echo '       "text_classifier": {'
echo '           "latest_version": "1.1", '
echo '           "description": "Modèle de test amélioré.",'
echo '           "url": "URL_DIRECTE_VERS_VOTRE_text_classifier_v1.1.joblib",'
echo '           "checksum_sha256": "SHA256_DU_FICHIER_MODELE_ICI"'
echo '       }'
echo '   }'
echo "3. Hébergez le fichier modèle (ex: text_classifier_v1.1.joblib) à l'URL spécifiée."
echo "4. Assurez-vous que votre AICompress local a une version 'plus ancienne' du modèle"
echo "   (ex: supprimez ~/.aicompress/models/text_classifier_config.json pour qu'il recrée la v1.0)."
echo "5. Lancez 'python aicompress_gui.py' et utilisez le bouton 'Vérifier MàJ Modèles IA'."
echo "--------------------------------------------------------------------------"

exit 0