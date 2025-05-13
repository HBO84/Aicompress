#!/bin/bash

echo "--------------------------------------------------------------------------"
echo "Script de préparation pour l'IA 'Chef d'Orchestre' (Étape 1)"
echo "Ce script va :"
echo "  1. Mettre à jour 'aicompress/ai_analyzer.py' pour l'extraction de features."
echo "  2. Créer un squelette pour 'create_decision_dataset.py'."
echo ""
echo "ATTENTION : Une sauvegarde de 'aicompress/ai_analyzer.py' est recommandée."
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

# Étape 1: Mise à jour de aicompress/ai_analyzer.py
echo_section_header "Étape 1: Mise à jour de aicompress/ai_analyzer.py"
cat << 'EOF' > aicompress/ai_analyzer.py
# aicompress/ai_analyzer.py
import os
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import numpy as np
import joblib
import math # Pour l'entropie
from collections import Counter # Pour l'entropie

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
current_text_classifier_version = DEFAULT_TEXT_CLASSIFIER_VERSION

def _default_log_analyzer(message): # Logueur spécifique à ce module pour éviter conflit de nom
    print(message)

def ensure_model_dir_exists(log_callback=_default_log_analyzer):
    try:
        os.makedirs(MODELS_SPECIFIC_DIR, exist_ok=True)
    except Exception as e:
        log_callback(f"[AI_ANALYZER] Erreur création dossier modèle {MODELS_SPECIFIC_DIR}: {e}")


def get_local_text_classifier_version(log_callback=_default_log_analyzer):
    global current_text_classifier_version
    ensure_model_dir_exists(log_callback=log_callback)
    if os.path.exists(TEXT_CLASSIFIER_CONFIG_PATH):
        try:
            with open(TEXT_CLASSIFIER_CONFIG_PATH, 'r') as f: config = json.load(f)
            current_text_classifier_version = config.get("version", DEFAULT_TEXT_CLASSIFIER_VERSION)
        except Exception as e:
            log_callback(f"[AI_ANALYZER] Erreur lecture config version: {e}. Défaut {DEFAULT_TEXT_CLASSIFIER_VERSION}.")
            current_text_classifier_version = DEFAULT_TEXT_CLASSIFIER_VERSION
    else: 
        current_text_classifier_version = DEFAULT_TEXT_CLASSIFIER_VERSION
    return current_text_classifier_version

def save_local_text_classifier_version(version_str, log_callback=_default_log_analyzer):
    global current_text_classifier_version
    ensure_model_dir_exists(log_callback=log_callback)
    try:
        with open(TEXT_CLASSIFIER_CONFIG_PATH, 'w') as f: json.dump({"version": version_str}, f)
        current_text_classifier_version = version_str
        log_callback(f"[AI_ANALYZER] Version modèle texte local MàJ: {version_str}")
    except Exception as e: log_callback(f"[AI_ANALYZER] Erreur sauvegarde config version: {e}")

def train_and_save_text_classifier(version_to_save=DEFAULT_TEXT_CLASSIFIER_VERSION, log_callback=_default_log_analyzer):
    global text_classifier_model
    ensure_model_dir_exists(log_callback=log_callback)
    log_callback(f"[AI_ANALYZER] Entraînement nouveau modèle texte (v {version_to_save})...")
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
        log_callback(f"[AI_ANALYZER] Modèle texte sauvegardé: {TEXT_CLASSIFIER_MODEL_PATH}")
        save_local_text_classifier_version(version_to_save, log_callback=log_callback)
        text_classifier_model = model
    except Exception as e:
        log_callback(f"[AI_ANALYZER] Erreur sauvegarde modèle texte: {e}"); text_classifier_model = model
    return text_classifier_model

def load_text_classifier(force_retrain=False, log_callback=_default_log_analyzer):
    global text_classifier_model
    ensure_model_dir_exists(log_callback=log_callback)
    current_version = get_local_text_classifier_version(log_callback=log_callback) # Met à jour la globale aussi
    log_callback(f"[AI_ANALYZER] Version locale modèle texte (au chargement): {current_version}")

    if not force_retrain and os.path.exists(TEXT_CLASSIFIER_MODEL_PATH):
        try:
            log_callback(f"[AI_ANALYZER] Chargement modèle texte: {TEXT_CLASSIFIER_MODEL_PATH}")
            text_classifier_model = joblib.load(TEXT_CLASSIFIER_MODEL_PATH)
            log_callback("[AI_ANALYZER] Modèle texte chargé."); return text_classifier_model
        except Exception as e: log_callback(f"[AI_ANALYZER] Erreur chargement modèle texte: {e}. Réentraînement.")
    return train_and_save_text_classifier(current_version, log_callback=log_callback)

text_classifier_model = load_text_classifier(log_callback=_default_log_analyzer)

def predict_text_content_type(text_content_str, log_callback=_default_log_analyzer):
    global text_classifier_model
    if text_classifier_model is None:
        log_callback("[AI_ANALYZER] Modèle texte non dispo. Tentative chargement/entraînement.")
        load_text_classifier(log_callback=log_callback)
        if text_classifier_model is None: return "unknown_text_model_unavailable"
    if not isinstance(text_content_str, str) or not text_content_str.strip(): return "unknown_text_or_empty"    
    try: return text_classifier_model.predict([text_content_str])[0]
    except Exception as e: log_callback(f"[AI_ANALYZER] Erreur prédiction type texte: {e}"); return "unknown_text_prediction_error"

def analyze_file_content(file_path, log_callback=_default_log_analyzer):
    # ... (Cette fonction reste comme avant, elle retourne le type de fichier string)
    # ... (Copiez ici votre version la plus récente et fonctionnelle de analyze_file_content)
    # Pour la concision, je ne la remets pas en entier, mais elle est nécessaire.
    # Assurez-vous qu'elle utilise bien son log_callback si elle en a un, sinon _default_log_analyzer
    if not os.path.exists(file_path): return "file_not_found"
    if os.path.getsize(file_path) == 0: return "empty_file"
    file_type_by_magic = "unknown"
    if PYTHON_MAGIC_AVAILABLE:
        try:
            mime_type = magic.from_file(file_path, mime=True)
            if mime_type: 
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
            else: file_type_by_magic = "unknown_mime_none"
            text_like_types = ["plain_text", "python_script", "generic_text", "html_document", "xml_document", "json_data", "css_stylesheet"]
            if file_type_by_magic in text_like_types:
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f: content_sample = f.read(2048) 
                    if file_type_by_magic == "python_script": return "python_script"
                    refined_text_type = predict_text_content_type(content_sample, log_callback=log_callback)
                    return refined_text_type if refined_text_type and "unknown" not in refined_text_type else file_type_by_magic
                except Exception: return file_type_by_magic 
            return file_type_by_magic
        except Exception as e: log_callback(f"[AI_ANALYZER] Avertissement: Erreur python-magic {file_path}: {e}")
    try: 
        with open(file_path, 'rb') as f_rb: content_sample_bytes = f_rb.read(512)
        if b'\0' in content_sample_bytes: return "unknown_binary_heuristic_null"
        try:
            content_sample_text = content_sample_bytes.decode('utf-8', errors='ignore') 
            non_printable = sum(1 for char_code in content_sample_bytes[:100] if not (32 <= char_code <= 126 or char_code in [9,10,13]))
            if non_printable > 10: return "unknown_binary_heuristic_nonprint"
            text_type = predict_text_content_type(content_sample_text, log_callback=log_callback)
            return text_type if text_type and "unknown" not in text_type else "generic_text_heuristic"
        except UnicodeDecodeError: return "unknown_binary_read_error"
    except Exception: return "error_during_fallback_analysis"


# --- NOUVELLES FONCTIONS POUR L'EXTRACTION DE FEATURES ---
def calculate_shannon_entropy(file_path, sample_size=10240, log_callback=_default_log_analyzer):
    """Calcule l'entropie de Shannon d'un échantillon du fichier."""
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        return 0.0
    try:
        with open(file_path, 'rb') as f:
            data = f.read(sample_size)
        if not data:
            return 0.0
        
        byte_counts = Counter(data)
        total_bytes = len(data)
        entropy = 0.0
        for count in byte_counts.values():
            probability = count / total_bytes
            entropy -= probability * math.log2(probability)
        # Normaliser l'entropie par log2(256) = 8 pour avoir une valeur entre 0 et 1 (si distribution uniforme)
        # Une entropie élevée (proche de 8 avant normalisation, ou proche de 1 après) indique des données plus aléatoires/compressées.
        return entropy / 8.0 
    except Exception as e:
        log_callback(f"[AI_ANALYZER] Erreur calcul entropie pour {file_path}: {e}")
        return 0.0 # Ou une autre valeur pour indiquer une erreur

def get_file_features(file_path, log_callback=_default_log_analyzer):
    """Extrait un dictionnaire de caractéristiques pour un fichier."""
    if not os.path.exists(file_path):
        return {"type": "file_not_found", "size": 0, "entropy": 0.0, "error": True}
    
    file_type = analyze_file_content(file_path, log_callback=log_callback) # Utilise la fonction existante
    file_size = os.path.getsize(file_path)
    # Lire un échantillon plus grand pour l'entropie si le fichier est gros
    sample_size_for_entropy = min(file_size, 1024 * 100) # Max 100KB pour l'entropie
    file_entropy = calculate_shannon_entropy(file_path, sample_size=sample_size_for_entropy, log_callback=log_callback)
    
    features = {
        "type": file_type,
        "size_bytes": file_size,
        "entropy_normalized": round(file_entropy, 4) # Arrondir pour la propreté
    }
    log_callback(f"[AI_ANALYZER] Features pour {os.path.basename(file_path)}: {features}")
    return features

if __name__ == '__main__': # Section de test du module
    print(f"[AI_ANALYZER TEST] python-magic: {PYTHON_MAGIC_AVAILABLE}")
    print(f"[AI_ANALYZER TEST] Modèle texte initial (v {get_local_text_classifier_version()}): {'Oui' if text_classifier_model else 'Non'}")
    
    # Créer des fichiers de test factices
    ensure_model_dir_exists() # Juste pour s'assurer que le dossier .aicompress est là
    test_dir = "temp_analyzer_test_files"
    os.makedirs(test_dir, exist_ok=True)
    
    file1_path = os.path.join(test_dir, "test1.txt")
    with open(file1_path, "w") as f: f.write("this is a simple english text file with some repetition repetition.")
    
    file2_path = os.path.join(test_dir, "test2.py")
    with open(file2_path, "w") as f: f.write("import os\ndef hello():\n  print('world')\n# comment")

    file3_path = os.path.join(test_dir, "test3_random.bin") # Fichier binaire plus aléatoire
    with open(file3_path, "wb") as f: f.write(os.urandom(1024))

    print(f"\n--- Test de get_file_features ---")
    for f_path in [file1_path, file2_path, file3_path]:
        if os.path.exists(f_path):
            features = get_file_features(f_path)
            print(f"  - Fichier: {os.path.basename(f_path)}, Features: {features}")
        else:
            print(f"  - Fichier de test {f_path} non trouvé.")

    # Nettoyage
    if os.path.exists(file1_path): os.remove(file1_path)
    if os.path.exists(file2_path): os.remove(file2_path)
    if os.path.exists(file3_path): os.remove(file3_path)
    if os.path.exists(test_dir): shutil.rmtree(test_dir)
    print("\nTests de ai_analyzer terminés.")
EOF
echo "aicompress/ai_analyzer.py mis à jour avec extraction de features."
echo ""

# Étape 2: Création du squelette pour create_decision_dataset.py
echo_section_header "Étape 2: Création du squelette pour create_decision_dataset.py"
cat << 'EOF' > create_decision_dataset.py
# create_decision_dataset.py
# Script pour générer un jeu de données pour entraîner l'IA "Chef d'Orchestre"

import os
import glob
import csv
import time
import json # Pour charger les métadonnées AIC si on compresse via core.py
import zipfile # Pour DEFLATE et STORED si on le fait manuellement
import bz2
import lzma
import numpy as np # Pourrait être utile pour le moteur AE

# Importer les modules nécessaires d'AICompress
from aicompress.ai_analyzer import get_file_features, _default_log_analyzer as log # Utiliser son logger
# Pour le moteur AE, il faudrait charger les modèles et les fonctions de prétraitement
# from aicompress.core import (preprocess_image_for_cifar10_ae, cifar10_color_encoder_loaded, 
#                              ensure_cifar10_color_models_loaded, PIL_AVAILABLE, KERAS_AVAILABLE)

# --- Configuration ---
# Dossier contenant les fichiers sources variés pour créer le dataset
SOURCE_FILES_DIR = "./dataset_source_files/"  # À CRÉER ET REMPLIR PAR L'UTILISATEUR
OUTPUT_CSV_FILE = "compression_decision_dataset.csv"

# Liste des compresseurs et de leurs paramètres à tester
# Format: (nom_methode, fonction_compression, dict_params_si_necessaire)
# La fonction_compression prendra (data_bytes, params) et retournera (compressed_data_bytes)
# ou None si échec.

COMPRESSION_METHODS = [
    ("STORED", lambda data, p: data, {}), # Pas de compression, retourne data originale
    ("DEFLATE_L1", lambda data, p: zipfile.compress(data, compresslevel=1) if hasattr(zipfile, 'compress') else zlib_deflate(data,1), {}), # Nécessite Python 3.7+ pour zipfile.compress
    ("DEFLATE_L6", lambda data, p: zipfile.compress(data, compresslevel=6) if hasattr(zipfile, 'compress') else zlib_deflate(data,6), {}),
    ("DEFLATE_L9", lambda data, p: zipfile.compress(data, compresslevel=9) if hasattr(zipfile, 'compress') else zlib_deflate(data,9), {}),
    ("BZIP2_L9", lambda data, p: bz2.compress(data, compresslevel=9), {}),
    ("LZMA_P0", lambda data, p: lzma.compress(data, format=lzma.FORMAT_XZ, preset=0), {}), # Preset 0-9
    ("LZMA_P6", lambda data, p: lzma.compress(data, format=lzma.FORMAT_XZ, preset=6), {}),
    ("LZMA_P9", lambda data, p: lzma.compress(data, format=lzma.FORMAT_XZ, preset=9), {}),
    # ("AE_CIFAR10_COLOR", compress_with_ae_cifar10, {}), # À implémenter si on inclut l'AE
]

# Helper pour DEFLATE si zipfile.compress n'est pas dispo (Python < 3.7)
# ou pour une compression en mémoire plus directe.
import zlib
def zlib_deflate(data_bytes, level):
    # zlib.compress attend un niveau de 0 à 9.
    # zipfile. níveis DEFLATE sont 0-9.
    # zlib: 0=pas de compression, 1=meilleure vitesse, 9=meilleure compression. -1=défaut (~6)
    # On va mapper: zipfile L1->zlib L1, L6->zlib L6, L9->zlib L9
    return zlib.compress(data_bytes, level=level)


# --- Fonction pour compresser avec l'Autoencodeur (Squelette) ---
# Doit être adaptée pour charger le modèle, prétraiter, encoder, quantifier
# def compress_with_ae_cifar10(data_bytes, params, file_path_for_preprocessing=None):
#     log(f"[DATASET_GEN] Tentative AE CIFAR10 pour {file_path_for_preprocessing}...")
#     if not (PIL_AVAILABLE and KERAS_AVAILABLE): return None
#     if not cifar10_color_encoder_loaded:
#         if not ensure_cifar10_color_models_loaded(log_callback=log): return None
#
#     # Pour utiliser l'AE, il faut le chemin du fichier original pour le prétraitement
#     if not file_path_for_preprocessing:
#         log("[DATASET_GEN] Chemin fichier original requis pour prétraitement AE.")
#         return None
#
#     try:
#         image_preprocessed, _ = preprocess_image_for_cifar10_ae(file_path_for_preprocessing, log_callback=log)
#         if image_preprocessed is None: return None
#
#         code_latent_float = cifar10_color_encoder_loaded.predict(image_preprocessed)
#         min_val, max_val = np.min(code_latent_float), np.max(code_latent_float)
#         
#         quant_uint8 = np.array([], dtype=np.uint8)
#         if max_val > min_val: norm_l=(code_latent_float-min_val)/(max_val-min_val); quant_uint8=(norm_l*255).astype(np.uint8)
#         elif max_val!=0: quant_uint8=np.full_like(code_latent_float,fill_value=np.clip(np.round(max_val),0,255),dtype=np.uint8)
#         
#         return quant_uint8.tobytes() # Retourne les octets du latent quantifié
#     except Exception as e:
#         log(f"[DATASET_GEN] Erreur compression AE CIFAR10: {e}")
#         return None

# --- Fonction Principale pour Générer le Dataset ---
def generate_dataset():
    log(f"Début de la génération du jeu de données : {OUTPUT_CSV_FILE}")
    
    if not os.path.exists(SOURCE_FILES_DIR) or not os.listdir(SOURCE_FILES_DIR):
        log(f"ERREUR: Le dossier source '{SOURCE_FILES_DIR}' est vide ou n'existe pas.")
        log(f"Veuillez créer ce dossier et y placer des fichiers variés pour l'entraînement.")
        return

    header = ["file_name", "file_type", "original_size_bytes", "entropy_normalized", 
              "best_method", "best_compressed_size_bytes", "time_best_ms"]
    # Ajouter des colonnes pour chaque méthode testée (taille et temps)
    for method_name, _, _ in COMPRESSION_METHODS:
        header.extend([f"{method_name}_size_bytes", f"{method_name}_time_ms"])
    # Si on ajoute l'AE, il faut une colonne pour lui aussi.

    with open(OUTPUT_CSV_FILE, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)

        source_filepaths = glob.glob(os.path.join(SOURCE_FILES_DIR, "*")) # Prend tous les fichiers
        log(f"Trouvé {len(source_filepaths)} fichiers sources à traiter.")

        for filepath in source_filepaths:
            if not os.path.isfile(filepath): continue # Ignorer les sous-dossiers pour l'instant

            file_name = os.path.basename(filepath)
            log(f"\nTraitement du fichier : {file_name}")

            features = get_file_features(filepath, log_callback=log)
            if features.get("error"):
                log(f"  Erreur extraction features pour {file_name}. Ignoré.")
                continue

            original_size = features["size_bytes"]
            if original_size == 0:
                log(f"  Fichier {file_name} est vide. Ignoré.")
                continue

            try:
                with open(filepath, 'rb') as f:
                    original_data = f.read()
            except Exception as e:
                log(f"  Erreur lecture {file_name}: {e}. Ignoré.")
                continue

            row_data = [file_name, features["type"], original_size, features["entropy_normalized"]]
            results_for_file = [] # (method_name, compressed_size, time_ms)

            for method_name, compress_func, params in COMPRESSION_METHODS:
                log(f"  Test méthode: {method_name}...")
                compressed_data = None
                comp_size = original_size # Par défaut (pour STORED ou si échec)
                
                start_time = time.perf_counter()
                try:
                    # if method_name == "AE_CIFAR10_COLOR":
                    #     # La fonction AE a besoin du chemin du fichier pour le prétraitement
                    #     compressed_data = compress_func(None, params, file_path_for_preprocessing=filepath)
                    # else:
                    compressed_data = compress_func(original_data, params)
                    
                    if compressed_data is not None:
                        comp_size = len(compressed_data)
                    else: # La fonction de compression a échoué
                        log(f"    Échec compression {method_name}. Utilisation taille originale.")
                        comp_size = original_size # Marquer comme échec
                except Exception as e_comp:
                    log(f"    ERREUR méthode {method_name}: {e_comp}")
                    comp_size = original_size # Marquer comme échec
                
                end_time = time.perf_counter()
                time_ms = (end_time - start_time) * 1000
                
                results_for_file.append((method_name, comp_size, time_ms))
                log(f"    {method_name}: Taille={comp_size} octets, Temps={time_ms:.2f} ms")

            # Déterminer la meilleure méthode (plus petite taille compressée)
            if not results_for_file:
                log(f"  Aucun résultat de compression pour {file_name}. Ignoré.")
                continue

            # Filtrer les résultats où comp_size est original_size si la méthode n'est pas STORED (échecs)
            # Sauf si STORED est effectivement le meilleur.
            # Ou plus simple : on prend juste le min. Si plusieurs méthodes donnent la même taille min, on prend la plus rapide.
            
            best_method_name = "N/A"
            best_size = float('inf')
            best_time = float('inf')

            for r_method, r_size, r_time in results_for_file:
                if r_size < best_size:
                    best_size = r_size
                    best_time = r_time
                    best_method_name = r_method
                elif r_size == best_size: # Si tailles égales, choisir le plus rapide
                    if r_time < best_time:
                        best_time = r_time
                        best_method_name = r_method
            
            row_data.extend([best_method_name, best_size, round(best_time, 2)])
            # Ajouter les tailles et temps de chaque méthode
            for _, r_size, r_time in results_for_file:
                row_data.extend([r_size, round(r_time, 2)])
            
            writer.writerow(row_data)
            log(f"  Meilleure méthode pour {file_name}: {best_method_name} (Taille: {best_size}, Temps: {best_time:.2f} ms)")

    log(f"\nJeu de données sauvegardé dans : {OUTPUT_CSV_FILE}")

if __name__ == '__main__':
    # Créer le dossier source s'il n'existe pas et donner des instructions
    if not os.path.exists(SOURCE_FILES_DIR):
        os.makedirs(SOURCE_FILES_DIR, exist_ok=True)
        print(f"Dossier source '{SOURCE_FILES_DIR}' créé.")
        print(f"Veuillez y placer une grande variété de fichiers (texte, code, images, binaires, etc.)")
        print(f"pour générer un jeu de données d'entraînement pertinent.")
        print(f"Modifiez la variable TRAIN_FILES_PATH_PATTERN dans ce script si vous voulez cibler des types spécifiques.")
        print(f"Une fois les fichiers en place, relancez ce script.")
    elif not os.listdir(SOURCE_FILES_DIR):
         print(f"Le dossier source '{SOURCE_FILES_DIR}' est vide.")
         print(f"Veuillez y placer des fichiers avant de lancer la génération du dataset.")
    else:
        generate_dataset()
EOF
echo "Squelette pour create_decision_dataset.py créé à la racine du projet."
echo ""

echo "--------------------------------------------------------------------------"
echo "Préparation pour l'IA 'Chef d'Orchestre' terminée."
echo "PROCHAINES ÉTAPES POUR VOUS :"
echo "1. Créez un dossier nommé 'dataset_source_files' à la racine de votre projet AICompressProject."
echo "2. Remplissez ce dossier avec une GRANDE VARIÉTÉ de types de fichiers (texte, code .py, .c, .java, JSON, XML,"
echo "   petites images PNG, JPG, BMP, peut-être des petits binaires exécutables, etc.)."
echo "   Plus les fichiers sont variés et nombreux, meilleur sera le jeu de données."
echo "3. (Optionnel) Modifiez la variable 'TRAIN_FILES_PATH_PATTERN' dans 'create_decision_dataset.py'"
echo "   si vous voulez changer la manière dont les fichiers sont sélectionnés dans 'dataset_source_files'."
echo "4. Exécutez 'python create_decision_dataset.py' pour générer le fichier CSV."
echo "   Cela peut prendre du temps en fonction du nombre et de la taille des fichiers sources."
echo "5. Une fois le fichier 'compression_decision_dataset.csv' généré, nous pourrons l'utiliser"
echo "   pour entraîner un modèle de classification qui choisira la meilleure méthode de compression."
echo "--------------------------------------------------------------------------"

exit 0