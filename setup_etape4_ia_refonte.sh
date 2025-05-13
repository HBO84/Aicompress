#!/bin/bash

echo "--------------------------------------------------------------------------"
echo "Script de mise à jour pour l'Étape 4 : IA au centre des décisions"
echo "Ce script va installer libmagic1, python-magic et mettre à jour"
echo "vos fichiers aicompress/ai_analyzer.py et aicompress/core.py."
echo ""
echo "ATTENTION : Une sauvegarde de votre travail est recommandée."
echo "Vous aurez besoin des droits sudo pour installer libmagic1."
echo "Assurez-vous que votre environnement virtuel Python est activé."
echo "--------------------------------------------------------------------------"
echo ""
read -p "Appuyez sur Entrée pour continuer, ou Ctrl+C pour annuler..."

# Vérifier si nous sommes (probablement) dans AICompressProject
if [ ! -d "aicompress" ] || [ ! -f "main.py" ]; then
    echo "ERREUR : Ce script doit être exécuté depuis la racine de AICompressProject."
    echo "Veuillez vous déplacer dans le bon dossier et réessayer."
    exit 1
fi

# Étape 1: Installation des dépendances pour python-magic
echo ">>> Étape 1: Installation de libmagic1 et python-magic..."
echo "Mise à jour des listes de paquets (nécessite sudo)..."
sudo apt update

echo "Installation de libmagic1 (nécessite sudo)..."
if sudo apt install -y libmagic1; then
    echo "libmagic1 installée avec succès."
else
    echo "ERREUR : L'installation de libmagic1 a échoué."
    echo "Veuillez vérifier les erreurs et l'installer manuellement si besoin."
    exit 1
fi

echo "Installation de python-magic (dans l'environnement virtuel actif)..."
if pip install python-magic; then
    echo "python-magic installée avec succès."
else
    echo "ERREUR : L'installation de python-magic a échoué."
    echo "Vérifiez que pip est fonctionnel et que votre environnement virtuel est activé."
    exit 1
fi
echo ""

# Étape 2: Mise à jour de aicompress/ai_analyzer.py
echo ">>> Étape 2: Mise à jour de aicompress/ai_analyzer.py..."
cat << 'EOF' > aicompress/ai_analyzer.py
# aicompress/ai_analyzer.py

import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import numpy as np

try:
    import magic
    PYTHON_MAGIC_AVAILABLE = True
except ImportError:
    PYTHON_MAGIC_AVAILABLE = False
    print("Avertissement (ai_analyzer.py): Bibliothèque 'python-magic' non trouvée. L'identification des types de fichiers binaires sera limitée.")
    print("Installation : pip install python-magic (et libmagic1 sur Debian/Ubuntu)")

# --- Modèle de classification de texte (inchangé pour l'instant) ---
TRAINING_DATA = [
    ("def class import def for if else elif try except finally with as pass return yield", "python_script"), # Labels plus spécifiques
    ("lambda def __init__ self cls args kwargs decorator", "python_script"),
    ("import os sys json re math numpy pandas sklearn", "python_script"),
    ("the a is are was were will be and or but for of to in on at", "english_text"),
    ("sentence paragraph word letter text document write read speak", "english_text"),
    ("hello world good morning example another however therefore because", "english_text"),
]
texts, labels = zip(*TRAINING_DATA)
text_classifier_model = make_pipeline(CountVectorizer(ngram_range=(1, 2)), MultinomialNB())
# print("Entraînement du modèle de classification de texte...") # Peut être mis sous condition si déjà entraîné
text_classifier_model.fit(texts, labels)
# print("Modèle de classification de texte entraîné.")

def predict_text_content_type(text_content_str):
    """Prédit le type de contenu pour une chaîne de caractères textuelle."""
    if not isinstance(text_content_str, str) or not text_content_str.strip():
        return "unknown_text_or_empty"
    
    prediction = text_classifier_model.predict([text_content_str])
    # probabilities = text_classifier_model.predict_proba([text_content_str]) # Si besoin de confiance
    # proba_percent = np.max(probabilities) * 100
    return prediction[0] # Retourne ex: "python_script", "english_text"

def analyze_file_content(file_path):
    """
    Analyse un fichier pour déterminer son type de contenu de manière plus détaillée.
    Retourne une chaîne décrivant le type (ex: 'jpeg_image', 'python_script', 'zip_archive', 'unknown_binary').
    """
    if not os.path.exists(file_path):
        return "file_not_found"
    if os.path.getsize(file_path) == 0:
        return "empty_file"

    file_type_by_magic = "unknown"
    # mime_type = "unknown" # Non utilisé directement en dehors de la condition

    # 1. Utiliser python-magic pour une identification basée sur les magic numbers
    if PYTHON_MAGIC_AVAILABLE:
        try:
            mime_type = magic.from_file(file_path, mime=True)
            # Simplifier certains types MIME courants
            if mime_type.startswith('image/jpeg'):
                file_type_by_magic = "jpeg_image"
            elif mime_type.startswith('image/png'):
                file_type_by_magic = "png_image"
            elif mime_type.startswith('image/gif'):
                file_type_by_magic = "gif_image"
            elif mime_type.startswith('application/pdf'):
                file_type_by_magic = "pdf_document"
            elif mime_type.startswith('application/zip'):
                file_type_by_magic = "zip_archive"
            elif mime_type.startswith('application/x-rar-compressed'):
                file_type_by_magic = "rar_archive"
            elif mime_type.startswith('application/x-7z-compressed'):
                file_type_by_magic = "7z_archive"
            elif mime_type.startswith('application/x-tar'):
                 file_type_by_magic = "tar_archive" # Tar sans compression
            elif mime_type.startswith('application/gzip') or mime_type.startswith('application/x-gzip'):
                file_type_by_magic = "gzip_archive"
            elif mime_type.startswith('application/x-bzip2'):
                file_type_by_magic = "bzip2_archive"
            elif mime_type.startswith('application/x-xz'):
                file_type_by_magic = "xz_archive"
            elif mime_type.startswith('audio/mpeg'):
                file_type_by_magic = "mp3_audio" # ou mpeg_audio
            elif mime_type.startswith('audio/'):
                file_type_by_magic = mime_type.replace('audio/', '').split(';')[0] + "_audio"
            elif mime_type.startswith('video/'):
                file_type_by_magic = mime_type.replace('video/', '').split(';')[0] + "_video"
            elif mime_type.startswith('text/x-python'):
                file_type_by_magic = "python_script"
            elif mime_type.startswith('text/html'):
                file_type_by_magic = "html_document"
            elif mime_type.startswith('text/xml'):
                file_type_by_magic = "xml_document"
            elif mime_type.startswith('text/plain'):
                file_type_by_magic = "plain_text"
            elif mime_type.startswith('text/'):
                file_type_by_magic = "generic_text"
            elif mime_type.startswith('application/octet-stream'):
                file_type_by_magic = "generic_binary"
            else:
                file_type_by_magic = mime_type.replace('/', '_').split(';')[0]
            
            # Si c'est identifié comme plain_text ou un script par magic, on peut essayer notre classifieur de texte
            if file_type_by_magic in ["plain_text", "python_script", "generic_text", "html_document", "xml_document"]:
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content_sample = f.read(2048) 
                    
                    # Si magic a déjà bien identifié (ex: python_script), on peut le garder ou affiner
                    if file_type_by_magic == "python_script":
                         return "python_script"

                    refined_text_type = predict_text_content_type(content_sample)
                    if refined_text_type and "unknown" not in refined_text_type:
                        return refined_text_type 
                    else: 
                        return file_type_by_magic # Retourner le type de magic si le classifieur n'est pas sûr
                except Exception:
                    return file_type_by_magic 
            
            return file_type_by_magic

        except Exception as e: 
            print(f"Avertissement: Erreur avec python-magic pour {file_path}: {e}")
            # Fallback si python-magic échoue

    # 2. Fallback : Si python-magic n'est pas dispo ou a échoué gravement
    try:
        with open(file_path, 'rb') as f_rb: # Lire en binaire pour détecter les nulls
            content_sample_bytes = f_rb.read(512)
        
        if b'\0' in content_sample_bytes: # Heuristique pour binaire
            return "unknown_binary_heuristic_null"
        
        # Essayer de décoder en texte
        try:
            content_sample_text = content_sample_bytes.decode('utf-8')
            # Heuristique de caractères non imprimables (plus robuste)
            non_printable_chars = sum(1 for char_code in content_sample_bytes[:100] if not (32 <= char_code <= 126 or char_code in [9, 10, 13]))
            if non_printable_chars > 10: # Seuil arbitraire
                 return "unknown_binary_heuristic_nonprint"

            text_type = predict_text_content_type(content_sample_text)
            if text_type and "unknown" not in text_type:
                return text_type
            else:
                return "generic_text_heuristic"
        except UnicodeDecodeError:
             return "unknown_binary_read_error" # Ne peut pas être lu comme UTF-8 -> binaire
    except Exception:
        return "error_during_fallback_analysis"

if __name__ == '__main__':
    print(f"python-magic disponible: {PYTHON_MAGIC_AVAILABLE}")
    print("\n--- Test de l'analyseur de contenu de fichier (avec python-magic si dispo) ---")
    
    # Créer des fichiers de test factices si python-magic n'est pas là pour un test minimal
    if not os.path.exists("test_sample.jpg"):
        with open("test_sample.jpg", "wb") as f: f.write(b'\xff\xd8\xff\xe0\x00\x10JFIF') # Magic bytes pour JPG
    if not os.path.exists("test_sample.png"):
        with open("test_sample.png", "wb") as f: f.write(b'\x89PNG\r\n\x1a\n') # Magic bytes pour PNG
    if not os.path.exists("test_sample.zip"):
        with open("test_sample.zip", "wb") as f: f.write(b'PK\x03\x04\x14\x00\x00\x00\x00\x00') # Magic bytes pour ZIP
    if not os.path.exists("test_sample.py"):
        with open("test_sample.py", "w") as f: f.write("print('hello from test_sample.py')\n# python")
    if not os.path.exists("test_sample.txt"):
        with open("test_sample.txt", "w") as f: f.write("This is a test text from test_sample.txt.")

    test_files_list = ["test_sample.jpg", "test_sample.png", "test_sample.zip", "test_sample.py", "test_sample.txt"]
    if os.path.exists("aicompress/ai_analyzer.py"): # Teste lui-même s'il existe
        test_files_list.append("aicompress/ai_analyzer.py")
    
    for filename_to_test in test_files_list:
        if os.path.exists(filename_to_test):
            analysis_result_test = analyze_file_content(filename_to_test)
            print(f"Fichier: {filename_to_test:<30} -> Type Prédit: {analysis_result_test}")
        else:
            print(f"Fichier de test {filename_to_test} non trouvé.")

    # Nettoyer les fichiers de test factices
    # for f_to_clean in ["test_sample.jpg", "test_sample.png", "test_sample.zip", "test_sample.py", "test_sample.txt"]:
    #     if os.path.exists(f_to_clean):
    #         os.remove(f_to_clean)
EOF
echo "aicompress/ai_analyzer.py mis à jour."
echo ""

# Étape 3: Mise à jour de aicompress/core.py
echo ">>> Étape 3: Mise à jour de aicompress/core.py..."
cat << 'EOF' > aicompress/core.py
# aicompress/core.py

import os
import zipfile
import shutil
import json 

# Importez les fonctions de votre analyseur IA
try:
    from .ai_analyzer import analyze_file_content
    AI_ANALYZER_AVAILABLE = True
    print("AI Analyzer importé avec succès depuis aicompress.core.")
except ImportError as e:
    print(f"Avertissement depuis core.py: AI Analyzer (ai_analyzer.py) non trouvé ou erreur à l'import: {e}")
    AI_ANALYZER_AVAILABLE = False
    def analyze_file_content(file_path): return "ai_analyzer_unavailable"

# Gestion de rarfile
try:
    import rarfile
    RARFILE_AVAILABLE = True
except ImportError:
    RARFILE_AVAILABLE = False
    print("Avertissement : La bibliothèque 'rarfile' n'est pas installée. La décompression RAR ne sera pas disponible.")

# ---- Définitions pour notre format ----
METADATA_FILENAME = "aicompress_metadata.json"
DEFAULT_AIC_EXTENSION = ".aic"

# PRECOMPRESSED_EXTENSIONS N'EST PLUS UTILISÉE ICI

def get_compression_settings(analysis_result):
    """
    Détermine le type de compression et le niveau basé UNIQUEMENT sur l'analyse IA.
    Retourne (compression_type, compress_level).
    """
    print(f"Décision de compression basée sur l'analyse IA: '{analysis_result}'")

    # Règle 1: Types de fichiers typiquement déjà bien compressés ou pour lesquels DEFLATE n'est pas idéal
    # ou formats qui ne bénéficient pas de la recompression DEFLATE.
    if analysis_result in [
        "jpeg_image", "png_image", "gif_image", "webp_image", # Images souvent compressées
        "mp3_audio", "aac_audio", "ogg_vorbis_audio", "opus_audio", "flac_audio", # Audio (flac est lossless mais déjà compressé)
        "mpeg_video", "mp4_video", "webm_video", "mkv_video", "quicktime_video", # Vidéo
        "pdf_document", # Souvent contient des flux compressés
        "gzip_archive", "bzip2_archive", "xz_archive", "lzma_archive", # Archives déjà compressées
        # Note: on laisse les archives conteneurs (zip, rar, 7z, tar) pour une règle plus bas
    ]:
        print(f"Type '{analysis_result}' détecté. Utilisation de ZIP_STORED.")
        return zipfile.ZIP_STORED, None

    # Règle 2: Types d'archives conteneurs (ZIP, RAR, 7z, TAR non compressé)
    # On tente une recompression DEFLATE, car leur contenu interne peut être peu ou pas compressé.
    elif analysis_result in ["zip_archive", "rar_archive", "7z_archive", "tar_archive"]:
        print(f"Type '{analysis_result}' détecté (archive conteneur). Application d'un niveau de compression DEFLATE moyen-doux (3).")
        return zipfile.ZIP_DEFLATED, 3 # Niveau de compromis pour ne pas perdre trop de temps si déjà compressé

    # Règle 3: Types de texte et code
    elif analysis_result in ["python_script", "c_source", "java_source", "javascript_source", 
                             "html_document", "xml_document", "json_data", "css_stylesheet"]: 
        print(f"Type '{analysis_result}' (code/texte structuré) détecté. Utilisation du niveau de compression élevé (9).")
        return zipfile.ZIP_DEFLATED, 9
    elif analysis_result in ["english_text", "plain_text", "generic_text", "generic_text_heuristic"]:
        print(f"Type '{analysis_result}' (texte général) détecté. Utilisation du niveau de compression moyen-élevé (7).")
        return zipfile.ZIP_DEFLATED, 7

    # Règle 4: Binaires non spécifiquement identifiés ou erreurs
    elif "binary" in analysis_result or \
         analysis_result in ["file_not_found", "empty_file", "unknown", 
                              "error_during_analysis", "ai_analyzer_unavailable",
                              "error_during_fallback_analysis"]:
        print(f"Type '{analysis_result}' (binaire non spécifique, inconnu ou erreur d'analyse). Utilisation d'un niveau de compression DEFLATE bas (2).")
        return zipfile.ZIP_DEFLATED, 2 

    # Règle par défaut si aucune autre ne correspond (devrait être rare)
    else:
        print(f"Type '{analysis_result}' non explicitement géré. Utilisation du niveau de compression DEFLATE par défaut (6).")
        return zipfile.ZIP_DEFLATED, 6

def compress_to_aic(input_paths, output_aic_path):
    """
    Compresse un ou plusieurs fichiers/dossiers dans un fichier .aic,
    en modulant la compression basée sur l'analyse IA améliorée.
    """
    print(f"Début de la compression IA-modulée (refonte) vers '{output_aic_path}'...")

    if not output_aic_path.endswith(DEFAULT_AIC_EXTENSION):
        print(f"Avertissement: le fichier de sortie '{output_aic_path}' n'a pas l'extension '{DEFAULT_AIC_EXTENSION}'.")

    processed_items_metadata = [] 

    try:
        with zipfile.ZipFile(output_aic_path, 'w') as zf:
            items_to_archive_details = [] 

            for item_path in input_paths:
                if not os.path.exists(item_path):
                    print(f"Attention : L'élément '{item_path}' n'existe pas et sera ignoré.")
                    processed_items_metadata.append({
                        "original_path": os.path.basename(item_path),
                        "status": "not_found",
                        "analysis": "N/A",
                        "compression_used": "N/A"
                    })
                    continue

                item_basename = os.path.basename(item_path)
                
                if os.path.isfile(item_path):
                    item_analysis_result = "ai_analyzer_unavailable"
                    if AI_ANALYZER_AVAILABLE:
                        item_analysis_result = analyze_file_content(item_path)
                    
                    comp_type, comp_level = get_compression_settings(item_analysis_result)
                    
                    items_to_archive_details.append({
                        "type": "file", 
                        "path": item_path, 
                        "arcname": item_basename,
                        "compression_type": comp_type,
                        "compression_level": comp_level
                    })
                    processed_items_metadata.append({
                        "original_name": item_basename,
                        "type_in_archive": "file",
                        "analysis": item_analysis_result,
                        "size_original_bytes": os.path.getsize(item_path),
                        "compression_used": f"{'STORED' if comp_type == zipfile.ZIP_STORED else 'DEFLATED'}{f' (level {comp_level})' if comp_level is not None and comp_type == zipfile.ZIP_DEFLATED else ''}"
                    })

                elif os.path.isdir(item_path):
                    dir_metadata_entry = {
                        "original_name": item_basename,
                        "type_in_archive": "directory",
                        "analysis": "N/A_directory",
                        "compression_used": "N/A_directory",
                        "files_within": []
                    }
                    for root, _, files in os.walk(item_path):
                        for file_in_dir in files:
                            full_file_path = os.path.join(root, file_in_dir)
                            archive_path_for_file = os.path.join(item_basename, os.path.relpath(full_file_path, item_path))
                            
                            file_specific_analysis = "ai_analyzer_unavailable"
                            if AI_ANALYZER_AVAILABLE:
                                file_specific_analysis = analyze_file_content(full_file_path)
                            
                            comp_type_f, comp_level_f = get_compression_settings(file_specific_analysis)

                            items_to_archive_details.append({
                                "type": "file_in_dir", 
                                "path": full_file_path, 
                                "arcname": archive_path_for_file,
                                "compression_type": comp_type_f,
                                "compression_level": comp_level_f
                            })
                            dir_metadata_entry["files_within"].append({
                                "path_in_archive": archive_path_for_file,
                                "analysis": file_specific_analysis,
                                "size_original_bytes": os.path.getsize(full_file_path),
                                "compression_used": f"{'STORED' if comp_type_f == zipfile.ZIP_STORED else 'DEFLATED'}{f' (level {comp_level_f})' if comp_level_f is not None and comp_type_f == zipfile.ZIP_DEFLATED else ''}"
                            })
                    processed_items_metadata.append(dir_metadata_entry)

            for item_detail in items_to_archive_details:
                if item_detail["compression_type"] == zipfile.ZIP_STORED:
                    zf.write(item_detail["path"], arcname=item_detail["arcname"], compress_type=zipfile.ZIP_STORED)
                else: 
                    zf.write(item_detail["path"], arcname=item_detail["arcname"], compress_type=zipfile.ZIP_DEFLATED, compresslevel=item_detail["compression_level"])
                print(f"Élément '{item_detail['path']}' ajouté (type: {'STORED' if item_detail['compression_type'] == zipfile.ZIP_STORED else 'DEFLATED'}, level: {item_detail['compression_level'] if item_detail['compression_type'] == zipfile.ZIP_DEFLATED else 'N/A'}) sous '{item_detail['arcname']}'.")

            metadata_content = {
                "aicompress_version": "0.3-alpha-ia-deep-analysis", 
                "ia_analyzer_status": "available" if AI_ANALYZER_AVAILABLE else "unavailable",
                "items_details": processed_items_metadata
            }
            metadata_str = json.dumps(metadata_content, indent=4)
            zf.writestr(METADATA_FILENAME, metadata_str)
            print(f"Fichier de métadonnées '{METADATA_FILENAME}' ajouté avec analyses IA détaillées.")
            
            print(f"Compression IA-modulée (refonte) terminée avec succès : '{output_aic_path}'")
            return True

    except FileNotFoundError:
        print(f"Erreur : Un des chemins d'entrée n'a pas été trouvé.")
        return False
    except Exception as e:
        print(f"Une erreur est survenue pendant la compression : {e}")
        import traceback
        traceback.print_exc()
        return False

def decompress_aic(aic_file_path, output_extract_path):
    """
    Décompresse un fichier .aic dans le dossier de destination spécifié.
    """
    print(f"Début de la décompression de '{aic_file_path}' vers '{output_extract_path}'...")

    if not os.path.exists(aic_file_path):
        print(f"Erreur : Le fichier archive '{aic_file_path}' n'existe pas.")
        return False

    try:
        os.makedirs(output_extract_path, exist_ok=True) 

        with zipfile.ZipFile(aic_file_path, 'r') as zf:
            try:
                metadata_str = zf.read(METADATA_FILENAME)
                metadata = json.loads(metadata_str)
                print("Métadonnées de l'archive :") 
                print(json.dumps(metadata, indent=4))
            except KeyError:
                print(f"Avertissement : Fichier de métadonnées '{METADATA_FILENAME}' non trouvé dans l'archive.")
            except json.JSONDecodeError:
                print(f"Avertissement : Impossible de parser les métadonnées.")

            members_to_extract = [member for member in zf.namelist() if member != METADATA_FILENAME]
            if not members_to_extract and METADATA_FILENAME in zf.namelist():
                print("Avertissement: L'archive ne contient que des métadonnées (ou est vide autrement).")
            
            for member in members_to_extract:
                zf.extract(member, path=output_extract_path)
            
            print(f"Décompression terminée avec succès dans '{output_extract_path}'.")
            return True

    except zipfile.BadZipFile:
        print(f"Erreur : '{aic_file_path}' n'est pas un fichier ZIP valide (ou .aic corrompu).")
        return False
    except Exception as e:
        print(f"Une erreur est survenue pendant la décompression : {e}")
        return False

def decompress_rar(rar_file_path, output_extract_path):
    """
    Décompresse un fichier .rar dans le dossier de destination spécifié.
    """
    if not RARFILE_AVAILABLE:
        print("Erreur : La fonctionnalité de décompression RAR n'est pas disponible car 'rarfile' n'a pas pu être importé.")
        return False

    print(f"Début de la décompression RAR de '{rar_file_path}' vers '{output_extract_path}'...")

    if not os.path.exists(rar_file_path):
        print(f"Erreur : Le fichier archive RAR '{rar_file_path}' n'existe pas.")
        return False

    try:
        os.makedirs(output_extract_path, exist_ok=True)
        with rarfile.RarFile(rar_file_path, 'r') as rf:
            rf.extractall(path=output_extract_path)
            print(f"Fichier RAR '{rar_file_path}' extrait avec succès dans '{output_extract_path}'.")
            return True
    except rarfile.NeedFirstVolume:
        print(f"Erreur : Archive RAR multi-volume détectée.") # Message simplifié
        return False
    except rarfile.NotRarFile:
        print(f"Erreur : '{rar_file_path}' n'est pas un fichier RAR valide ou est corrompu.")
        return False
    except rarfile.NoRarEntry: # Peut arriver si l'archive est vide
        print(f"Avertissement : L'archive RAR '{rar_file_path}' est vide.")
        return False # Ou True si on considère que c'est un succès d'extraire une archive vide
    except Exception as e: 
        print(f"Une erreur est survenue pendant la décompression du RAR : {e}")
        if "unrar" in str(e).lower() and ("not found" in str(e).lower() or "aucun fichier ou dossier" in str(e).lower()):
             print("Veuillez vous assurer que l'outil 'unrar' est installé et accessible dans votre PATH.")
        return False

def extract_archive(archive_path, output_dir):
    """
    Fonction générique pour tenter d'extraire une archive.
    Détecte le type et appelle la fonction de décompression appropriée.
    """
    print(f"Tentative d'extraction de '{archive_path}' vers '{output_dir}'...")
    
    try:
        os.makedirs(output_dir, exist_ok=True)
    except Exception as e:
        print(f"Erreur lors de la création du dossier de sortie '{output_dir}': {e}")
        return False

    if not os.path.exists(archive_path):
        print(f"Erreur : Le fichier archive '{archive_path}' n'existe pas.")
        return False

    # Utiliser l'analyseur IA pour déterminer le type d'archive si possible, sinon se baser sur l'extension.
    archive_type_analysis = "unknown_extension_based"
    if AI_ANALYZER_AVAILABLE:
        # Note: analyze_file_content lit le fichier. Pourrait être optimisé si on ne veut que le type pour extraction.
        archive_type_analysis = analyze_file_content(archive_path) 
        print(f"Analyse IA pour l'extraction de '{archive_path}': {archive_type_analysis}")


    # Décision basée sur l'analyse IA ou l'extension en fallback
    if archive_type_analysis == "aic_custom_format" or archive_path.endswith(DEFAULT_AIC_EXTENSION): # Supposons que l'IA puisse identifier notre .aic
        print(f"Format .aic détecté pour '{archive_path}'.")
        return decompress_aic(archive_path, output_dir)
    elif archive_type_analysis == "zip_archive" or archive_path.endswith(".zip"):
        print(f"Format .zip standard détecté pour '{archive_path}'.")
        try:
            with zipfile.ZipFile(archive_path, 'r') as zf:
                zf.extractall(output_dir)
                print(f"Fichier .zip '{archive_path}' extrait avec succès dans '{output_dir}'.")
                return True
        except zipfile.BadZipFile:
            print(f"Erreur : '{archive_path}' n'est pas un fichier ZIP valide ou est corrompu.")
            return False
        except Exception as e:
            print(f"Une erreur est survenue pendant la décompression du ZIP : {e}")
            return False
    elif archive_type_analysis == "rar_archive" or archive_path.endswith(".rar"):
        print(f"Format .rar détecté pour '{archive_path}'.")
        return decompress_rar(archive_path, output_dir)
    # Ajoutez d'autres formats ici (7z, tar.gz, etc.) en vous basant sur archive_type_analysis
    # elif archive_type_analysis == "7z_archive" or archive_path.endswith(".7z"):
    # ...
    else:
        # Fallback si l'analyse IA n'est pas conclusive pour l'extraction et l'extension n'est pas gérée
        # On peut tenter une décompression ZIP si c'est un zipfile valide, comme dernier recours
        if zipfile.is_zipfile(archive_path):
            print(f"Type non explicitement reconnu pour '{archive_path}', mais semble être un ZIP. Tentative de décompression ZIP.")
            try:
                with zipfile.ZipFile(archive_path, 'r') as zf:
                    zf.extractall(output_dir)
                    print(f"Fichier '{archive_path}' extrait (comme ZIP) avec succès dans '{output_dir}'.")
                    return True
            except Exception as e:
                print(f"Échec de la tentative de décompression ZIP pour '{archive_path}': {e}")
                return False
        print(f"Format d'archive non reconnu ou non supporté pour '{archive_path}' (Analyse: {archive_type_analysis}).")
        return False
EOF
echo "aicompress/core.py mis à jour."
echo ""

echo "--------------------------------------------------------------------------"
echo "Mise à jour terminée."
echo "N'oubliez pas de vérifier votre fichier main.py pour tester avec une"
echo "variété de types de fichiers (images, pdf, archives zip, code, texte)."
echo ""
echo "Pour tester l'analyseur IA seul (depuis la racine de AICompressProject):"
echo "  python -m aicompress.ai_analyzer"
echo "Pour exécuter les tests principaux :"
echo "  python main.py"
echo "--------------------------------------------------------------------------"

exit 0