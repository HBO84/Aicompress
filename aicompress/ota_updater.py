# aicompress/ota_updater.py
import os
import json
import requests
import hashlib
import shutil

# Fonction de log par défaut pour ce module si aucun callback n'est fourni
def _default_log(message):
    print(message)

# Importer les configurations et fonctions nécessaires depuis ai_analyzer
try:
    from .ai_analyzer import (
        MODELS_SPECIFIC_DIR, 
        TEXT_CLASSIFIER_MODEL_NAME, 
        TEXT_CLASSIFIER_CONFIG_PATH, 
        TEXT_CLASSIFIER_MODEL_PATH,
        save_local_text_classifier_version,
        load_text_classifier, 
        get_local_text_classifier_version,
        ensure_model_dir_exists
        # _default_log n'est plus importé d'ici
    )
    AI_ANALYZER_IMPORTS_OK = True
except ImportError as e:
    _default_log(f"ERREUR (ota_updater.py): Impossible d'importer depuis ai_analyzer: {e}")
    AI_ANALYZER_IMPORTS_OK = False
    MODELS_SPECIFIC_DIR = os.path.join(os.path.expanduser("~"), ".aicompress_fallback", "models")
    TEXT_CLASSIFIER_MODEL_NAME = "text_classifier.joblib"
    TEXT_CLASSIFIER_CONFIG_PATH = os.path.join(MODELS_SPECIFIC_DIR, "text_classifier_config.json")
    TEXT_CLASSIFIER_MODEL_PATH = os.path.join(MODELS_SPECIFIC_DIR, TEXT_CLASSIFIER_MODEL_NAME)
    def ensure_model_dir_exists(): os.makedirs(MODELS_SPECIFIC_DIR, exist_ok=True)
    def save_local_text_classifier_version(v, log_cb=_default_log): log_cb(f"Fallback save_local_text_classifier_version {v}")
    def load_text_classifier(force_retrain=False, log_cb=_default_log): log_cb("Fallback load_text_classifier"); return None
    def get_local_text_classifier_version(log_cb=_default_log): log_cb("Fallback get_local_text_classifier_version"); return "0.0"


# !!! IMPORTANT: REMPLACEZ CETTE URL PAR CELLE DE VOTRE FICHIER model_info.json HÉBERGÉ !!!
MODEL_INFO_URL = "https://gist.githubusercontent.com/HBO84/c9235724d2bc5042dec2d7ad82ddeec6/raw/86997ded9485944ab0dfb47bd58e5922e0635c66/model_info.json"
# Exemple: MODEL_INFO_URL = "https://raw.githubusercontent.com/VotreNomUser/VotreRepo/main/model_info.json"


def _calculate_sha256(filepath, log_callback=_default_log):
    sha256_hash = hashlib.sha256()
    try:
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    except IOError as e:
        log_callback(f"[OTA_UPDATER] Erreur lecture fichier pour checksum '{filepath}': {e}")
        return None

def check_for_model_updates(log_callback=_default_log):
    if not AI_ANALYZER_IMPORTS_OK:
        log_callback("[OTA_UPDATER] ERREUR: Dépendances internes (ai_analyzer) non chargées. MàJ impossible.")
        return {}
        
    log_callback("[OTA_UPDATER] Vérification des mises à jour modèles IA...")
    if "URL_DE_VOTRE_MODEL_INFO.json" in MODEL_INFO_URL or "VotreNomUser/IDdeVotreGist" in MODEL_INFO_URL : 
        log_callback(f"[OTA_UPDATER] ERREUR: MODEL_INFO_URL ('{MODEL_INFO_URL}') semble être un placeholder. Veuillez la configurer dans aicompress/ota_updater.py.")
        return {}
        
    try:
        log_callback(f"[OTA_UPDATER] Récupération des informations depuis : {MODEL_INFO_URL}")
        response = requests.get(MODEL_INFO_URL, timeout=15)
        log_callback(f"[OTA_UPDATER] Statut de la réponse du serveur (model_info.json): {response.status_code}")
        response.raise_for_status()
        server_model_info = response.json()
        log_callback("[OTA_UPDATER] Informations modèles récupérées et parsées du serveur.")
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
        server_version = str(tc_server_info.get("latest_version", "0.0")) # Assurer que c'est une chaîne
        local_version = str(get_local_text_classifier_version()) # Assurer que c'est une chaîne
        
        log_callback(f"[OTA_UPDATER] Text Classifier - Local: v{local_version}, Serveur: v{server_version}")
        
        # Utiliser une comparaison de version plus robuste si nécessaire (ex: via packaging.version)
        # Pour une comparaison simple de chaînes comme "1.1" > "1.0":
        if server_version > local_version: 
            log_callback(f"[OTA_UPDATER] Nouvelle version disponible pour Text Classifier: {server_version}")
            updates_available["text_classifier"] = tc_server_info
        else:
            log_callback("[OTA_UPDATER] Text Classifier est à jour.")
            
    return updates_available

def download_and_install_model(model_name, model_info, log_callback=_default_log):
    if not AI_ANALYZER_IMPORTS_OK:
        log_callback("[OTA_UPDATER] ERREUR: Dépendances internes (ai_analyzer) non chargées. Installation impossible.")
        return False

    ensure_model_dir_exists()
    model_url = model_info.get("url")
    expected_checksum = model_info.get("checksum_sha256")
    new_version = str(model_info.get("latest_version", "0.0")) # Assurer str

    if not model_url:
        log_callback(f"[OTA_UPDATER] ERREUR: URL manquante pour modèle {model_name}.")
        return False

    local_model_file_path = None
    config_update_function = None
    reload_fn = None 

    if model_name == "text_classifier":
        local_model_file_path = TEXT_CLASSIFIER_MODEL_PATH
        config_update_function = save_local_text_classifier_version
        reload_fn = lambda: load_text_classifier(force_retrain=False)
    else:
        log_callback(f"[OTA_UPDATER] ERREUR: Nom de modèle '{model_name}' inconnu pour téléchargement.")
        return False
    
    download_path_temp = local_model_file_path + ".tmp"

    log_callback(f"[OTA_UPDATER] Téléchargement de {model_name} v{new_version}...")
    log_callback(f"[OTA_UPDATER] URL EXACTE UTILISÉE : '{model_url}'")
    
    try:
        headers_to_send = {
            'User-Agent': 'AICompress-Updater/0.1',
            'Accept': '*/*' 
        }
        log_callback(f"[OTA_UPDATER] Utilisation des en-têtes : {headers_to_send}")

        with requests.get(model_url, stream=True, timeout=120, headers=headers_to_send, allow_redirects=True) as r:
            log_callback(f"[OTA_UPDATER] Statut de la réponse du serveur (modèle) : {r.status_code}")
            log_callback(f"[OTA_UPDATER] En-têtes de la réponse du serveur (modèle) : {r.headers}")
            
            if not r.ok:
                 log_callback(f"[OTA_UPDATER] ERREUR HTTP {r.status_code} - {r.reason} lors du téléchargement du modèle.")
                 try:
                     log_callback(f"[OTA_UPDATER] Début du contenu de la réponse (erreur): {r.text[:500]}")
                 except Exception: # r.text pourrait échouer si le contenu n'est pas textuel
                     log_callback(f"[OTA_UPDATER] Impossible de lire le début du contenu de la réponse d'erreur.")
                 return False
            
            # r.raise_for_status() # Déjà géré par if not r.ok

            with open(download_path_temp, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192 * 16): 
                    f.write(chunk)
        
        downloaded_file_size = os.path.getsize(download_path_temp) if os.path.exists(download_path_temp) else -1
        log_callback(f"[OTA_UPDATER] {model_name} téléchargé : {download_path_temp} (Taille: {downloaded_file_size} octets)")

        if downloaded_file_size <= 0 : # Vérifier si le fichier est vide ou non existant
            log_callback(f"[OTA_UPDATER] ERREUR: Fichier téléchargé est vide ou n'existe pas.")
            if os.path.exists(download_path_temp): os.remove(download_path_temp)
            return False

        if expected_checksum:
            log_callback(f"[OTA_UPDATER] Vérification checksum pour {model_name}...")
            downloaded_checksum = _calculate_sha256(download_path_temp, log_callback)
            if downloaded_checksum != expected_checksum:
                log_callback(f"[OTA_UPDATER] ERREUR: Checksum invalide. Attendu: {expected_checksum}, Obtenu: {downloaded_checksum}")
                log_callback(f"[OTA_UPDATER] Fichier téléchargé par le script CONSERVÉ pour inspection : {download_path_temp}")
                # os.remove(download_path_temp) # COMMENTÉ POUR INSPECTION
                return False
            log_callback(f"[OTA_UPDATER] Checksum vérifié.")

        backup_path = local_model_file_path + ".bak"
        if os.path.exists(local_model_file_path):
            if os.path.exists(backup_path): os.remove(backup_path)
            shutil.move(local_model_file_path, backup_path)
            log_callback(f"[OTA_UPDATER] Ancien modèle sauvegardé en .bak: {backup_path}")
        
        shutil.move(download_path_temp, local_model_file_path)
        log_callback(f"[OTA_UPDATER] {model_name} installé: {local_model_file_path}")

        if config_update_function: config_update_function(new_version)
        if reload_fn: reload_fn() 
        
        log_callback(f"[OTA_UPDATER] Modèle {model_name} (v{new_version}) mis à jour et rechargé.")
        return True
    except requests.exceptions.Timeout:
        log_callback(f"[OTA_UPDATER] Timeout lors du téléchargement du modèle {model_name} depuis {model_url}.")
    except requests.exceptions.RequestException as e:
        log_callback(f"[OTA_UPDATER] Erreur de téléchargement pour {model_name}: {e}")
    except IOError as e:
        log_callback(f"[OTA_UPDATER] Erreur d'écriture fichier pour {model_name}: {e}")
    except Exception as e:
        log_callback(f"[OTA_UPDATER] Erreur inattendue pendant l'installation de {model_name}: {e}")
        import traceback; log_callback(f"[OTA_UPDATER] Traceback: {traceback.format_exc()}")
    
    # Nettoyage en cas d'erreur après le téléchargement mais avant le déplacement final
    if os.path.exists(download_path_temp):
        # Si on veut le garder pour inspection en cas d'erreur autre que checksum,
        # il faudrait une logique plus fine. Ici, on le supprime.
        # La logique de conservation est déjà gérée pour l'échec du checksum.
        if expected_checksum and downloaded_checksum != expected_checksum : # Si on est ici et que c'était un échec de checksum
             pass # Déjà conservé
        else:
             if os.path.exists(download_path_temp): os.remove(download_path_temp)
    return False

if __name__ == '__main__':
    print("--- Test OTA Updater (nécessite configuration MODEL_INFO_URL et serveur de test) ---")
    if "URL_DE_VOTRE_MODEL_INFO.json" in MODEL_INFO_URL or "VotreNomUser/IDdeVotreGist" in MODEL_INFO_URL:
        print("MODIFIEZ MODEL_INFO_URL dans aicompress/ota_updater.py pour tester.")
    elif not AI_ANALYZER_IMPORTS_OK:
        print("Imports depuis ai_analyzer ont échoué. Vérifiez les erreurs console au démarrage de la GUI si elle est utilisée.")
    else:
        ensure_model_dir_exists() 
        print(f"Utilisation du dossier de modèles: {MODELS_SPECIFIC_DIR}")
        # Simuler une version locale plus ancienne pour le test
        save_local_text_classifier_version("0.9") 
        print(f"Version locale (avant test): {get_local_text_classifier_version()}")
        
        updates = check_for_model_updates()
        if updates:
            print("Mises à jour disponibles:", updates)
            if "text_classifier" in updates:
                info = updates["text_classifier"]
                user_input = input(f"Test: Télécharger text_classifier version {info.get('latest_version')}? (o/N): ")
                if user_input.lower() == 'o':
                    if download_and_install_model("text_classifier", info):
                        print("Mise à jour réussie (test script).")
                    else:
                        print("Échec de la mise à jour (test script).")
                    print(f"Version locale (après test): {get_local_text_classifier_version()}")
        else:
            print("Aucune mise à jour ou erreur vérification.")