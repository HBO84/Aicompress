# aicompress/external_handlers.py

import os
import zipfile 

# --- Variable globale du module pour le logger ---
# Sera initialisée avec un fallback, puis mise à jour par _initialize_dependencies
_ext_log = lambda m: print(f"[EXT_HANDLER_UNINIT_LOG] {m}") 

# --- Variables globales du module pour les dépendances (initialisées avec des fallbacks) ---
_ext_RARFILE_AVAILABLE = False
_ext_rarfile_module = None 
_ext_AI_ANALYZER_AVAILABLE = False
_ext_analyze_file_content_func = lambda fp, lc: "analyzer_unavailable_in_ext_fb"
_ext_DEFAULT_AIC_EXTENSION = ".aic"
_ext_decompress_aic_func = lambda arc, od, pwd, lc: (False, "AICDecompressorNotSetInExtHandler_fb")
PY7ZR_AVAILABLE_EXT = False # Flag pour py7zr, défini par son propre import

try:
    import py7zr
    PY7ZR_AVAILABLE_EXT = True
except ImportError:
    # _ext_log pourrait ne pas être encore le bon logger ici si _initialize_dependencies n'a pas été appelée
    # print("[EXTERNAL_HANDLERS] AVERTISSEMENT: py7zr non trouvé. Décompression .7z désactivée.")
    pass # Le log sera fait dans _initialize_dependencies ou à l'usage


def _initialize_dependencies(dependencies: dict):
    """
    Initialise les dépendances nécessaires depuis le module aic_file_handler.
    Appelé par aic_file_handler.py à la fin de son chargement.
    """
    global _ext_RARFILE_AVAILABLE, _ext_rarfile_module, _ext_AI_ANALYZER_AVAILABLE
    global _ext_analyze_file_content_func, _ext_DEFAULT_AIC_EXTENSION, _ext_decompress_aic_func
    global _ext_log # Important de déclarer qu'on modifie la globale _ext_log

    # Mettre à jour le logger en premier
    passed_log_callback = dependencies.get("log_callback")
    if callable(passed_log_callback):
        _ext_log = passed_log_callback
    
    _ext_log("[EXTERNAL_HANDLERS] Initialisation des dépendances...")
    
    _ext_RARFILE_AVAILABLE = dependencies.get("RARFILE_AVAILABLE_flag", False)
    _ext_rarfile_module = dependencies.get("rarfile_module", None) # Le module rarfile lui-même
    _ext_AI_ANALYZER_AVAILABLE = dependencies.get("AI_ANALYZER_AVAILABLE_flag", False)
    _ext_analyze_file_content_func = dependencies.get("analyze_file_content_func", _ext_analyze_file_content_func)
    _ext_DEFAULT_AIC_EXTENSION = dependencies.get("DEFAULT_AIC_EXTENSION_const", ".aic")
    _ext_decompress_aic_func = dependencies.get("decompress_aic_func", _ext_decompress_aic_func)
    
    if not callable(_ext_decompress_aic_func) or _ext_decompress_aic_func.__name__.startswith('_fallback'):
        _ext_log("[EXTERNAL_HANDLERS] ERREUR critique: decompress_aic_func n'a pas été correctement initialisée depuis aic_file_handler !")
    if not callable(_ext_analyze_file_content_func) or _ext_analyze_file_content_func.__name__.startswith('_fallback'):
        _ext_log("[EXTERNAL_HANDLERS] AVERT: analyze_file_content_func n'a pas été correctement initialisée.")
        
    if _ext_RARFILE_AVAILABLE and _ext_rarfile_module is None:
        _ext_log("[EXTERNAL_HANDLERS] AVERT: RARFILE_AVAILABLE est True mais le module rarfile n'a pas été passé.")
    
    if not PY7ZR_AVAILABLE_EXT:
        _ext_log("[EXTERNAL_HANDLERS] INFO: Bibliothèque py7zr non trouvée. Décompression .7z sera désactivée.")
    else:
        _ext_log("[EXTERNAL_HANDLERS] INFO: Bibliothèque py7zr est disponible.")

    _ext_log("[EXTERNAL_HANDLERS] Dépendances initialisées et vérifiées.")


def decompress_rar(rar_file_path, output_extract_path, password=None, log_callback=None):
    logger = log_callback if callable(log_callback) else _ext_log
    if not _ext_RARFILE_AVAILABLE or _ext_rarfile_module is None:
        logger("[EXTERNAL_HANDLERS] ERREUR: Module rarfile non disponible pour decompress_rar.")
        return False, "RARLibNotAvailable"

    logger(f"[EXTERNAL_HANDLERS] Décompression RAR: '{rar_file_path}'...")
    if not os.path.exists(rar_file_path):
        logger(f"[EXTERNAL_HANDLERS] ERREUR: Fichier RAR '{rar_file_path}' non trouvé.")
        return False, "FileNotFound"
    try:
        os.makedirs(output_extract_path, exist_ok=True)
        with _ext_rarfile_module.RarFile(rar_file_path, 'r', pwd=password) as rf:
            rf.extractall(path=output_extract_path)
        logger(f"[EXTERNAL_HANDLERS] Fichier RAR '{rar_file_path}' extrait avec succès.")
        return True, "Success"
    except _ext_rarfile_module.PasswordRequired: 
        logger(f"[EXTERNAL_HANDLERS] ERREUR: Mot de passe requis pour RAR '{rar_file_path}'.")
        return False, "PasswordErrorRAR"
    except _ext_rarfile_module.WrongPassword:
        logger(f"[EXTERNAL_HANDLERS] ERREUR: Mot de passe incorrect pour RAR '{rar_file_path}'.")
        return False, "PasswordErrorRAR"
    except _ext_rarfile_module.BadRarFile as e_br:
        logger(f"[EXTERNAL_HANDLERS] ERREUR: Fichier RAR invalide/corrompu ou mauvais mdp '{rar_file_path}'. {e_br}")
        return False, "BadRarFileOrPassword"
    except Exception as e:
        logger(f"[EXTERNAL_HANDLERS] Erreur décompression RAR : {e}"); return False, f"UnknownErrorRAR: {e}"

def decompress_7z(archive_7z_path, output_extract_path, password=None, log_callback=None):
    logger = log_callback if callable(log_callback) else _ext_log
    if not PY7ZR_AVAILABLE_EXT:
        logger("[EXTERNAL_HANDLERS] ERREUR: py7zr non disponible pour décompresser .7z")
        return False, "Py7zrLibNotAvailable"
    logger(f"[EXTERNAL_HANDLERS] Décompression 7-Zip de '{archive_7z_path}'...")
    if not os.path.exists(archive_7z_path):
        logger(f"[EXTERNAL_HANDLERS] ERREUR: Fichier .7z '{archive_7z_path}' non trouvé."); return False, "FileNotFound"
    try:
        os.makedirs(output_extract_path, exist_ok=True)
        with py7zr.SevenZipFile(archive_7z_path, mode='r', password=password) as zf_7z:
            zf_7z.extractall(path=output_extract_path)
        logger(f"[EXTERNAL_HANDLERS] Fichier .7z '{archive_7z_path}' extrait."); return True, "Success"
    except py7zr.exceptions.PasswordRequired: logger(f"ERREUR: Mdp requis pour .7z '{archive_7z_path}'."); return False, "PasswordError7z"
    except py7zr.exceptions.Bad7zFile: logger(f"ERREUR: Fichier .7z invalide/mauvais mdp '{archive_7z_path}'."); return False, "Bad7zFileOrPassword"
    except Exception as e: logger(f"Erreur décompression .7z : {e}"); return False, f"UnknownError7z: {e}"

def extract_archive(archive_path, output_dir, password=None, log_callback=None):
    logger = log_callback if callable(log_callback) else _ext_log
    logger(f"[EXTERNAL_HANDLERS] Tentative d'extraction de '{archive_path}' vers '{output_dir}'...")
    if not os.path.exists(archive_path):
        logger(f"[EXTERNAL_HANDLERS] ERREUR: Archive '{archive_path}' non trouvée."); return False, "FileNotFound"
    try: os.makedirs(output_dir, exist_ok=True)
    except Exception as e: logger(f"[EXTERNAL_HANDLERS] ERREUR création dossier sortie '{output_dir}': {e}"); return False, "OutputDirError"

    _, extension = os.path.splitext(archive_path); extension = extension.lower()
    analysis_for_extract = "unknown_type"
    if _ext_AI_ANALYZER_AVAILABLE and callable(_ext_analyze_file_content_func):
        analysis_for_extract = _ext_analyze_file_content_func(archive_path, log_callback=logger) 
    else: logger("[EXTERNAL_HANDLERS] AI Analyzer non disponible pour extract_archive.")

    if extension == _ext_DEFAULT_AIC_EXTENSION or analysis_for_extract == "aic_custom_format": 
        logger(f"[EXTERNAL_HANDLERS] Format AIC détecté, appel de la fonction de décompression AIC.")
        if callable(_ext_decompress_aic_func):
            return _ext_decompress_aic_func(archive_path, output_dir, password_decompress=password, log_callback=logger)
        else: logger(f"[EXTERNAL_HANDLERS] ERREUR: Fonction décompression AIC non initialisée."); return False, "AICDecompressorNotInitialized"
    elif extension == ".zip" or analysis_for_extract == "zip_archive":
        logger(f"[EXTERNAL_HANDLERS] Format .zip standard détecté."); success, status = False, "InitErrorZIP"
        try:
            with zipfile.ZipFile(archive_path, 'r') as zf_zip: zf_zip.extractall(path=output_dir, pwd=password.encode('utf-8') if password else None)
            logger(f"[EXTERNAL_HANDLERS] .zip '{archive_path}' extrait."); success, status = True, "Success"
        except RuntimeError as e_rt:
            if "password" in str(e_rt).lower(): status = "PasswordErrorZIP"
            else: status = f"RuntimeErrorZIP: {e_rt}"
            logger(f"[EXTERNAL_HANDLERS] ERREUR ZIP: {status} pour '{archive_path}'.")
        except Exception as e_ex_zip: status = f"UnknownErrorZIP: {e_ex_zip}"; logger(f"[EXTERNAL_HANDLERS] Erreur ZIP: {e_ex_zip}")
        return success, status
    elif extension == ".rar" or analysis_for_extract == "rar_archive":
        logger(f"[EXTERNAL_HANDLERS] Format .rar détecté."); return decompress_rar(archive_path, output_dir, password, logger)
    elif extension == ".7z" or analysis_for_extract == "7z_archive":
        logger(f"[EXTERNAL_HANDLERS] Format .7z détecté."); return decompress_7z(archive_path, output_dir, password, logger)
    else: 
        if zipfile.is_zipfile(archive_path):
            logger(f"[EXTERNAL_HANDLERS] Type non reconnu mais semble ZIP..."); success, status = False, "InitErrorZIPFallback"
            try:
                with zipfile.ZipFile(archive_path, 'r') as zf_zip_fb: zf_zip_fb.extractall(path=output_dir, pwd=password.encode('utf-8') if password else None)
                logger(f"[EXTERNAL_HANDLERS] Extrait (comme ZIP)."); success, status = True, "Success"
            except RuntimeError as e_rt_fb:
                if "password" in str(e_rt_fb).lower(): status = "PasswordErrorZIPFallback"
                else: status = f"RuntimeErrorZIPFallback: {e_rt_fb}"
                logger(f"[EXTERNAL_HANDLERS] ERREUR ZIP (fallback): {status} pour '{archive_path}'.")
            except Exception as e_ex_fb: status = f"UnknownErrorZIPFallback: {e_ex_fb}"; logger(f"[EXTERNAL_HANDLERS] Échec fallback ZIP: {e_ex_fb}")
            return success, status
        logger(f"[EXTERNAL_HANDLERS] Format archive non supporté: '{archive_path}' (Analyse: {analysis_for_extract}).")
        return False, "UnsupportedFormat"

# Fin de aicompress/external_handlers.py