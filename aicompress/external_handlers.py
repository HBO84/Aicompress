# aicompress/external_handlers.py

import os
import zipfile # Utilisé par aic_file_handler.extract_archive pour is_zipfile

# --- Imports Conditionnels pour les bibliothèques externes ---
try:
    import rarfile
    RARFILE_AVAILABLE_EXT = True # Nom spécifique pour ce module
except ImportError:
    RARFILE_AVAILABLE_EXT = False
    rarfile = None # Placeholder si rarfile n'est pas importé

try:
    import py7zr
    PY7ZR_AVAILABLE_EXT = True # Nom spécifique pour ce module
except ImportError:
    PY7ZR_AVAILABLE_EXT = False

# Logger de secours si celui de aic_file_handler n'est pas passé (ne devrait pas arriver)
_fallback_ext_log = lambda m: print(f"[EXT_HANDLER_FB_LOG] {m}")

# --- Fonctions de Décompression pour Formats Externes Uniquement ---

def decompress_rar(rar_file_path, output_extract_path, password=None, 
                   log_callback=_fallback_ext_log, progress_callback=None, cancel_event=None):
    logger = log_callback if callable(log_callback) else _fallback_ext_log
    
    if not RARFILE_AVAILABLE_EXT:
        logger("[EXTERNAL_HANDLERS] ERREUR: Module rarfile non disponible.")
        if callable(progress_callback): progress_callback(0, 1) 
        return False, "RARLibNotAvailable"

    # Configuration de UNRAR_TOOL si nécessaire pour PyInstaller (déplacé ici pour autonomie)
    import sys
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS') and rarfile:
        application_path = sys._MEIPASS
        unrar_exe_path_in_bundle = os.path.join(application_path, "unrar.exe")
        if hasattr(rarfile, 'UNRAR_TOOL'):
            if os.path.exists(unrar_exe_path_in_bundle):
                rarfile.UNRAR_TOOL = unrar_exe_path_in_bundle
                logger(f"[EXTERNAL_HANDLERS] Chemin UNRAR_TOOL configuré sur: {unrar_exe_path_in_bundle}")
            # else:
                # logger(f"[EXTERNAL_HANDLERS] AVERT: unrar.exe non trouvé dans le bundle pour config UNRAR_TOOL.")
    
    logger(f"[EXTERNAL_HANDLERS] Décompression RAR: '{os.path.basename(rar_file_path)}'...")
    if not os.path.exists(rar_file_path):
        logger(f"[EXTERNAL_HANDLERS] ERREUR: Fichier RAR '{rar_file_path}' non trouvé.")
        if callable(progress_callback): progress_callback(0, 1)
        return False, "FileNotFound"
    
    num_rar_members = 0; processed_rar_members = 0; success_overall = True; status_overall = "Success"
    operation_cancelled_internally = False
    try:
        os.makedirs(output_extract_path, exist_ok=True)
        rar_options = {'mode': 'r'};
        if password: rar_options['pwd'] = password
        with rarfile.RarFile(rar_file_path, **rar_options) as rf:
            members_to_extract = rf.infolist(); num_rar_members = len(members_to_extract)
            logger(f"[EXTERNAL_HANDLERS] Archive RAR contient {num_rar_members} membres.")
            if callable(progress_callback) and num_rar_members > 0: progress_callback(0, num_rar_members)
            for member in members_to_extract:
                if cancel_event and cancel_event.is_set(): operation_cancelled_internally = True; break
                logger(f"[EXTERNAL_HANDLERS] Extraction membre RAR : {member.filename}")
                try: rf.extract(member, path=output_extract_path, overwrite=True)
                except Exception as e_member_ex:
                    logger(f"[EXTERNAL_HANDLERS] Erreur extraction membre RAR '{member.filename}': {e_member_ex}"); success_overall = False
                processed_rar_members += 1
                if callable(progress_callback): progress_callback(processed_rar_members, num_rar_members)
            if not operation_cancelled_internally and not success_overall: status_overall = "PartialSuccessRAR"
    except rarfile.PasswordRequired: status_overall = "PasswordErrorRAR"; success_overall = False
    except rarfile.WrongPassword: status_overall = "PasswordErrorRAR"; success_overall = False
    except rarfile.BadRarFile as e_br: status_overall = "BadRarFileOrPassword"; success_overall = False; logger(f"ERR RAR: {e_br}")
    except Exception as e: status_overall = f"UnknownErrorRAR: {e}"; success_overall = False; logger(f"ERR RAR: {e}")
    if operation_cancelled_internally: status_overall = "Cancelled"; success_overall = False
    if callable(progress_callback): progress_callback(processed_rar_members if num_rar_members > 0 else (1 if success_overall else 0), num_rar_members if num_rar_members > 0 else 1)
    return success_overall, status_overall

def decompress_7z(archive_7z_path, output_extract_path, password=None, 
                  log_callback=_fallback_ext_log, progress_callback=None, cancel_event=None):
    logger = log_callback if callable(log_callback) else _fallback_ext_log
    if not PY7ZR_AVAILABLE_EXT:
        logger("[EXTERNAL_HANDLERS] ERREUR: py7zr non disponible.")
        if callable(progress_callback): progress_callback(0, 1); 
        return False, "Py7zrLibNotAvailable"
    logger(f"[EXTERNAL_HANDLERS] Décompression 7-Zip (extractall) de '{os.path.basename(archive_7z_path)}'...")
    # ... (logique de decompress_7z avec extractall(), comme la dernière version que je vous ai donnée)
    success_overall = False; status_overall = "UnknownError7z"
    try:
        os.makedirs(output_extract_path, exist_ok=True)
        if cancel_event and cancel_event.is_set(): return False, "CancelledBeforeStart"
        if callable(progress_callback): progress_callback(0, 1)
        with py7zr.SevenZipFile(archive_7z_path, mode='r', password=password) as zf_7z:
            zf_7z.extractall(path=output_extract_path)
        success_overall = True; status_overall = "Success"
        logger(f"[EXTERNAL_HANDLERS] .7z '{os.path.basename(archive_7z_path)}' extrait.")
    except py7zr.exceptions.PasswordRequired: status_overall = "PasswordError7z"
    except py7zr.exceptions.Bad7zFile as e_bad_7z: status_overall = "Bad7zFileOrPassword"
    except Exception as e: status_overall = f"UnknownError7z: {e}"
    if callable(progress_callback): progress_callback(1 if success_overall else 0, 1)
    return success_overall, status_overall