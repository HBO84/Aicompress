# aicompress/external_handlers.py

import os
import sys
import zipfile 

# --- Imports Conditionnels pour les bibliothèques externes ---
try:
    import rarfile
    RARFILE_AVAILABLE = True
except ImportError:
    RARFILE_AVAILABLE = False

try:
    import py7zr
    PY7ZR_AVAILABLE = True
except ImportError:
    PY7ZR_AVAILABLE = False


# --- Fonction de Log par Défaut pour ce Module ---
def _default_log(message):
    print(f"[EXT_HANDLERS] {message}")


# --- Configuration UNRAR pour PyInstaller ---
# Tenter de configurer le chemin pour unrar.exe si nous sommes dans un bundle PyInstaller
if RARFILE_AVAILABLE and getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
    application_path = sys._MEIPASS
    unrar_exe_path_in_bundle = os.path.join(application_path, "unrar.exe")
    if hasattr(rarfile, 'UNRAR_TOOL'):
        if os.path.exists(unrar_exe_path_in_bundle):
            rarfile.UNRAR_TOOL = unrar_exe_path_in_bundle
            _default_log(f"Chemin UNRAR_TOOL configuré sur : {unrar_exe_path_in_bundle}")
        else:
            _default_log(f"AVERTISSEMENT: unrar.exe non trouvé dans le bundle à {unrar_exe_path_in_bundle}")


# --- Fonctions de Décompression pour Formats Externes ---

def decompress_rar(archive_path, output_dir, password=None, 
                   log_callback=_default_log, progress_callback=None, 
                   cancel_event=None, conflict_policy='overwrite'):
    logger = log_callback
    
    if not RARFILE_AVAILABLE: # Utilise le flag défini en haut de ce fichier
        logger("[EXT_HANDLERS] ERREUR: Module rarfile non disponible.")
        if callable(progress_callback): progress_callback(0, 1) 
        return False, "RARLibNotAvailable"
    
    # ... (le reste de la fonction decompress_rar avec la boucle membre par membre et la gestion des conflits, comme avant) ...
    if cancel_event and cancel_event.is_set(): return False, "CancelledBeforeStart"
    logger(f"Décompression RAR: '{os.path.basename(archive_path)}' (Politique Conflit: {conflict_policy})")
    if not os.path.exists(archive_path): return False, "FileNotFound"
    
    num_rar_members = 0; processed_rar_members = 0; success_overall = True; status_overall = "Success"; operation_cancelled_internally = False
    try:
        os.makedirs(output_dir, exist_ok=True)
        rar_options = {'mode': 'r'};
        if password: rar_options['pwd'] = password
        with rarfile.RarFile(archive_path, **rar_options) as rf:
            members_to_extract = rf.infolist(); num_rar_members = len(members_to_extract)
            logger(f"Archive RAR contient {num_rar_members} membres.")
            if callable(progress_callback) and num_rar_members > 0: progress_callback(0, num_rar_members)
            for member in members_to_extract:
                if cancel_event and cancel_event.is_set(): operation_cancelled_internally = True; break 
                target_path = os.path.join(output_dir, member.filename)
                should_extract = True
                if os.path.exists(target_path):
                    if member.is_dir(): pass
                    elif conflict_policy == 'skip': logger(f"AVERT: Fichier RAR '{member.filename}' existe, ignoré."); should_extract = False
                    elif conflict_policy == 'rename': logger(f"AVERT: Renommage non implémenté. Fichier '{member.filename}' ignoré."); should_extract = False
                if should_extract:
                    logger(f"Extraction membre RAR : {member.filename}")
                    try: rf.extract(member, path=output_dir)
                    except Exception as e_member_ex: logger(f"Erreur extraction membre RAR '{member.filename}': {e_member_ex}"); success_overall = False; status_overall = f"MemberExtractErrorRAR"
                processed_rar_members += 1
                if callable(progress_callback): progress_callback(processed_rar_members, num_rar_members)
            if not operation_cancelled_internally and not success_overall: logger(f"RAR traité avec des erreurs d'extraction.")
            elif not operation_cancelled_internally and success_overall: logger(f"RAR '{os.path.basename(archive_path)}' extrait avec succès.")
    except (rarfile.PasswordRequired, rarfile.WrongPassword): status_overall = "PasswordErrorRAR"; success_overall = False
    except rarfile.BadRarFile as e_br: status_overall = "BadRarFileOrPassword"; success_overall = False
    except Exception as e: status_overall = f"UnknownErrorRAR: {e}"; success_overall = False
    if operation_cancelled_internally: status_overall = "Cancelled"; success_overall = False
    if callable(progress_callback): progress_callback(processed_rar_members if success_overall else 0, num_rar_members if num_rar_members > 0 else 1)
    return success_overall, status_overall


def decompress_7z(archive_path, output_dir, password=None, 
                  log_callback=_default_log, progress_callback=None, cancel_event=None):
    logger = log_callback
    if not PY7ZR_AVAILABLE:
        logger("[EXT_HANDLERS] ERREUR: Bibliothèque py7zr non disponible.")
        return False, "Py7zrLibNotAvailable"
    logger(f"[EXT_HANDLERS] Décompression 7-Zip (mode extractall) de '{os.path.basename(archive_path)}'...")
    # ... (le reste de la fonction decompress_7z utilisant extractall(), comme avant) ...
    success_overall = False; status_overall = "UnknownError7z"
    try:
        os.makedirs(output_dir, exist_ok=True)
        if cancel_event and cancel_event.is_set(): return False, "CancelledBeforeStart"
        if callable(progress_callback): progress_callback(0, 1)
        with py7zr.SevenZipFile(archive_path, mode='r', password=password) as zf_7z:
            zf_7z.extractall(path=output_dir)
        success_overall = True; status_overall = "Success"
        logger(f".7z '{os.path.basename(archive_path)}' extrait avec extractall().")
    except py7zr.exceptions.PasswordRequired: status_overall = "PasswordError7z"
    except py7zr.exceptions.Bad7zFile: status_overall = "Bad7zFileOrPassword"
    except Exception as e: status_overall = f"UnknownError7z: {e}"
    if callable(progress_callback): progress_callback(1 if success_overall else 0, 1)
    return success_overall, status_overall
