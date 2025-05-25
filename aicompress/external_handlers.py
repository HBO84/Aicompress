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


def decompress_rar(rar_file_path, output_extract_path, password=None, 
                   log_callback=None, progress_callback=None, cancel_event=None): # AJOUT cancel_event
    logger = log_callback if callable(log_callback) else _ext_log 
    
    if not _ext_RARFILE_AVAILABLE or _ext_rarfile_module is None:
        logger("[EXTERNAL_HANDLERS] ERREUR: Module rarfile non disponible pour decompress_rar.")
        if callable(progress_callback): progress_callback(0, 1) 
        return False, "RARLibNotAvailable"

    if cancel_event and cancel_event.is_set():
        logger("[EXTERNAL_HANDLERS] Annulation RAR détectée avant le début.")
        return False, "CancelledBeforeStart"

    logger(f"[EXTERNAL_HANDLERS] Décompression RAR: '{os.path.basename(rar_file_path)}'...")
    if not os.path.exists(rar_file_path):
        logger(f"[EXTERNAL_HANDLERS] ERREUR: Fichier RAR '{rar_file_path}' non trouvé.")
        if callable(progress_callback): progress_callback(0, 1)
        return False, "FileNotFound"
    
    num_rar_members = 0
    processed_rar_members = 0
    success_overall = False # Sera mis à True seulement si tout réussit sans annulation
    status_overall = "UnknownErrorRAR"
    operation_cancelled_internally = False

    try:
        os.makedirs(output_extract_path, exist_ok=True)
        
        rar_options = {'mode': 'r'}
        if password:
            rar_options['pwd'] = password
        
        with _ext_rarfile_module.RarFile(rar_file_path, **rar_options) as rf:
            members_to_extract = []
            try:
                if cancel_event and cancel_event.is_set(): raise InterruptedError("Annulation avant rf.infolist()")
                members_to_extract = rf.infolist()
                num_rar_members = len(members_to_extract)
                logger(f"[EXTERNAL_HANDLERS] Archive RAR contient {num_rar_members} membres.")
                if callable(progress_callback) and num_rar_members > 0:
                    progress_callback(0, num_rar_members)
            except InterruptedError: operation_cancelled_internally = True
            except Exception as e_infolist:
                logger(f"[EXTERNAL_HANDLERS] ERREUR: Impossible de lister les membres de l'archive RAR: {e_infolist}")
                # Cela peut arriver avec un mauvais mot de passe pour les en-têtes chiffrés.
                # PasswordRequired ou WrongPassword devraient être attrapées par le bloc externe.
                status_overall = "BadRarFile_Header" # Erreur plus spécifique
                # success_overall reste False

            if not operation_cancelled_internally and num_rar_members > 0:
                for member in members_to_extract:
                    if cancel_event and cancel_event.is_set():
                        operation_cancelled_internally = True
                        logger("[EXTERNAL_HANDLERS] Annulation détectée pendant l'extraction des membres RAR.")
                        break # Sortir de la boucle d'extraction des membres
                    
                    try:
                        rf.extract(member, path=output_extract_path)
                    except Exception as e_member_ex:
                        logger(f"[EXTERNAL_HANDLERS] Erreur extraction membre RAR '{member.filename}': {e_member_ex}")
                        success_overall = False # Marquer qu'au moins un fichier a échoué
                        if status_overall == "Success" or status_overall == "UnknownErrorRAR" : # Ne pas écraser une erreur de mdp
                            status_overall = f"MemberExtractErrorRAR: {member.filename}"
                        # On continue d'essayer les autres pour la progression, mais le succès global est compromis
                    
                    processed_rar_members += 1
                    if callable(progress_callback) and num_rar_members > 0:
                        progress_callback(processed_rar_members, num_rar_members)
                
                if not operation_cancelled_internally:
                    if processed_rar_members == num_rar_members and success_overall: # Si tout s'est bien passé
                         # success_overall aura été mis à False si une e_member_ex s'est produite
                        logger(f"[EXTERNAL_HANDLERS] Fichier RAR '{os.path.basename(rar_file_path)}' traité.")
                        status_overall = "Success"
                    elif processed_rar_members > 0 : # Au moins une extraction, mais peut-être des erreurs
                         logger(f"[EXTERNAL_HANDLERS] Fichier RAR '{os.path.basename(rar_file_path)}' traité avec {processed_rar_members}/{num_rar_members} membres extraits (erreurs possibles).")
                         if status_overall == "UnknownErrorRAR": status_overall = "PartialSuccessRAR" # Ou un autre code
                    # success_overall est déjà à False si une erreur de membre a eu lieu

            elif not operation_cancelled_internally and num_rar_members == 0 and not members_to_extract: # Cas où infolist a réussi mais est vide
                success_overall = True # Archive vide est un "succès" d'extraction
                status_overall = "SuccessEmptyArchive"
                logger(f"[EXTERNAL_HANDLERS] Archive RAR '{os.path.basename(rar_file_path)}' est vide.")


    except _ext_rarfile_module.PasswordRequired: 
        logger(f"[EXTERNAL_HANDLERS] ERREUR: Mot de passe requis pour RAR '{rar_file_path}'.")
        status_overall = "PasswordErrorRAR"; success_overall = False
    except _ext_rarfile_module.WrongPassword:
        logger(f"[EXTERNAL_HANDLERS] ERREUR: Mot de passe incorrect pour RAR '{rar_file_path}'.")
        status_overall = "PasswordErrorRAR"; success_overall = False
    except _ext_rarfile_module.BadRarFile as e_br:
        logger(f"[EXTERNAL_HANDLERS] ERREUR: Fichier RAR invalide/corrompu pour '{rar_file_path}'. {e_br}")
        status_overall = "BadRarFileOrPassword"; success_overall = False
    except InterruptedError: # Attraper si on a levé nous-mêmes
        logger(f"[EXTERNAL_HANDLERS] Décompression RAR annulée pour '{rar_file_path}'.")
        operation_cancelled_internally = True; status_overall = "Cancelled"; success_overall = False
    except Exception as e:
        logger(f"[EXTERNAL_HANDLERS] Erreur inattendue lors de la décompression RAR : {e}")
        status_overall = f"UnknownErrorRAR: {e}"; success_overall = False
    
    if operation_cancelled_internally:
        status_overall = "Cancelled"
        success_overall = False

    if callable(progress_callback):
        final_count = processed_rar_members if num_rar_members > 0 else (1 if success_overall else 0)
        final_total = num_rar_members if num_rar_members > 0 else 1
        progress_callback(final_count, final_total)
        
    return success_overall, status_overall


def decompress_7z(archive_7z_path, output_extract_path, password=None, 
                  log_callback=None, progress_callback=None): # progress_callback est reçu mais ne sera plus utilisé en détail ici
    logger = log_callback if callable(log_callback) else _ext_log
    
    if not PY7ZR_AVAILABLE_EXT: # Flag défini en haut du module external_handlers.py
        logger("[EXTERNAL_HANDLERS] ERREUR: Bibliothèque py7zr non disponible pour décompresser .7z")
        if callable(progress_callback): progress_callback(0, 1) # Indiquer échec initial
        return False, "Py7zrLibNotAvailable"

    logger(f"[EXTERNAL_HANDLERS] Décompression 7-Zip (mode extractall) de '{os.path.basename(archive_7z_path)}'...")
    if not os.path.exists(archive_7z_path):
        logger(f"[EXTERNAL_HANDLERS] ERREUR: Fichier .7z '{archive_7z_path}' non trouvé.")
        if callable(progress_callback): progress_callback(0, 1)
        return False, "FileNotFound"
    
    success_overall = False
    status_overall = "UnknownError7z"

    try:
        os.makedirs(output_extract_path, exist_ok=True)
        
        # La GUI aura mis la barre en mode indéterminé.
        # On peut appeler le callback une fois au début et une fois à la fin pour 0% -> 100% si on veut.
        if callable(progress_callback):
            progress_callback(0, 1) # 0 sur 1 item (l'archive entière)

        with py7zr.SevenZipFile(archive_7z_path, mode='r', password=password) as zf_7z:
            zf_7z.extractall(path=output_extract_path) # Utiliser extractall pour plus de robustesse
        
        success_overall = True 
        status_overall = "Success"
        logger(f"[EXTERNAL_HANDLERS] Fichier .7z '{os.path.basename(archive_7z_path)}' extrait avec extractall().")

    except py7zr.exceptions.PasswordRequired:
        logger(f"[EXTERNAL_HANDLERS] ERREUR: Mot de passe requis pour .7z '{archive_7z_path}'.")
        status_overall = "PasswordError7z" # Ce code est utilisé par la GUI pour redemander
    except py7zr.exceptions.Bad7zFile as e_bad_7z: 
        logger(f"[EXTERNAL_HANDLERS] ERREUR: Fichier .7z invalide/corrompu ou mauvais mot de passe pour '{archive_7z_path}'. Détail: {e_bad_7z}")
        status_overall = "Bad7zFileOrPassword" # Indique un problème plus grave
    except Exception as e:
        logger(f"[EXTERNAL_HANDLERS] Erreur inattendue lors de la décompression .7z : {e}")
        status_overall = f"UnknownError7z: {e}"

    if callable(progress_callback): # Mise à jour finale de la barre
        progress_callback(1 if success_overall else 0, 1)
        
    return success_overall, status_overall

def extract_archive(archive_path, output_dir, password=None, 
                    log_callback=None, progress_callback=None, cancel_event=None): # ADDED cancel_event
    logger = log_callback if callable(log_callback) else _ext_log
    
    logger(f"[EXTERNAL_HANDLERS] Tentative d'extraction de '{os.path.basename(archive_path)}' vers '{output_dir}'...")
    
    if cancel_event and cancel_event.is_set():
        logger("[EXTERNAL_HANDLERS] Annulation détectée avant le début de l'extraction.")
        return False, "CancelledBeforeStart"
        
    if not os.path.exists(archive_path):
        logger(f"[EXTERNAL_HANDLERS] ERREUR: Archive '{archive_path}' non trouvée."); return False, "FileNotFound"
    try:
        os.makedirs(output_dir, exist_ok=True)
    except Exception as e:
        logger(f"[EXTERNAL_HANDLERS] ERREUR création dossier sortie '{output_dir}': {e}"); return False, "OutputDirError"

    _, extension = os.path.splitext(archive_path)
    extension = extension.lower()
    
    analysis_for_extract = "unknown_type"
    if _ext_AI_ANALYZER_AVAILABLE and callable(_ext_analyze_file_content_func):
        if cancel_event and cancel_event.is_set(): return False, "Cancelled"
        analysis_for_extract = _ext_analyze_file_content_func(archive_path, log_callback=logger) 
    
    if cancel_event and cancel_event.is_set(): return False, "Cancelled"

    if extension == _ext_DEFAULT_AIC_EXTENSION or analysis_for_extract == "aic_custom_format": 
        logger(f"[EXTERNAL_HANDLERS] Format AIC détecté, appel de la fonction de décompression AIC.")
        if callable(_ext_decompress_aic_func):
            # Passer cancel_event à la fonction de décompression AIC
            return _ext_decompress_aic_func(archive_path, output_dir, 
                                            password_decompress=password, 
                                            log_callback=logger,
                                            progress_callback=progress_callback,
                                            cancel_event=cancel_event) # PASSING cancel_event
        else:
            logger(f"[EXTERNAL_HANDLERS] ERREUR: Fonction décompression AIC non initialisée."); return False, "AICDecompressorNotInitialized"
    
    elif extension == ".zip" or analysis_for_extract == "zip_archive":
        logger(f"[EXTERNAL_HANDLERS] Format .zip standard détecté.")
        success, status = False, "InitErrorZIP"
        num_files_in_zip = 0; processed_zip_files = 0
        try:
            with zipfile.ZipFile(archive_path, 'r') as zf_zip:
                members_to_extract = zf_zip.infolist()
                num_files_in_zip = len(members_to_extract)
                logger(f"[EXTERNAL_HANDLERS] Archive ZIP contient {num_files_in_zip} membres.")
                if callable(progress_callback) and num_files_in_zip > 0: progress_callback(0, num_files_in_zip)

                for member in members_to_extract:
                    if cancel_event and cancel_event.is_set(): status="Cancelled"; success=False; break
                    try:
                        zf_zip.extract(member, path=output_dir, pwd=password.encode('utf-8') if password else None)
                    except RuntimeError as e_member_rt: 
                        if password and "password" in str(e_member_rt).lower():
                             logger(f"[EXTERNAL_HANDLERS] ERREUR Mdp sur membre ZIP '{member.filename}'. Arrêt.")
                             status = "PasswordErrorZIPMember"; success = False; break 
                        raise 
                    except Exception as e_member_ex:
                        logger(f"[EXTERNAL_HANDLERS] Erreur extraction membre ZIP '{member.filename}': {e_member_ex}")
                        status = f"MemberExtractErrorZIP: {e_member_ex}"; success = False; break 
                    processed_zip_files += 1
                    if callable(progress_callback) and num_files_in_zip > 0: progress_callback(processed_zip_files, num_files_in_zip)
                
                if not (cancel_event and cancel_event.is_set()) and processed_zip_files == num_files_in_zip:
                    success = True; status = "Success"
                    logger(f"[EXTERNAL_HANDLERS] .zip '{archive_path}' extrait ({num_files_in_zip} membres).")

        except RuntimeError as e_rt: 
            if "password" in str(e_rt).lower(): status = "PasswordErrorZIP"
            else: status = f"RuntimeErrorZIP: {e_rt}"
            logger(f"[EXTERNAL_HANDLERS] ERREUR ZIP: {status} pour '{archive_path}'.")
        except Exception as e_ex_zip: status = f"UnknownErrorZIP: {e_ex_zip}"; logger(f"[EXTERNAL_HANDLERS] Erreur ZIP: {e_ex_zip}")
        
        if callable(progress_callback) and num_files_in_zip > 0: 
            progress_callback(processed_zip_files if success else 0, num_files_in_zip)
        return success, status
        
    elif extension == ".rar" or analysis_for_extract == "rar_archive":
        logger(f"[EXTERNAL_HANDLERS] Format .rar détecté.")
        # La fonction decompress_rar doit aussi être modifiée pour accepter cancel_event
        return decompress_rar(archive_path, output_dir, password, logger, progress_callback, cancel_event) # Passer cancel_event

    elif extension == ".7z" or analysis_for_extract == "7z_archive":
        logger(f"[EXTERNAL_HANDLERS] Format .7z détecté.")
        # La fonction decompress_7z (utilisant extractall) ne peut pas facilement être annulée en cours.
        # On vérifie avant de commencer.
        if cancel_event and cancel_event.is_set(): return False, "CancelledBefore7zStart"
        success, status = decompress_7z(archive_path, output_dir, password, logger, progress_callback) # progress_callback gère 0/1 et 1/1
        if cancel_event and cancel_event.is_set() and not success : # Si annulé PENDANT par un autre moyen (improbable pour extractall)
            return False, "Cancelled"
        return success, status
    
    else: 
        if zipfile.is_zipfile(archive_path): # Fallback pour .zip sans extension
            logger(f"[EXTERNAL_HANDLERS] Type non reconnu mais semble ZIP...");
            # Réutiliser la logique ZIP détaillée pour la progression et l'annulation
            # Pour cela, on pourrait appeler extract_archive récursivement avec une extension .zip "forcée"
            # ou dupliquer/factoriser la logique ZIP. Pour l'instant, duplication simple pour le test.
            # (Cette section est identique à la branche .zip ci-dessus)
            success, status = False, "InitErrorZIPFallback"
            num_files_in_zip_fb = 0; processed_zip_files_fb = 0
            try:
                with zipfile.ZipFile(archive_path, 'r') as zf_zip_fb:
                    members_to_extract_fb = zf_zip_fb.infolist()
                    num_files_in_zip_fb = len(members_to_extract_fb)
                    if callable(progress_callback) and num_files_in_zip_fb > 0: progress_callback(0, num_files_in_zip_fb)
                    for member in members_to_extract_fb:
                        if cancel_event and cancel_event.is_set(): status="Cancelled"; success=False; break
                        zf_zip_fb.extract(member, path=output_dir, pwd=password.encode('utf-8') if password else None)
                        processed_zip_files_fb += 1
                        if callable(progress_callback) and num_files_in_zip_fb > 0: progress_callback(processed_zip_files_fb, num_files_in_zip_fb)
                    if not (cancel_event and cancel_event.is_set()) and processed_zip_files_fb == num_files_in_zip_fb:
                        success = True; status = "Success"
                        logger(f"[EXTERNAL_HANDLERS] Extrait (comme ZIP).")
            except RuntimeError as e_rt_fb:
                if "password" in str(e_rt_fb).lower(): status = "PasswordErrorZIPFallback"
                else: status = f"RuntimeErrorZIPFallback: {e_rt_fb}"
            except Exception as e_ex_fb: status = f"UnknownErrorZIPFallback: {e_ex_fb}"
            if not success and status != "Cancelled": logger(f"[EXTERNAL_HANDLERS] ERREUR ZIP (fallback): {status} pour '{archive_path}'.")
            if callable(progress_callback) and num_files_in_zip_fb > 0: progress_callback(processed_zip_files_fb if success else 0, num_files_in_zip_fb)
            return success, status
        
        logger(f"[EXTERNAL_HANDLERS] Format archive non supporté: '{archive_path}' (Analyse: {analysis_for_extract}).")
        return False, "UnsupportedFormat"



# Fin de aicompress/external_handlers.py