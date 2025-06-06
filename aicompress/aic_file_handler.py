# aicompress/aic_file_handler.py (Version sans Autoencodeur)

import os
import zipfile
import shutil
import json
import numpy as np
import tempfile
import multiprocessing

from . import external_handlers


# --- Fonction de Log par Défaut ---
def _default_log(message):
    print(message)


# --- Initialisation des Flags de Disponibilité Globaux ---
PIL_AVAILABLE = False
AI_ANALYZER_AVAILABLE = False
CRYPTOGRAPHY_AVAILABLE = False
ZSTD_AVAILABLE = False
CLASSIC_COMPRESSORS_LOADED = False
BROTLI_AVAILABLE = False
ORCHESTRATOR_LOADED_SUCCESSFULLY = False
# PY7ZR_SUPPORT_AVAILABLE est géré par __init__.py via external_handlers


# --- Fallback Functions ---
def _fb_analyze_content(fp, lc=_default_log):
    lc(f"[AIC_HANDLER_FB] analyze_file_content")
    return "analyzer_unavailable"


def _fb_get_features(fp, lc=_default_log):
    lc(f"[AIC_HANDLER_FB] get_file_features")
    return {"type": "analyzer_unavailable", "error": True}


def _fb_encrypt(d, p, l):
    lc(f"[AIC_HANDLER_FB] encrypt")
    return d, None, None, None


def _fb_decrypt(e, p, s, i, t, l):
    lc(f"[AIC_HANDLER_FB] decrypt")
    return e


def _fb_classic_comp(d, params=None, l=_default_log):
    l(f"[AIC_HANDLER_FB] classic_compress")
    return None


def _fb_classic_decomp(d, params=None, l=_default_log):
    l(f"[AIC_HANDLER_FB] classic_decompress")
    return None


_fb_stored_comp = lambda d, p=None, l=_default_log: d
_fb_stored_decomp = lambda d, p=None, l=_default_log: d


def _fb_get_comp_settings(fp, ai, lc=_default_log):
    lc(f"[AIC_HANDLER_FB] get_comp_settings")
    return "DEFLATE", {"level": 6}


analyze_file_content, get_file_features = _fb_analyze_content, _fb_get_features
encrypt_data, decrypt_data = _fb_encrypt, _fb_decrypt
stored_compress, stored_decompress = _fb_stored_comp, _fb_stored_decomp
deflate_compress, deflate_decompress = _fb_classic_comp, _fb_classic_decomp
bzip2_compress, bzip2_decompress = _fb_classic_comp, _fb_classic_decomp
lzma_compress, lzma_decompress = _fb_classic_comp, _fb_classic_decomp
zstd_compress, zstd_decompress = _fb_classic_comp, _fb_classic_decomp
brotli_compress, brotli_decompress = _fb_classic_comp, _fb_classic_decomp
get_compression_settings = _fb_get_comp_settings
METHOD_STORED, METHOD_DEFLATE, METHOD_BZIP2, METHOD_LZMA, METHOD_ZSTD, METHOD_BROTLI = (
    "STORED",
    "DEFLATE",
    "BZIP2",
    "LZMA",
    "ZSTD",
    "BROTLI",
)

# --- Imports Conditionnels et Mise à Jour des Flags et Fonctions ---
try:
    from PIL import Image, ImageOps

    PIL_AVAILABLE = True
    _default_log("[AIC_HANDLER] Pillow OK.")
except ImportError:
    _default_log("AVERT (aic_handler.py): Pillow non trouvée.")
# TensorFlow/Keras n'est plus une dépendance directe de ce module.

try:
    from .ai_analyzer import (
        get_file_features as _real_gff,
        analyze_file_content as _real_afc,
        AI_ANALYZER_AVAILABLE as _ai_flag,
    )

    if _ai_flag:
        get_file_features, analyze_file_content, AI_ANALYZER_AVAILABLE = (
            _real_gff,
            _real_afc,
            True,
        )
        _default_log("[AIC_HANDLER] ai_analyzer OK.")
    else:
        _default_log("[AIC_HANDLER] ai_analyzer chargé mais non dispo (interne).")
except ImportError as e:
    _default_log(f"AVERT (aic_handler.py): Échec import ai_analyzer: {e}")



try:
    from .crypto_utils import (
        encrypt_data as _re_cu,
        decrypt_data as _rd_cu,
        CRYPTOGRAPHY_AVAILABLE as _cf_cu_flag,
    )

    if _cf_cu_flag:
        encrypt_data, decrypt_data, CRYPTOGRAPHY_AVAILABLE = _re_cu, _rd_cu, True
        _default_log("[AIC_HANDLER] crypto_utils OK.")
    else:
        _default_log(
            "AVERT (aic_handler.py): crypto_utils chargé, mais cryptography indisponible."
        )
except ImportError as e:
    _default_log(f"AVERT (aic_handler.py): crypto_utils.py non trouvé: {e}")

try:
    from .classic_compressors import (
        stored_compress as _cs_s,
        deflate_compress as _cs_d,
        bzip2_compress as _cs_b,
        lzma_compress as _cs_l,
        zstd_compress as _cs_z,
        brotli_compress as _cs_br,
        stored_decompress as _cs_sd,
        deflate_decompress as _cs_dd,
        bzip2_decompress as _cs_bd,
        lzma_decompress as _cs_ld,
        zstd_decompress as _cs_zd,
        brotli_decompress as _cs_brd,
        METHOD_STORED as _M_S_cl,
        METHOD_DEFLATE as _M_D_cl,
        METHOD_BZIP2 as _M_B_cl,
        METHOD_LZMA as _M_L_cl,
        METHOD_ZSTD as _M_Z_cl,
        METHOD_BROTLI as _M_BR_cl,
        ZSTD_AVAILABLE as ZSTD_FLAG_cl,
        BROTLI_AVAILABLE as BROTLI_FLAG_cl,
        CLASSIC_COMPRESSORS_READY as CC_READY_FLAG_cl,
    )

    if CC_READY_FLAG_cl:
        stored_compress = _cs_s
        deflate_compress = _cs_d
        bzip2_compress = _cs_b
        lzma_compress = _cs_l
        zstd_compress = _cs_z
        brotli_compress = _cs_br
        stored_decompress = _cs_sd
        deflate_decompress = _cs_dd
        bzip2_decompress = _cs_bd
        lzma_decompress = _cs_ld
        zstd_decompress = _cs_zd
        brotli_decompress = _cs_brd
        (
            METHOD_STORED,
            METHOD_DEFLATE,
            METHOD_BZIP2,
            METHOD_LZMA,
            METHOD_ZSTD,
            METHOD_BROTLI,
        ) = (_M_S_cl, _M_D_cl, _M_B_cl, _M_L_cl, _M_Z_cl, _M_BR_cl)
        CLASSIC_COMPRESSORS_LOADED = True
        ZSTD_AVAILABLE = ZSTD_FLAG_cl
        BROTLI_AVAILABLE = BROTLI_FLAG_cl
        _default_log("[AIC_HANDLER] classic_compressors OK.")
        if not ZSTD_AVAILABLE:
            _default_log("[AIC_HANDLER] AVERT: Zstd non dispo via classic_compressors.")
        if not BROTLI_AVAILABLE:
            _default_log(
                "[AIC_HANDLER] AVERT: Brotli non dispo via classic_compressors."
            )
    else:
        _default_log("[AIC_HANDLER] classic_compressors chargé mais non opérationnel.")
except ImportError as e:
    _default_log(f"AVERT (aic_handler.py): classic_compressors.py non trouvé: {e}")

# L'import de ae_engine est supprimé

try:
    from .orchestrator import (
        get_compression_settings as _real_gcs_orch,
        ORCHESTRATOR_IS_READY as ORCH_FLAG_MOD,
    )

    if ORCH_FLAG_MOD:
        get_compression_settings = _real_gcs_orch
        ORCHESTRATOR_LOADED_SUCCESSFULLY = True
        _default_log("[AIC_HANDLER] orchestrator et son modèle sont prêts.")
    else:
        _default_log(
            "[AIC_HANDLER] orchestrator chargé, mais son modèle interne non prêt."
        )
except ImportError as e:
    _default_log(f"AVERT (aic_handler.py): orchestrator.py non trouvé: {e}")

METADATA_FILENAME = "aicompress_metadata.json"
DEFAULT_AIC_EXTENSION = ".aic"
# LATENT_FILE_EXTENSION = ".aic_latent"; # SUPPRIMÉ
ENCRYPTED_FILE_EXTENSION = ".aic_enc"


def _is_archive_file(file_path, log_callback=_default_log):
    SUPPORTED_RECURSIVE_TYPES = [
        "zip_archive",
        "rar_archive",
        "7z_archive",
        "aic_custom_format",
    ]
    SUPPORTED_RECURSIVE_EXTENSIONS = [
        ".zip",
        ".rar",
        ".7z",
        DEFAULT_AIC_EXTENSION.lower(),
    ]
    if not AI_ANALYZER_AVAILABLE:
        ext = os.path.splitext(file_path)[1].lower()
        return ext in SUPPORTED_RECURSIVE_EXTENSIONS
    analysis_type = analyze_file_content(file_path, log_callback=log_callback)
    return analysis_type in SUPPORTED_RECURSIVE_TYPES


def _count_total_individual_files(input_paths, log_callback=_default_log):
    total_files = 0
    for item_path in input_paths:
        if not os.path.exists(item_path):
            continue
        if os.path.isfile(item_path):
            total_files += 1
        elif os.path.isdir(item_path):
            for _, _, files_in_dir in os.walk(item_path):
                total_files += len(files_in_dir)
    log_callback(
        f"[AIC_HANDLER] Estimation initiale du nombre de fichiers à traiter: {total_files}"
    )
    return total_files


# Fonction "Travailleur" MODIFIÉE
def _process_and_add_file_to_aic(
    file_path_on_disk,
    arcname_to_use,
    password_compress,
    log_callback,
    cancel_event=None,
):
    """
    Prépare les données d'un fichier unique pour l'archivage AIC.
    Effectue l'analyse, la décision IA, la compression, le chiffrement.
    NE GÈRE PAS L'ÉCRITURE ZIP NI LA PROGRESSION GLOBALE.
    Retourne un tuple: (final_arcname_in_zip, data_to_write, item_metadata_dict, status_string)
    status_string peut être "completed", "cancelled", "read_error", etc.
    """
    item_basename = os.path.basename(file_path_on_disk)
    current_file_status = "processing"  # Statut par défaut
    item_meta = {
        "original_name": arcname_to_use,
        "status": current_file_status,
        "analysis": "N/A",
        "size_original_bytes": 0,
        "compression_method_used": METHOD_STORED,
        "compression_params": {},
        "encrypted": False,
        "crypto_salt_hex": None,
        "crypto_iv_hex": None,
        "crypto_tag_hex": None,
        # Les champs liés à l'AE ont été supprimés
    }

    log_callback(
        f"[AIC_WORKER] Traitement de '{item_basename}' pour archive sous '{arcname_to_use}'"
    )

    if cancel_event and cancel_event.is_set():
        log_callback(f"[AIC_WORKER] Annulation avant traitement de {item_basename}.")
        item_meta["status"] = "cancelled_before_processing"
        return arcname_to_use, None, item_meta, "cancelled"

    try:
        item_meta["size_original_bytes"] = os.path.getsize(file_path_on_disk)
    except OSError as e_size:
        log_callback(f"[AIC_WORKER] ERREUR TAILLE pour {file_path_on_disk}: {e_size}")
        item_meta["status"] = "read_error_size"
        return arcname_to_use, None, item_meta, "read_error"

    if AI_ANALYZER_AVAILABLE:
        item_meta["analysis"] = analyze_file_content(
            file_path_on_disk, log_callback=log_callback
        )

    if cancel_event and cancel_event.is_set():
        log_callback(f"[AIC_WORKER] Annulation après analyse pour {item_basename}.")
        item_meta["status"] = "cancelled_after_analysis"
        return arcname_to_use, None, item_meta, "cancelled"

    # Obtenir la décision de l'orchestrateur
    comp_method_name_from_ia, comp_params_dict_from_ia = get_compression_settings(
        file_path_on_disk, item_meta["analysis"], log_callback=log_callback
    )
    item_meta["compression_method_used"] = comp_method_name_from_ia
    item_meta["compression_params"] = comp_params_dict_from_ia

    if cancel_event and cancel_event.is_set():
        log_callback(f"[AIC_WORKER] Annulation après décision IA pour {item_basename}.")
        item_meta["status"] = "cancelled_after_ia"
        return arcname_to_use, None, item_meta, "cancelled"

    final_arcname_for_zip_member = arcname_to_use
    data_to_process_further = b""

    try:  # Lecture du fichier
        if cancel_event and cancel_event.is_set():
            item_meta["status"] = "cancelled_before_read"
            return arcname_to_use, None, item_meta, "cancelled"
        with open(file_path_on_disk, "rb") as f_orig:
            data_to_process_further = f_orig.read()
    except Exception as e_read:
        log_callback(f"[AIC_WORKER] ERREUR lecture fichier {item_basename}: {e_read}")
        item_meta["status"] = "read_error"
        return arcname_to_use, None, item_meta, "read_error"

    if cancel_event and cancel_event.is_set():
        item_meta["status"] = "cancelled_before_compression"
        return arcname_to_use, None, item_meta, "cancelled"

    # Compression classique
    data_after_internal_comp = data_to_process_further
    if CLASSIC_COMPRESSORS_LOADED:
        if comp_method_name_from_ia == METHOD_BZIP2:
            data_after_internal_comp = bzip2_compress(
                data_to_process_further, comp_params_dict_from_ia, log_callback
            )
        elif comp_method_name_from_ia == METHOD_LZMA:
            data_after_internal_comp = lzma_compress(
                data_to_process_further, comp_params_dict_from_ia, log_callback
            )
        elif comp_method_name_from_ia == METHOD_ZSTD:
            data_after_internal_comp = zstd_compress(
                data_to_process_further, comp_params_dict_from_ia, log_callback
            )
        elif comp_method_name_from_ia == METHOD_BROTLI:
            data_after_internal_comp = brotli_compress(
                data_to_process_further, comp_params_dict_from_ia, log_callback
            )
        elif comp_method_name_from_ia == METHOD_DEFLATE:
            data_after_internal_comp = deflate_compress(
                data_to_process_further, comp_params_dict_from_ia, log_callback
            )
        elif comp_method_name_from_ia == METHOD_STORED:
            data_after_internal_comp = stored_compress(
                data_to_process_further, comp_params_dict_from_ia, log_callback
            )

        if data_after_internal_comp is None:  # Échec de la compression classique
            log_callback(
                f"[AIC_WORKER] Échec compresseur {comp_method_name_from_ia}. Utilisation données originales pour {item_basename}."
            )
            data_after_internal_comp = (
                data_to_process_further  # Revenir aux données avant cette compression
            )
            item_meta["compression_method_used"] = METHOD_STORED
            item_meta["compression_params"] = {}
    elif comp_method_name_from_ia not in [
        METHOD_STORED,
        METHOD_DEFLATE,
        METHOD_BROTLI,
    ]:  # Si pas de compresseurs classiques chargés
        log_callback(
            f"[AIC_WORKER] AVERT: classic_compressors non chargé, {comp_method_name_from_ia} non applicable. Fallback STORED."
        )
        item_meta["compression_method_used"] = METHOD_STORED
        item_meta["compression_params"] = {}

    if cancel_event and cancel_event.is_set():
        item_meta["status"] = "cancelled_before_encrypt"
        return arcname_to_use, None, item_meta, "cancelled"

    # Chiffrement
    data_to_write_to_zip = data_after_internal_comp
    if (
        password_compress
        and data_to_write_to_zip is not None
        and CRYPTOGRAPHY_AVAILABLE
    ):
        encrypted_bundle = encrypt_data(
            data_to_write_to_zip, password_compress, log_callback=log_callback
        )
        if cancel_event and cancel_event.is_set():
            item_meta["status"] = "cancelled_during_encrypt"
            return arcname_to_use, None, item_meta, "cancelled"

        if (
            encrypted_bundle
            and encrypted_bundle[0] is not None
            and encrypted_bundle[1] is not None
        ):
            data_to_write_to_zip, salt, iv, tag = encrypted_bundle
            item_meta["encrypted"] = True
            item_meta["crypto_salt_hex"] = salt.hex()
            item_meta["crypto_iv_hex"] = iv.hex()
            item_meta["crypto_tag_hex"] = tag.hex()
            final_arcname_for_zip_member += ENCRYPTED_FILE_EXTENSION
        else:
            log_callback(
                f"[AIC_WORKER] AVERT: Échec chiffrement {item_basename}. Stockage non chiffré."
            )
            item_meta["encrypted"] = False
            if final_arcname_for_zip_member.endswith(ENCRYPTED_FILE_EXTENSION):
                final_arcname_for_zip_member = final_arcname_for_zip_member[
                    : -len(ENCRYPTED_FILE_EXTENSION)
                ]

    if data_to_write_to_zip is not None:
        item_meta["status"] = "completed"
        current_file_status = "completed"
    else:
        log_callback(
            f"[AIC_WORKER] AVERT: Pas de données à préparer pour {item_basename} (compression a pu échouer)."
        )
        item_meta["status"] = "error_no_data_to_write"
        current_file_status = "error_no_data"

    return (
        final_arcname_for_zip_member,
        data_to_write_to_zip,
        item_meta,
        current_file_status,
    )


def compress_to_aic(input_paths, output_aic_path, password_compress=None, 
                    log_callback=_default_log, progress_callback=None, 
                    cancel_event=None, recursively_optimize=False,
                    pool_info_callback=None):
    log_callback(f"[AIC_HANDLER] Compression AIC vers '{os.path.basename(output_aic_path)}' (Chiffré: {'Oui' if password_compress else 'Non'})...")
    if recursively_optimize:
        log_callback("[AIC_HANDLER] Mode d'optimisation récursive des archives ACTIVÉ.")
        
    all_items_metadata = [] 
    actual_files_to_process_list = [] 
    temp_dirs_to_cleanup = []
    operation_cancelled_flag = False

    def discover_files_for_processing(paths_to_scan, base_arc_prefix_for_discovery=""):
        nonlocal operation_cancelled_flag, actual_files_to_process_list, temp_dirs_to_cleanup, all_items_metadata
        
        for item_path_discover in paths_to_scan:
            if cancel_event and cancel_event.is_set(): operation_cancelled_flag=True; return
            
            item_basename = os.path.basename(item_path_discover)
            current_arc_name_discover = os.path.join(base_arc_prefix_for_discovery, item_basename) if base_arc_prefix_for_discovery else item_basename

            if not os.path.exists(item_path_discover):
                all_items_metadata.append({"original_name": current_arc_name_discover, "status": "not_found", "size_original_bytes":0}); continue

            if recursively_optimize and os.path.isfile(item_path_discover) and _is_archive_file(item_path_discover, log_callback):
                log_callback(f"[AIC_HANDLER_DISCOVERY] Archive interne détectée : {current_arc_name_discover}")
                
                # --- CORRECTION DE NOMMAGE ---
                archive_basename_no_ext, _ = os.path.splitext(item_basename)
                archive_dir_name_in_aic = os.path.join(base_arc_prefix_for_discovery, archive_basename_no_ext)
                # --- FIN CORRECTION ---
                
                temp_dir_rec = tempfile.mkdtemp(prefix="aicompress_rec_"); temp_dirs_to_cleanup.append(temp_dir_rec)
                log_callback(f"[AIC_HANDLER] Extraction de '{item_basename}' dans : {temp_dir_rec}")
                
                try:
                    success_extract, status_extract = extract_archive(
                        item_path_discover, temp_dir_rec, password=None, 
                        log_callback=log_callback, cancel_event=cancel_event, progress_callback=None
                    )
                    
                    if cancel_event and cancel_event.is_set(): operation_cancelled_flag=True; return 
                    
                    if success_extract:
                        log_callback(f"[AIC_HANDLER_DISCOVERY] Extraction de '{current_arc_name_discover}' réussie. Traitement du contenu.")
                        # Utiliser le nom sans extension pour le dossier virtuel dans les métadonnées
                        all_items_metadata.append({
                            "original_name": archive_dir_name_in_aic, # Utiliser le nom corrigé
                            "type_in_archive": "directory", 
                            "status": "recursively_processed_archive_header", 
                            "size_original_bytes": os.path.getsize(item_path_discover) if os.path.exists(item_path_discover) else 0
                        })
                        # Passer le nom sans extension comme préfixe pour les fichiers internes
                        discover_files_for_processing( [os.path.join(temp_dir_rec, f) for f in sorted(os.listdir(temp_dir_rec))], 
                                                       base_arc_prefix_for_discovery=archive_dir_name_in_aic) # Utiliser le nom corrigé
                    else: 
                        log_callback(f"[AIC_HANDLER_DISCOVERY] AVERT: Échec extraction archive interne '{current_arc_name_discover}'. Traitée comme fichier normal. Status: {status_extract}")
                        actual_files_to_process_list.append({'source_path': item_path_discover, 'archive_path': current_arc_name_discover})
                
                except Exception as e_discover_rec:
                     log_callback(f"[AIC_HANDLER_DISCOVERY] ERREUR pendant la découverte/extraction de l'archive '{current_arc_name_discover}': {e_discover_rec}. Traitée comme fichier normal.")
                     actual_files_to_process_list.append({'source_path': item_path_discover, 'archive_path': current_arc_name_discover})
            
            elif os.path.isfile(item_path_discover):
                actual_files_to_process_list.append({'source_path': item_path_discover, 'archive_path': current_arc_name_discover})
            
            elif os.path.isdir(item_path_discover):
                all_items_metadata.append({"original_name": current_arc_name_discover, "type_in_archive": "directory", "status": "processed_placeholder"})
                discover_files_for_processing([os.path.join(item_path_discover, f) for f in sorted(os.listdir(item_path_discover))], 
                                              base_arc_prefix_for_discovery=current_arc_name_discover)
            if operation_cancelled_flag: return
    
    total_individual_files_for_progress = 0
    
    try:
        log_callback("[AIC_HANDLER] Phase 1: Découverte de tous les fichiers à traiter...")
        discover_files_for_processing(input_paths)
        
        if operation_cancelled_flag: 
            raise InterruptedError("Annulation pendant la découverte des fichiers.")

        total_individual_files_for_progress = len(actual_files_to_process_list)
        log_callback(f"[AIC_HANDLER] Nombre total de fichiers individuels à compresser : {total_individual_files_for_progress}")

        if callable(progress_callback) and total_individual_files_for_progress > 0:
            progress_callback(0, total_individual_files_for_progress)
        
        if total_individual_files_for_progress == 0:
            # ... (logique pour les archives vides comme avant)
            log_callback("[AIC_HANDLER] Aucun fichier individuel à compresser après découverte.")
            with zipfile.ZipFile(output_aic_path, 'w') as zf_empty:
                final_meta_empty = [m for m in all_items_metadata if m.get("status") not in ["recursively_processed_archive_header", "processed_placeholder"]]
                metadata_final_content_empty = {"aicompress_version":"1.5.1-recursive-fix","items_details":final_meta_empty, "global_encryption_hint":bool(password_compress and CRYPTOGRAPHY_AVAILABLE)}
                zf_empty.writestr(METADATA_FILENAME,json.dumps(metadata_final_content_empty,indent=4))
            return True, "SuccessEmptyOrDirsOnly"

        log_callback("[AIC_HANDLER] Phase 2: Préparation des données des fichiers en parallèle...")
        
        manager = multiprocessing.Manager()
        mp_worker_cancel_event = manager.Event()
        tasks_for_pool = []
        for task_item in actual_files_to_process_list:
            tasks_for_pool.append(
                (task_item['source_path'], task_item['archive_path'], 
                 password_compress, _default_log, mp_worker_cancel_event) 
            )

        results_from_workers = []
        if tasks_for_pool:
            # ... (logique du multiprocessing avec apply_async comme dans la version précédente)
            num_processes = min(os.cpu_count() or 1, len(tasks_for_pool))
            log_callback(f"[AIC_HANDLER] Lancement du pool de {num_processes} processus pour {len(tasks_for_pool)} fichiers...")
            if callable(pool_info_callback): pool_info_callback(num_processes, len(tasks_for_pool))
            with multiprocessing.Pool(processes=num_processes) as pool:
                async_results = [pool.apply_async(_process_and_add_file_to_aic, args=task_args) for task_args in tasks_for_pool]
                processed_files_count = 0
                for res_obj in async_results:
                    if cancel_event and cancel_event.is_set(): mp_worker_cancel_event.set(); pool.terminate(); pool.join(); operation_cancelled_flag = True; break
                    try: result_item = res_obj.get(); results_from_workers.append(result_item)
                    except Exception as e_worker_ex: log_callback(f"[AIC_HANDLER] Tâche worker échouée: {e_worker_ex}")
                    processed_files_count += 1
                    if callable(progress_callback): progress_callback(processed_files_count, total_individual_files_for_progress)
                if operation_cancelled_flag: raise InterruptedError("Pool de processus arrêté")
        
        if operation_cancelled_flag: raise InterruptedError("Opération annulée par l'utilisateur.")

        log_callback("[AIC_HANDLER] Phase 3: Écriture de l'archive AIC et des métadonnées...")
        with zipfile.ZipFile(output_aic_path, 'w') as zf: 
            for result_item in results_from_workers:
                # ... (logique d'écriture des résultats dans le ZIP comme avant)
                if result_item is None: continue 
                final_arcname, data_to_write, item_meta_result, file_status = result_item
                all_items_metadata.append(item_meta_result)
                if file_status == "cancelled": operation_cancelled_flag = True 
                elif data_to_write is not None and file_status == "completed":
                    try: zf.writestr(final_arcname, data_to_write, compress_type=zipfile.ZIP_STORED)
                    except Exception as e_zip_w: log_callback(f"[AIC_HANDLER] ERREUR écriture ZIP pour {final_arcname}: {e_zip_w}")
            
            if operation_cancelled_flag: raise InterruptedError("Opération annulée pendant l'écriture.")

            final_metadata_list = [m for m in all_items_metadata if m.get("status") not in ["recursively_processed_archive_header", "processed_placeholder"]]
            metadata_final_content = {"aicompress_version":"1.5.1-recursive-fix","items_details":final_metadata_list, "global_encryption_hint":bool(password_compress and CRYPTOGRAPHY_AVAILABLE)}
            zf.writestr(METADATA_FILENAME,json.dumps(metadata_final_content,indent=4))
            
            log_callback(f"[AIC_HANDLER] Compression AIC terminée: '{output_aic_path}'"); return True,"Success"
            
    except InterruptedError: 
        log_callback("[AIC_HANDLER] Processus de compression annulé."); return False, "Cancelled"
    except Exception as e: 
        log_callback(f"[AIC_HANDLER] ERREUR MAJEURE compression: {e}"); import traceback
        log_callback(f"[AIC_HANDLER] Traceback: {traceback.format_exc()}"); 
        if callable(progress_callback) and total_individual_files_for_progress >= 0: 
            progress_callback(0, total_individual_files_for_progress if total_individual_files_for_progress > 0 else 1) 
        return False,f"Error: {e}"
    finally: 
        for temp_dir_to_clean in temp_dirs_to_cleanup:
            log_callback(f"[AIC_HANDLER] Nettoyage final du dossier temporaire : {temp_dir_to_clean}")
            shutil.rmtree(temp_dir_to_clean, ignore_errors=True)
        if 'manager' in locals() and manager: 
            try: manager.shutdown()
            except Exception as e_manager_shutdown: log_callback(f"[AIC_HANDLER] Erreur arrêt manager: {e_manager_shutdown}")




# --- decompress_aic (MODIFIÉE pour supprimer la logique AE et l'annulation) ---
def decompress_aic(archive_path, output_dir, password=None, 
                   log_callback=_default_log, progress_callback=None, 
                   cancel_event=None, conflict_policy='overwrite'):
    log_callback(f"[AIC_HANDLER_DECOMP] >>> Début Décompression AIC: '{os.path.basename(archive_path)}' (Politique Conflit: {conflict_policy})")
    
    if cancel_event and cancel_event.is_set():
        return False, "CancelledBeforeStart"
    if not os.path.exists(archive_path):
        log_callback(f"[AIC_HANDLER_DECOMP] ERREUR: Archive '{archive_path}' non trouvée.")
        if callable(progress_callback): progress_callback(0, 1)
        return False, "FileNotFound"
    
    total_items_from_meta = 0
    processed_items_count = 0
    operation_cancelled_internally = False

    try:
        os.makedirs(output_dir, exist_ok=True)
        with zipfile.ZipFile(archive_path, 'r') as zf:
            log_callback(f"[AIC_HANDLER_DECOMP] Archive ZIP ouverte.")
            metadata = None
            metadata_loaded_successfully = False
            
            try:
                if cancel_event and cancel_event.is_set(): raise InterruptedError("Annulation avant lecture métadonnées")
                metadata_str = zf.read(METADATA_FILENAME)
                metadata = json.loads(metadata_str)
                log_callback("[AIC_HANDLER_DECOMP] Métadonnées lues.")
                metadata_loaded_successfully = True
                items_in_meta = metadata.get('items_details', [])
                total_items_from_meta = len(items_in_meta)
                if callable(progress_callback) and total_items_from_meta > 0:
                    progress_callback(0, total_items_from_meta) 
            except InterruptedError:
                operation_cancelled_internally = True
            except Exception as e_meta: 
                log_callback(f"[AIC_HANDLER_DECOMP] ERREUR métadonnées: {e_meta}. Tentative ZIP standard...");
                try:
                    if cancel_event and cancel_event.is_set(): raise InterruptedError("Annulation avant fallback extractall")
                    zf.extractall(path=output_dir, pwd=password.encode('utf-8') if password else None)
                    if callable(progress_callback): progress_callback(1, 1) 
                    return True, "SuccessZipFallback"
                except InterruptedError: operation_cancelled_internally = True
                except Exception as e_gen_fb: return False, f"ZipFallbackGenericError: {e_gen_fb}"

            if operation_cancelled_internally: return False, "Cancelled"
            if not (metadata_loaded_successfully and metadata and "items_details" in metadata):
                if callable(progress_callback): progress_callback(0, 1)
                return False, "InvalidMetadata" 
            
            log_callback(f"[AIC_HANDLER_DECOMP] Traitement de {total_items_from_meta} items...")
            files_written_by_custom_logic = set() 

            for item_meta in metadata["items_details"]:
                if cancel_event and cancel_event.is_set():
                    operation_cancelled_internally = True; break
                
                original_arcname = item_meta.get("original_name")
                
                if not original_arcname:
                    log_callback("[AIC_HANDLER_DECOMP] AVERT: item_meta sans original_name. Ignoré.")
                    # On incrémente quand même pour que la progression soit juste
                    processed_items_count += 1
                    if callable(progress_callback) and total_items_from_meta > 0:
                        progress_callback(processed_items_count, total_items_from_meta)
                    continue

                output_final_path = os.path.join(output_dir, original_arcname)
                log_callback(f"[AIC_HANDLER_DECOMP] Préparation item: {original_arcname}")

                if item_meta.get("status") == "not_found":
                    log_callback(f" Ignoré (marqué 'not_found'): {original_arcname}")
                elif item_meta.get("type_in_archive") == "directory":
                    if not os.path.exists(output_final_path):
                        os.makedirs(output_final_path, exist_ok=True)
                    files_written_by_custom_logic.add(output_final_path)
                else: # C'est un fichier à extraire
                    if os.path.exists(output_final_path):
                        if conflict_policy == 'skip':
                            log_callback(f"[AIC_HANDLER_DECOMP] AVERT: '{original_arcname}' existe déjà, ignoré.")
                            processed_items_count += 1
                            if callable(progress_callback): progress_callback(processed_items_count, total_items_from_meta)
                            continue
                        elif conflict_policy == 'rename':
                            log_callback(f"[AIC_HANDLER_DECOMP] AVERT: Renommage non implémenté. '{original_arcname}' ignoré.")
                            processed_items_count += 1
                            if callable(progress_callback): progress_callback(processed_items_count, total_items_from_meta)
                            continue
                        elif conflict_policy == 'overwrite':
                            log_callback(f"[AIC_HANDLER_DECOMP] Info: '{original_arcname}' existe déjà, sera écrasé.")

                    comp_method = item_meta.get("compression_method_used")
                    is_encrypted = item_meta.get("encrypted", False)
                    member_name_in_zip = original_arcname
                    if is_encrypted:
                        member_name_in_zip += ENCRYPTED_FILE_EXTENSION

                    data_from_zip = None
                    try:
                        if cancel_event and cancel_event.is_set(): raise InterruptedError()
                        data_from_zip = zf.read(member_name_in_zip)
                    except InterruptedError: operation_cancelled_internally = True; break
                    except KeyError:
                        if not is_encrypted and original_arcname in zf.namelist():
                            try: data_from_zip = zf.read(original_arcname)
                            except Exception as e_rf: log_callback(f"Échec lecture fallback '{original_arcname}': {e_rf}")
                        if data_from_zip is None: log_callback(f"ERREUR: Membre '{member_name_in_zip}' ou '{original_arcname}' non trouvé.")
                    except RuntimeError as e_rt_m:
                        if password and "password" in str(e_rt_m).lower(): return False, f"PasswordErrorFile:{member_name_in_zip}"
                        log_callback(f"Erreur Runtime ZIP lecture '{member_name_in_zip}': {e_rt_m}")
                    
                    if operation_cancelled_internally: break

                    if data_from_zip is not None:
                        data_to_process = data_from_zip
                        if is_encrypted:
                            if cancel_event and cancel_event.is_set(): operation_cancelled_internally = True; break
                            if not CRYPTOGRAPHY_AVAILABLE: log_callback("ERREUR: Crypto non dispo."); data_to_process = None
                            elif not password: return False, f"PasswordNeededForDecryption:{original_arcname}"
                            else:
                                salt_h, iv_h, tag_h = item_meta.get("crypto_salt_hex"), item_meta.get("crypto_iv_hex"), item_meta.get("crypto_tag_hex")
                                if not (salt_h and iv_h and tag_h): data_to_process = None
                                else:
                                    salt, iv, tag = bytes.fromhex(salt_h), bytes.fromhex(iv_h), bytes.fromhex(tag_h)
                                    data_to_process = decrypt_data(data_from_zip, password, salt, iv, tag, log_callback)
                                    if data_to_process is None: return False, f"PasswordErrorDecryption:{original_arcname}"
                        
                        if operation_cancelled_internally: break
                        if data_to_process is not None:
                            final_data_to_write = None
                            if CLASSIC_COMPRESSORS_LOADED:
                                if cancel_event and cancel_event.is_set(): operation_cancelled_internally = True; break
                                params_from_meta = item_meta.get("compression_params", {})
                                if comp_method == METHOD_BZIP2: final_data_to_write = bzip2_decompress(data_to_process, params_from_meta, log_callback)
                                elif comp_method == METHOD_LZMA: final_data_to_write = lzma_decompress(data_to_process, params_from_meta, log_callback)
                                elif comp_method == METHOD_ZSTD: final_data_to_write = zstd_decompress(data_to_process, params_from_meta, log_callback)
                                elif comp_method == METHOD_BROTLI: final_data_to_write = brotli_decompress(data_to_process, params_from_meta, log_callback)
                                elif comp_method == METHOD_DEFLATE: final_data_to_write = deflate_decompress(data_to_process, params_from_meta, log_callback)
                                elif comp_method == METHOD_STORED: final_data_to_write = stored_decompress(data_to_process, params_from_meta, log_callback)
                                else: final_data_to_write = data_to_process
                                if final_data_to_write is None and comp_method != METHOD_STORED: log_callback(f"Échec décomp {comp_method} pour {original_arcname}.")
                            else: final_data_to_write = data_to_process
                            
                            if final_data_to_write is not None:
                                if cancel_event and cancel_event.is_set(): operation_cancelled_internally = True; break
                                try:
                                    os.makedirs(os.path.dirname(output_final_path), exist_ok=True)
                                    with open(output_final_path, "wb") as f_out: f_out.write(final_data_to_write)
                                    files_written_by_custom_logic.add(output_final_path)
                                except Exception as e_write: log_callback(f"ERREUR Écriture fichier '{output_final_path}': {e_write}")
                
                processed_items_count += 1
                if callable(progress_callback) and total_items_from_meta > 0:
                    progress_callback(processed_items_count, total_items_from_meta)
            
            if operation_cancelled_internally: return False, "Cancelled"

            log_callback("[AIC_HANDLER_DECOMP] Phase extraction générale des membres ZIP restants (sécurité)...")
            for member_info in zf.infolist():
                if cancel_event and cancel_event.is_set(): operation_cancelled_internally = True; break
                # ... (le reste de la boucle de fallback comme avant)
            if operation_cancelled_internally: return False, "Cancelled"

            if callable(progress_callback) and total_items_from_meta > 0 and processed_items_count >= total_items_from_meta:
                 progress_callback(total_items_from_meta, total_items_from_meta)

            log_callback(f"[AIC_HANDLER_DECOMP] Fin de la décompression AIC dans '{output_dir}'.")
            return True, "Success"
            
    except InterruptedError: return False, "Cancelled"
    except zipfile.BadZipFile: return False, "BadZipFile"
    except Exception as e: 
        log_callback(f"[AIC_HANDLER_DECOMP] ERREUR majeure décompression AIC: {e}"); import traceback
        log_callback(f"[AIC_HANDLER_DECOMP] Traceback: {traceback.format_exc()}"); return False, f"UnknownError: {e}"


def extract_archive(archive_path, output_dir, password=None, 
                    log_callback=_default_log, progress_callback=None, 
                    cancel_event=None, conflict_policy='overwrite'): # AJOUT de conflict_policy
    log_callback(f"[AIC_HANDLER] Point d'entrée extraction pour '{os.path.basename(archive_path)}' (Politique Conflit: {conflict_policy})")
    
    if cancel_event and cancel_event.is_set():
        return False, "CancelledBeforeStart"
        
    if not os.path.exists(archive_path):
        log_callback(f"[AIC_HANDLER] ERREUR: Archive '{archive_path}' non trouvée."); return False, "FileNotFound"
    try:
        os.makedirs(output_dir, exist_ok=True)
    except Exception as e:
        log_callback(f"[AIC_HANDLER] ERREUR création dossier sortie '{output_dir}': {e}"); return False, "OutputDirError"

    _, extension = os.path.splitext(archive_path)
    extension = extension.lower()
    
    analysis_for_extract = "unknown_type"
    if AI_ANALYZER_AVAILABLE:
        if cancel_event and cancel_event.is_set(): return False, "Cancelled"
        analysis_for_extract = analyze_file_content(archive_path, log_callback=log_callback) 
    
    if cancel_event and cancel_event.is_set(): return False, "Cancelled"

    # Cas 1: Archive AIC
    if extension == DEFAULT_AIC_EXTENSION.lower() or analysis_for_extract == "aic_custom_format": 
        log_callback(f"[AIC_HANDLER] Format AIC détecté. Appel de decompress_aic.")
        # Passer la politique à decompress_aic
        return decompress_aic(archive_path, output_dir, password, log_callback, progress_callback, cancel_event, conflict_policy)
    
    # Cas 2: Archive ZIP standard
    elif extension == ".zip" or analysis_for_extract == "zip_archive":
        log_callback(f"[AIC_HANDLER] Format .zip standard détecté pour '{os.path.basename(archive_path)}'.")
        success_overall, status_overall = False, "InitErrorZIP"
        num_zip_members = 0; processed_zip_members = 0; operation_cancelled_internally = False
        try:
            with zipfile.ZipFile(archive_path, 'r') as zf_zip:
                members_to_extract = []; 
                try:
                    if cancel_event and cancel_event.is_set(): raise InterruptedError("Annulation avant zf_zip.infolist()")
                    members_to_extract = zf_zip.infolist()
                    num_zip_members = len(members_to_extract)
                    log_callback(f"[AIC_HANDLER] Archive ZIP contient {num_zip_members} membres.")
                except Exception as e_infolist_zip:
                    log_callback(f"[AIC_HANDLER] ERREUR listage membres ZIP: {e_infolist_zip}"); status_overall = "BadZipFile_Header"
                
                if not operation_cancelled_internally and num_zip_members > 0 and status_overall == "InitErrorZIP":
                    success_overall = True 
                    if callable(progress_callback): progress_callback(0, num_zip_members)
                    for member in members_to_extract:
                        if cancel_event and cancel_event.is_set():
                            operation_cancelled_internally = True; status_overall = "Cancelled"; success_overall = False; break
                        
                        target_path = os.path.join(output_dir, member.filename)
                        
                        # --- GESTION DE CONFLIT POUR ZIP ---
                        if os.path.exists(target_path):
                            if member.is_dir():
                                pass # C'est un dossier, l'extraction créera le contenu dedans
                            elif conflict_policy == 'skip':
                                log_callback(f"[AIC_HANDLER] AVERT: Fichier ZIP '{member.filename}' existe déjà, ignoré.")
                                processed_zip_members += 1
                                if callable(progress_callback): progress_callback(processed_zip_members, num_zip_members)
                                continue # Passer au membre suivant
                            elif conflict_policy == 'rename':
                                 log_callback(f"[AIC_HANDLER] AVERT: Renommage non implémenté pour ZIP. Fichier '{member.filename}' ignoré.")
                                 processed_zip_members += 1
                                 if callable(progress_callback): progress_callback(processed_zip_members, num_zip_members)
                                 continue
                            elif conflict_policy == 'overwrite':
                                log_callback(f"[AIC_HANDLER] Info: Fichier ZIP '{member.filename}' existe déjà, sera écrasé.")
                        
                        log_callback(f"[AIC_HANDLER] Extraction membre ZIP : {member.filename}")
                        try: zf_zip.extract(member, path=output_dir, pwd=password.encode('utf-8') if password else None)
                        except Exception as e_member_ex:
                            log_callback(f"[AIC_HANDLER] Erreur extraction membre ZIP '{member.filename}': {e_member_ex}")
                            success_overall = False; 
                            if status_overall == "Success" or status_overall == "InitErrorZIP" : status_overall = f"MemberExtractErrorZIP"

                        processed_zip_members += 1
                        if callable(progress_callback): progress_callback(processed_zip_members, num_zip_members)
                    
                    if not operation_cancelled_internally and success_overall: status_overall = "Success"
                    
        except zipfile.BadZipFile: status_overall = "BadZipFile"; success_overall = False
        except RuntimeError as e_rt: 
            if password and ("password" in str(e_rt).lower()): status_overall = "PasswordErrorZIP"
            else: status_overall = f"RuntimeErrorZIP: {e_rt}"
            success_overall = False
        except InterruptedError: operation_cancelled_internally = True
        except Exception as e_ex_zip: status_overall = f"UnknownErrorZIP: {e_ex_zip}"; success_overall = False
        
        if operation_cancelled_internally: status_overall = "Cancelled"; success_overall = False
        if callable(progress_callback) and num_zip_members > 0: 
            final_count = processed_zip_members if success_overall else 0
            progress_callback(final_count, num_zip_members)
        return success_overall, status_overall
            
    # Cas 3 & 4: RAR et 7z (appellent external_handlers)
    elif extension == ".rar" or analysis_for_extract == "rar_archive":
        log_callback(f"[AIC_HANDLER] Format .rar détecté. Appel de external_handlers.decompress_rar.")
        # Passer la politique à decompress_rar
        return external_handlers.decompress_rar(archive_path, output_dir, password, log_callback, progress_callback, cancel_event, conflict_policy)
            
    elif extension == ".7z" or analysis_for_extract == "7z_archive":
        log_callback(f"[AIC_HANDLER] Format .7z détecté. Appel de external_handlers.decompress_7z.")
        # La politique de conflit ne s'applique pas car 7z utilise extractall pour la robustesse.
        # On passe quand même les arguments pour la cohérence, mais decompress_7z ne les utilisera pas.
        return external_handlers.decompress_7z(archive_path, output_dir, password, log_callback, progress_callback, cancel_event)
            
    else: # Fallback pour les .zip sans extension
        if zipfile.is_zipfile(archive_path):
            log_callback(f"[AIC_HANDLER] Type non reconnu mais semble ZIP, tentative d'extraction...")
            # Ici aussi, on devrait appliquer la politique de conflit. Pour l'instant, on utilise extractall qui écrase.
            try:
                with zipfile.ZipFile(archive_path, 'r') as zf:
                    zf.extractall(path=output_dir, pwd=password.encode('utf-8') if password else None)
                return True, "SuccessZipFallback"
            except Exception as e_zip_fb:
                return False, f"ZipFallbackError: {e_zip_fb}"

        log_callback(f"[AIC_HANDLER] Format archive non supporté: '{os.path.basename(archive_path)}'")
        return False, "UnsupportedFormat"


# --- Initialisation de external_handlers (doit être à la fin) ---
# (Cette section reste la même)
