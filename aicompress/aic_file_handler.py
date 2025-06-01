# aicompress/aic_file_handler.py (Version du 21 Mai - Correction TypeError progress_callback)

from . import external_handlers # Importer notre nouveau toolkit
import os
import zipfile 
import shutil
import json
import numpy as np
import tempfile # Pour créer des dossiers temporaires
import shutil   # Pour supprimer les dossiers temporaires
# --- Fonction de Log par Défaut ---
def _default_log(message):
    print(message)

# --- Initialisation des Flags de Disponibilité Globaux ---
PIL_AVAILABLE = False; KERAS_AVAILABLE = False; AI_ANALYZER_AVAILABLE = False 
RARFILE_AVAILABLE = False; IMPORTED_RARFILE_MODULE = None 
CRYPTOGRAPHY_AVAILABLE = False; ZSTD_AVAILABLE = False 
CLASSIC_COMPRESSORS_LOADED = False; AE_ENGINE_LOADED = False 
ORCHESTRATOR_LOADED_SUCCESSFULLY = False; PY7ZR_SUPPORT_AVAILABLE = False 

# --- Fallback Functions ---
def _fb_analyze_content(fp, lc=_default_log): lc(f"[AIC_HANDLER_FB] analyze_file_content"); return "analyzer_unavailable"
def _fb_get_features(fp, lc=_default_log): lc(f"[AIC_HANDLER_FB] get_file_features"); return {"type":"analyzer_unavailable","error":True}
def _fb_encrypt(d,p,l): lc(f"[AIC_HANDLER_FB] encrypt"); return d,None,None,None 
def _fb_decrypt(e,p,s,i,t,l): lc(f"[AIC_HANDLER_FB] decrypt"); return e
def _fb_classic_comp(d,p,l): lc(f"[AIC_HANDLER_FB] classic_compress"); return None
def _fb_classic_decomp(d,p,l): lc(f"[AIC_HANDLER_FB] classic_decompress"); return None
_fb_stored_comp = lambda d,p=None,l=_default_log:d; _fb_stored_decomp = lambda d,p=None,l=_default_log:d
def _fb_ensure_ae(lc): lc(f"[AIC_HANDLER_FB] ensure_ae"); return False
def _fb_encode_ae(ip,lc): lc(f"[AIC_HANDLER_FB] encode_ae"); return {"error":True}
def _fb_decode_ae(ldb,lsi,qp,od,ops,lc): lc(f"[AIC_HANDLER_FB] decode_ae"); return False
def _fb_get_comp_settings(fp,ai,lc): lc(f"[AIC_HANDLER_FB] get_comp_settings"); return "DEFLATE", {"level":6}

analyze_file_content, get_file_features = _fb_analyze_content, _fb_get_features
encrypt_data, decrypt_data = _fb_encrypt, _fb_decrypt
stored_compress, stored_decompress = _fb_stored_comp, _fb_stored_decomp
deflate_compress, deflate_decompress = _fb_classic_comp, _fb_classic_decomp
bzip2_compress, bzip2_decompress = _fb_classic_comp, _fb_classic_decomp
lzma_compress, lzma_decompress = _fb_classic_comp, _fb_classic_decomp
zstd_compress, zstd_decompress = _fb_classic_comp, _fb_classic_decomp
brotli_compress, brotli_decompress = _fb_classic_comp, _fb_classic_decomp 
ensure_ae_models_loaded, encode_image_with_ae, decode_latent_to_image_ae = _fb_ensure_ae, _fb_encode_ae, _fb_decode_ae
get_compression_settings = _fb_get_comp_settings
METHOD_STORED,METHOD_DEFLATE,METHOD_BZIP2,METHOD_LZMA,METHOD_ZSTD,METHOD_BROTLI = "STORED","DEFLATE","BZIP2","LZMA","ZSTD","BROTLI"

try: from PIL import Image, ImageOps; PIL_AVAILABLE = True; _default_log("[AIC_HANDLER] Pillow OK.")
except ImportError: _default_log("AVERT (aic_handler.py): Pillow non trouvée.")
try: import tensorflow as tf; KERAS_AVAILABLE = True; _default_log("[AIC_HANDLER] TensorFlow/Keras OK.")
except ImportError: _default_log("AVERT (aic_handler.py): TensorFlow/Keras non trouvé.")
try:
    from .ai_analyzer import get_file_features as _real_gff, analyze_file_content as _real_afc, AI_ANALYZER_AVAILABLE as _ai_flag
    if _ai_flag: get_file_features, analyze_file_content, AI_ANALYZER_AVAILABLE = _real_gff, _real_afc, True; _default_log("[AIC_HANDLER] ai_analyzer OK.")
    else: _default_log("[AIC_HANDLER] ai_analyzer chargé mais non dispo (interne).")
except ImportError as e: _default_log(f"AVERT (aic_handler.py): Échec import ai_analyzer: {e}")
try: 
    import rarfile as actual_rarfile_module_import 
    IMPORTED_RARFILE_MODULE = actual_rarfile_module_import
    RARFILE_AVAILABLE = True; _default_log("[AIC_HANDLER] rarfile OK.")
except ImportError: _default_log("AVERT (aic_handler.py): rarfile non installée.")
try:
    from .crypto_utils import encrypt_data as _re_cu, decrypt_data as _rd_cu, CRYPTOGRAPHY_AVAILABLE as _cf_cu_flag
    if _cf_cu_flag: encrypt_data, decrypt_data, CRYPTOGRAPHY_AVAILABLE = _re_cu, _rd_cu, True; _default_log("[AIC_HANDLER] crypto_utils OK.")
    else: _default_log("AVERT (aic_handler.py): crypto_utils chargé, mais cryptography indisponible.")
except ImportError as e: _default_log(f"AVERT (aic_handler.py): crypto_utils.py non trouvé: {e}")
try:
    from .classic_compressors import (stored_compress as _cs_s, deflate_compress as _cs_d, bzip2_compress as _cs_b, lzma_compress as _cs_l, zstd_compress as _cs_z, brotli_compress as _cs_br,
                                      stored_decompress as _cs_sd, deflate_decompress as _cs_dd, bzip2_decompress as _cs_bd, lzma_decompress as _cs_ld, zstd_decompress as _cs_zd, brotli_decompress as _cs_brd,
                                      METHOD_STORED as _M_S_cl, METHOD_DEFLATE as _M_D_cl, METHOD_BZIP2 as _M_B_cl, METHOD_LZMA as _M_L_cl, METHOD_ZSTD as _M_Z_cl, METHOD_BROTLI as _M_BR_cl,
                                      ZSTD_AVAILABLE as ZSTD_FLAG_cl, BROTLI_AVAILABLE as BROTLI_FLAG_cl, CLASSIC_COMPRESSORS_READY as CC_READY_FLAG_cl )
    if CC_READY_FLAG_cl:
        stored_compress=_cs_s; deflate_compress=_cs_d; bzip2_compress=_cs_b; lzma_compress=_cs_l; zstd_compress=_cs_z; brotli_compress=_cs_br
        stored_decompress=_cs_sd; deflate_decompress=_cs_dd; bzip2_decompress=_cs_bd; lzma_decompress=_cs_ld; zstd_decompress=_cs_zd; brotli_decompress=_cs_brd
        METHOD_STORED,METHOD_DEFLATE,METHOD_BZIP2,METHOD_LZMA,METHOD_ZSTD,METHOD_BROTLI=_M_S_cl,_M_D_cl,_M_B_cl,_M_L_cl,_M_Z_cl,_M_BR_cl
        CLASSIC_COMPRESSORS_LOADED = True; ZSTD_AVAILABLE = ZSTD_FLAG_cl; BROTLI_AVAILABLE = BROTLI_FLAG_cl; _default_log("[AIC_HANDLER] classic_compressors OK.")
        if not ZSTD_AVAILABLE: _default_log("[AIC_HANDLER] AVERT: Zstd non dispo via classic_compressors.")
        if not BROTLI_AVAILABLE: _default_log("[AIC_HANDLER] AVERT: Brotli non dispo via classic_compressors.")
    else: _default_log("[AIC_HANDLER] classic_compressors chargé mais non opérationnel.")
except ImportError as e: _default_log(f"AVERT (aic_handler.py): classic_compressors.py non trouvé: {e}")
try:
    from .ae_engine import (ensure_cifar10_color_ae_models_loaded as _rae_ensure, encode_image_with_cifar10_ae as _rae_enc,
                            decode_latent_to_image_cifar10_ae as _rae_dec, AE_ENGINE_LOADED as _AE_ENG_FLAG_MOD,
                            PIL_AVAILABLE_AE as _ae_pil_flag, KERAS_AVAILABLE_AE as _ae_keras_flag)
    if _AE_ENG_FLAG_MOD:
        ensure_ae_models_loaded,encode_image_with_ae,decode_latent_to_image_ae=_rae_ensure,_rae_enc,_rae_dec
        PIL_AVAILABLE_FOR_AE = _ae_pil_flag; KERAS_AVAILABLE_FOR_AE = _ae_keras_flag; AE_ENGINE_LOADED = True; 
        _default_log("[AIC_HANDLER] ae_engine OK.")
    else: _default_log("[AIC_HANDLER] ae_engine chargé mais non opérationnel (interne).")
except ImportError as e: _default_log(f"AVERT (aic_handler.py): ae_engine.py non trouvé: {e}")
try:
    from .orchestrator import get_compression_settings as _real_gcs_orch, ORCHESTRATOR_IS_READY as ORCH_FLAG_MOD
    if ORCH_FLAG_MOD: get_compression_settings = _real_gcs_orch; ORCHESTRATOR_LOADED_SUCCESSFULLY = True; _default_log("[AIC_HANDLER] orchestrator et son modèle sont prêts.")
    else: _default_log("[AIC_HANDLER] orchestrator chargé, mais son modèle interne non prêt.")
except ImportError as e: _default_log(f"AVERT (aic_handler.py): orchestrator.py non trouvé: {e}")

METADATA_FILENAME = "aicompress_metadata.json"; DEFAULT_AIC_EXTENSION = ".aic"
LATENT_FILE_EXTENSION = ".aic_latent"; ENCRYPTED_FILE_EXTENSION = ".aic_enc"
# NOUVELLE Fonction helper pour compter les fichiers
def _count_total_individual_files(input_paths, log_callback=_default_log):
    total_files = 0
    for item_path in input_paths:
        if not os.path.exists(item_path):
            continue # Ignorer les chemins non existants pour le comptage
        if os.path.isfile(item_path):
            total_files += 1
        elif os.path.isdir(item_path):
            for _, _, files_in_dir in os.walk(item_path):
                total_files += len(files_in_dir)
    log_callback(f"[AIC_HANDLER] Nombre total de fichiers individuels à traiter: {total_files}")
    return total_files

# Dans aicompress/aic_file_handler.py

# ... (gardez tous les imports et les autres fonctions comme _default_log, les fallbacks, 
#      get_compression_settings, compress_to_aic, decompress_aic, etc.)
def _is_archive_file(file_path, log_callback=_default_log):
    """Vérifie si un fichier est un type d'archive que nous pouvons décompresser récursivement."""
    # Liste des types d'archives que external_handlers.extract_archive peut gérer
    SUPPORTED_RECURSIVE_TYPES = [
        'zip_archive',
        'rar_archive',
        '7z_archive',
        'aic_custom_format' # Notre propre format
        # On pourrait ajouter 'gzip_archive', 'tar_archive' ici si on ajoutait leur support
    ]
    # Liste des extensions pour le fallback si l'analyseur d'IA n'est pas disponible
    SUPPORTED_RECURSIVE_EXTENSIONS = ['.zip', '.rar', '.7z', DEFAULT_AIC_EXTENSION.lower()]

    if not AI_ANALYZER_AVAILABLE:
        ext = os.path.splitext(file_path)[1].lower()
        return ext in SUPPORTED_RECURSIVE_EXTENSIONS
    
    analysis_type = analyze_file_content(file_path, log_callback=log_callback)
    return analysis_type in SUPPORTED_RECURSIVE_TYPES



def _process_and_add_file_to_aic(zf, file_path_on_disk, arcname_to_use, password_compress, 
                                 log_callback, progress_state, cancel_event=None): # AJOUT cancel_event
    item_basename = os.path.basename(file_path_on_disk)
    log_callback(f"[AIC_HANDLER] Début traitement pour: {item_basename}")

    # Vérifier l'annulation au début du traitement d'un fichier
    if cancel_event and cancel_event.is_set():
        log_callback(f"[AIC_HANDLER] Annulation détectée avant de traiter {item_basename}.")
        # Retourner un item_meta indiquant l'annulation ou un statut spécial
        return {"original_name": arcname_to_use, "status": "cancelled_before_processing"}

    item_meta = { 
        "original_name": arcname_to_use, "status": "processing", "analysis": "N/A", 
        "size_original_bytes": 0, "compression_method_used": METHOD_STORED, 
        "compression_params": {}, "encrypted": False, "crypto_salt_hex": None, 
        "crypto_iv_hex": None, "crypto_tag_hex": None, "original_image_dims": None, 
        "latent_quant_params": None, "latent_shape_info": None 
    }
    try: 
        item_meta["size_original_bytes"] = os.path.getsize(file_path_on_disk)
    except OSError as e_size: 
        log_callback(f"[AIC_HANDLER] ERREUR: Taille de {file_path_on_disk}: {e_size}")
        item_meta["status"]="read_error_size"; # La progression sera mise à jour à la fin de la fonction
        # Inutile de vérifier cancel_event ici car on retourne déjà
        if progress_state and len(progress_state) == 3: progress_state[0] += 1 # Marquer comme traité pour la progression
        return item_meta

    if AI_ANALYZER_AVAILABLE: 
        item_meta["analysis"] = analyze_file_content(file_path_on_disk, log_callback=log_callback)
    
    if cancel_event and cancel_event.is_set(): log_callback(f"[AIC_HANDLER] Annulation après analyse {item_basename}."); item_meta["status"] = "cancelled"; return item_meta

    comp_method_name_from_ia, comp_params_dict_from_ia = get_compression_settings(file_path_on_disk, item_meta["analysis"], log_callback=log_callback)
    item_meta["compression_method_used"] = comp_method_name_from_ia; item_meta["compression_params"] = comp_params_dict_from_ia
    
    if cancel_event and cancel_event.is_set(): log_callback(f"[AIC_HANDLER] Annulation après décision IA pour {item_basename}."); item_meta["status"] = "cancelled"; return item_meta

    final_arcname_for_zip_member = arcname_to_use; data_to_process_further = b""
    
    if comp_method_name_from_ia == "MOTEUR_AE_CIFAR10_COLOR":
        if AE_ENGINE_LOADED:
            # Vérifier avant l'encodage AE potentiellement long
            if cancel_event and cancel_event.is_set(): log_callback(f"[AIC_HANDLER] Annulation avant encodage AE {item_basename}."); item_meta["status"] = "cancelled"; return item_meta
            ae_encoding_result = encode_image_with_ae(file_path_on_disk, log_callback=log_callback) # Supposer que encode_image_with_ae ne gère pas cancel_event en interne pour l'instant
            if cancel_event and cancel_event.is_set(): log_callback(f"[AIC_HANDLER] Annulation après encodage AE {item_basename}."); item_meta["status"] = "cancelled"; return item_meta

            if ae_encoding_result and not ae_encoding_result.get("error"):
                for key_ae in ["original_image_dims","latent_quant_params","latent_shape_info"]:
                    if key_ae in ae_encoding_result: item_meta[key_ae] = ae_encoding_result[key_ae]
                data_to_process_further = ae_encoding_result["latent_data_bytes"]; final_arcname_for_zip_member = arcname_to_use + LATENT_FILE_EXTENSION
            else: 
                log_msg=ae_encoding_result.get('error_message','Erreur AE') if ae_encoding_result else 'Pas de résultat AE'
                log_callback(f"[AIC_HANDLER] Échec AE {item_basename}: {log_msg}. Fallback."); comp_method_name_from_ia=METHOD_DEFLATE; comp_params_dict_from_ia={"level":1}
                item_meta["compression_method_used"]=comp_method_name_from_ia; item_meta["compression_params"]=comp_params_dict_from_ia
        else: 
             log_callback(f"[AIC_HANDLER] Moteur AE non prêt. Fallback DEFLATE pour {item_basename}."); comp_method_name_from_ia=METHOD_DEFLATE; comp_params_dict_from_ia={"level":1}
             item_meta["compression_method_used"]=comp_method_name_from_ia; item_meta["compression_params"]=comp_params_dict_from_ia
    
    if comp_method_name_from_ia != "MOTEUR_AE_CIFAR10_COLOR": 
        try:
            with open(file_path_on_disk, 'rb') as f_orig: data_to_process_further = f_orig.read()
        except Exception as e_read: 
            log_callback(f"[AIC_HANDLER] ERREUR lecture fichier {item_basename}: {e_read}"); item_meta["status"]="read_error"; 
            if progress_state and len(progress_state) == 3: progress_state[0] += 1; 
            return item_meta

    if cancel_event and cancel_event.is_set(): log_callback(f"[AIC_HANDLER] Annulation avant compression/chiffrement {item_basename}."); item_meta["status"] = "cancelled"; return item_meta

    data_after_internal_comp = data_to_process_further
    # ... (logique de compression classique comme avant, appelant bzip2_compress, etc.)
    # Ces fonctions de classic_compressors sont rapides ou bloquantes.
    # On pourrait ajouter un check cancel_event avant chaque appel si elles sont très longues pour certains cas.
    # Pour l'instant, on vérifie avant ce bloc.
    if CLASSIC_COMPRESSORS_LOADED:
        if comp_method_name_from_ia == METHOD_BZIP2: data_after_internal_comp = bzip2_compress(data_to_process_further, comp_params_dict_from_ia, log_callback)
        elif comp_method_name_from_ia == METHOD_LZMA: data_after_internal_comp = lzma_compress(data_to_process_further, comp_params_dict_from_ia, log_callback)
        elif comp_method_name_from_ia == METHOD_ZSTD: data_after_internal_comp = zstd_compress(data_to_process_further, comp_params_dict_from_ia, log_callback)
        elif comp_method_name_from_ia == METHOD_BROTLI: data_after_internal_comp = brotli_compress(data_to_process_further, comp_params_dict_from_ia, log_callback)
        elif comp_method_name_from_ia == METHOD_DEFLATE: data_after_internal_comp = deflate_compress(data_to_process_further, comp_params_dict_from_ia, log_callback)
        elif comp_method_name_from_ia == METHOD_STORED: data_after_internal_comp = stored_compress(data_to_process_further, comp_params_dict_from_ia, log_callback)
        if data_after_internal_comp is None : 
            log_callback(f"[AIC_HANDLER] Échec {comp_method_name_from_ia}. Utilisation données originales."); data_after_internal_comp = data_to_process_further
            if comp_method_name_from_ia != "MOTEUR_AE_CIFAR10_COLOR": item_meta["compression_method_used"]=METHOD_STORED; item_meta["compression_params"]={}; comp_method_name_from_ia=METHOD_STORED
    elif comp_method_name_from_ia not in ["MOTEUR_AE_CIFAR10_COLOR", METHOD_STORED, METHOD_DEFLATE, METHOD_BROTLI]:
        log_callback(f"[AIC_HANDLER] AVERT: classic_compressors.py non chargé, {comp_method_name_from_ia} non applicable. Fallback STORED.")
        item_meta["compression_method_used"]=METHOD_STORED; item_meta["compression_params"]={}; comp_method_name_from_ia=METHOD_STORED
    
    if cancel_event and cancel_event.is_set(): log_callback(f"[AIC_HANDLER] Annulation avant chiffrement pour {item_basename}."); item_meta["status"] = "cancelled"; return item_meta

    data_to_write_to_zip = data_after_internal_comp
    if password_compress and data_to_write_to_zip and CRYPTOGRAPHY_AVAILABLE:
        # Le chiffrement peut être un peu long
        encrypted_bundle = encrypt_data(data_to_write_to_zip, password_compress, log_callback=log_callback)
        if cancel_event and cancel_event.is_set(): log_callback(f"[AIC_HANDLER] Annulation après chiffrement {item_basename}."); item_meta["status"] = "cancelled"; return item_meta
        if encrypted_bundle and encrypted_bundle[0] is not None and encrypted_bundle[1] is not None:
            data_to_write_to_zip,salt,iv,tag=encrypted_bundle; item_meta["encrypted"]=True; item_meta["crypto_salt_hex"]=salt.hex(); item_meta["crypto_iv_hex"]=iv.hex(); item_meta["crypto_tag_hex"]=tag.hex()
            final_arcname_for_zip_member += ENCRYPTED_FILE_EXTENSION
        else: log_callback(f"[AIC_HANDLER] AVERT: Échec chiffrement {item_basename}.")
    
    if data_to_write_to_zip is not None: 
        zf.writestr(final_arcname_for_zip_member, data_to_write_to_zip, compress_type=zipfile.ZIP_STORED)
        log_callback(f"[AIC_HANDLER] Données '{item_basename}' écrites: '{final_arcname_for_zip_member}'.")
        item_meta["status"] = "completed" # Marquer comme complété
    else: 
        log_callback(f"[AIC_HANDLER] AVERT: Pas de données à écrire pour {item_basename}.")
        item_meta["status"] = "write_error_no_data"

    if progress_state and len(progress_state) == 3:
        progress_state[0] += 1 
        current_processed_count, total_files, callback_func = progress_state
        if callable(callback_func) and total_files > 0:
            callback_func(current_processed_count, total_files)
            
    return item_meta


# Dans aicompress/aic_file_handler.py
# Assurez-vous que _count_total_individual_files et _is_archive_file sont définies avant cette fonction,
# et que _process_and_add_file_to_aic est définie et accepte bien les 7 arguments
# (zf, file_path_on_disk, arcname_to_use, password_compress, log_callback, progress_state, cancel_event)

def compress_to_aic(input_paths, output_aic_path, password_compress=None, 
                    log_callback=_default_log, progress_callback=None, 
                    cancel_event=None, recursively_optimize=False):
    log_callback(f"[AIC_HANDLER] Compression AIC vers '{output_aic_path}' (Chiffré: {'Oui' if password_compress else 'Non'})...")
    if recursively_optimize:
        log_callback("[AIC_HANDLER] Mode d'optimisation récursive des archives ACTIVÉ.")
        
    all_items_metadata = []
    
    # total_individual_files est défini ici et devrait être accessible dans le except
    total_individual_files = _count_total_individual_files(input_paths, log_callback)
    progress_state_list = [0, total_individual_files, progress_callback] 

    if callable(progress_callback) and total_individual_files > 0:
        progress_callback(0, total_individual_files)

    operation_cancelled_flag = False # Doit être accessible par process_item_recursively

    # --- Définition de la fonction interne ---
    def process_item_recursively(item_path, base_arc_path=""):
        nonlocal operation_cancelled_flag # Pour modifier le flag de la portée externe
        
        if cancel_event and cancel_event.is_set():
            operation_cancelled_flag = True
            return

        item_basename = os.path.basename(item_path)
        # S'assurer que current_arc_path est bien construit
        current_arc_path = os.path.join(base_arc_path, item_basename) if base_arc_path else item_basename

        if not os.path.exists(item_path):
            log_callback(f"[AIC_HANDLER] Item non trouvé (ignoré) : {item_path}")
            all_items_metadata.append({"original_name": current_arc_path, "status": "not_found"})
            return

        if recursively_optimize and os.path.isfile(item_path) and _is_archive_file(item_path, log_callback):
            log_callback(f"--- Début de l'optimisation de l'archive interne : {current_arc_path} ---")
            
            # L'import local de external_handlers n'est PAS nécessaire pour extract_archive
            # from . import external_handlers # SUPPRIMER CET IMPORT LOCAL
            
            temp_dir = tempfile.mkdtemp(prefix="aicompress_rec_")
            log_callback(f"[AIC_HANDLER] Extraction de '{item_basename}' dans : {temp_dir}")
            
            try:
                # --- CORRECTION ICI : Appeler extract_archive directement ---
                # extract_archive est maintenant une fonction de ce même module (aic_file_handler.py)
                # ou importée dans __init__.py et accessible globalement si on importe depuis là.
                # Pour un appel interne, on peut l'appeler directement si elle est définie dans ce fichier.
                # Assumons qu'elle est définie dans ce fichier ou accessible.
                success_extract, status_extract = extract_archive( # Appel direct
                    item_path, temp_dir, password=None, 
                    log_callback=log_callback, 
                    progress_callback=None, 
                    cancel_event=cancel_event
                )
                # --- FIN CORRECTION ---
                
                if cancel_event and cancel_event.is_set(): operation_cancelled_flag = True
                
                if success_extract and not operation_cancelled_flag:
                    all_items_metadata.append({
                        "original_name": current_arc_path, 
                        "type_in_archive": "directory", 
                        "status": "recursively_processed"
                    })
                    for root, _, files_in_dir in os.walk(temp_dir):
                        if operation_cancelled_flag: break
                        for file_in_d in files_in_dir:
                            if cancel_event and cancel_event.is_set(): operation_cancelled_flag = True; break
                            
                            full_file_path_in_temp = os.path.join(root, file_in_d)
                            arc_path_for_subfile = os.path.join(current_arc_path, 
                                                                os.path.relpath(full_file_path_in_temp, temp_dir))
                            
                            meta = _process_and_add_file_to_aic(zf, full_file_path_in_temp, arc_path_for_subfile, 
                                                                password_compress, log_callback, 
                                                                progress_state_list, 
                                                                cancel_event)
                            all_items_metadata.append(meta)
                            if meta.get("status", "").startswith("cancelled"): operation_cancelled_flag = True; break
                        if operation_cancelled_flag: break
                    log_callback(f"--- Fin de l'optimisation de l'archive interne : {current_arc_path} ---")
                elif not operation_cancelled_flag: 
                    log_callback(f"[AIC_HANDLER] AVERT: Échec extraction archive interne '{current_arc_path}'. Traitée comme fichier normal. Status: {status_extract}")
                    all_items_metadata.append(_process_and_add_file_to_aic(zf, item_path, current_arc_path, password_compress, log_callback, progress_state_list, cancel_event))
                
                if operation_cancelled_flag:
                    log_callback(f"[AIC_HANDLER] Optimisation de '{current_arc_path}' annulée.")
                    return 

            finally:
                log_callback(f"[AIC_HANDLER] Nettoyage du dossier temporaire : {temp_dir}")
                shutil.rmtree(temp_dir, ignore_errors=True)
        
        elif os.path.isfile(item_path): 
            all_items_metadata.append(
                _process_and_add_file_to_aic(zf, item_path, current_arc_path, 
                                             password_compress, log_callback, 
                                             progress_state_list, cancel_event)
            )
        elif os.path.isdir(item_path):
            log_callback(f"[AIC_HANDLER] Traitement du dossier : {current_arc_path}")
            all_items_metadata.append({"original_name": current_arc_path, "type_in_archive": "directory", "status": "processed", "size_original_bytes": 0}) 
            for sub_item_name in sorted(os.listdir(item_path)): 
                if operation_cancelled_flag: break
                process_item_recursively(os.path.join(item_path, sub_item_name), current_arc_path) 
            if operation_cancelled_flag:
                log_callback(f"[AIC_HANDLER] Traitement du dossier '{current_arc_path}' annulé.")
                return
        
        if all_items_metadata and all_items_metadata[-1].get("status", "").startswith("cancelled"):
            operation_cancelled_flag = True
    # --- Fin de la fonction interne process_item_recursively ---

    try:
        with zipfile.ZipFile(output_aic_path, 'w') as zf: 
            for top_level_path in input_paths: # Item de premier niveau
                if operation_cancelled_flag: break # Vérifier avant de traiter le prochain item de haut niveau
                process_item_recursively(top_level_path)
            
            if operation_cancelled_flag:
                log_callback("[AIC_HANDLER] Opération de compression annulée. L'archive .aic peut être incomplète ou invalide.")
                return False, "Cancelled" 
            
            metadata_final_content = {"aicompress_version":"1.3-recursive","items_details":all_items_metadata, "global_encryption_hint":bool(password_compress and CRYPTOGRAPHY_AVAILABLE)}
            zf.writestr(METADATA_FILENAME,json.dumps(metadata_final_content,indent=4))
            
            if callable(progress_callback) and total_individual_files > 0:
                 progress_callback(progress_state_list[0], total_individual_files) 
            
            log_callback(f"[AIC_HANDLER] Métadonnées écrites. Compression AIC terminée: '{output_aic_path}'")
            return True,"Success"
    except Exception as e: 
        log_callback(f"[AIC_HANDLER] ERREUR MAJEURE compression: {e}"); import traceback
        log_callback(f"[AIC_HANDLER] Traceback: {traceback.format_exc()}"); 
        if callable(progress_callback): 
            # Assurer que total_individual_files est défini ici
            # Si l'erreur se produit avant sa définition complète, cela pourrait encore être un problème.
            # total_individual_files est défini en dehors du try, donc elle devrait être accessible.
            progress_callback(0, total_individual_files if total_individual_files > 0 else 1) 
        return False,f"Error: {e}"

# --- Fonction de Décompression Principale pour .aic (MODIFIÉE pour progress_callback) ---
def decompress_aic(aic_file_path, output_extract_path, password_decompress=None, 
                   log_callback=_default_log, progress_callback=None, cancel_event=None): # AJOUT cancel_event
    log_callback(f"[AIC_HANDLER_DECOMP] >>> Début Décompression AIC: '{os.path.basename(aic_file_path)}' vers '{output_extract_path}' (Mdp: {'Oui' if password_decompress else 'Non'})")    
    
    if cancel_event and cancel_event.is_set():
        log_callback("[AIC_HANDLER_DECOMP] Annulation détectée avant le début.")
        return False, "CancelledBeforeStart"
    
    if not os.path.exists(aic_file_path):
        log_callback(f"[AIC_HANDLER_DECOMP] ERREUR: Archive '{aic_file_path}' non trouvée.")
        if callable(progress_callback): progress_callback(0, 1) # Indiquer échec si barre active
        return False, "FileNotFound"
    
    total_items_from_meta = 0
    processed_items_count = 0
    operation_cancelled_internally = False

    try:
        os.makedirs(output_extract_path, exist_ok=True)
        log_callback(f"[AIC_HANDLER_DECOMP] Dossier de sortie assuré: {output_extract_path}")
        
        with zipfile.ZipFile(aic_file_path, 'r') as zf:
            log_callback(f"[AIC_HANDLER_DECOMP] Archive ZIP ouverte. Namelist: {zf.namelist()}")
            metadata = None
            metadata_loaded_successfully = False
            
            try: # Bloc try pour la lecture des métadonnées
                if cancel_event and cancel_event.is_set(): raise InterruptedError("Annulation avant lecture métadonnées")
                metadata_str = zf.read(METADATA_FILENAME)
                metadata = json.loads(metadata_str)
                log_callback("[AIC_HANDLER_DECOMP] Métadonnées lues et parsées.")
                metadata_loaded_successfully = True
                items_in_meta = metadata.get('items_details', [])
                total_items_from_meta = len(items_in_meta)
                if callable(progress_callback) and total_items_from_meta > 0 :
                    progress_callback(0, total_items_from_meta) 
            
            except InterruptedError: # Attraper notre propre InterruptedError
                log_callback("[AIC_HANDLER_DECOMP] Annulation pendant la lecture des métadonnées.")
                operation_cancelled_internally = True
            except Exception as e_meta: 
                log_callback(f"[AIC_HANDLER_DECOMP] ERREUR métadonnées: {e_meta}. Tentative ZIP standard...");
                try: 
                    if cancel_event and cancel_event.is_set(): raise InterruptedError("Annulation avant fallback extractall")
                    zf.extractall(path=output_extract_path, pwd=password_decompress.encode('utf-8') if password_decompress else None)
                    log_callback("[AIC_HANDLER_DECOMP] Extraction ZIP standard (fallback) réussie.")
                    if callable(progress_callback): progress_callback(1, 1) 
                    return True, "SuccessZipFallback"
                except InterruptedError: operation_cancelled_internally = True
                except RuntimeError as e_zip_fb: 
                    if password_decompress and ("password" in str(e_zip_fb).lower()):
                        if callable(progress_callback): progress_callback(0,1); return False, "PasswordErrorZipFallback"
                    log_callback(f"[AIC_HANDLER_DECOMP] Erreur runtime (fallback ZIP): {e_zip_fb}")
                    if callable(progress_callback): progress_callback(0,1); return False, f"ZipFallbackError: {e_zip_fb}"
                except Exception as e_gen_fb: 
                    log_callback(f"[AIC_HANDLER_DECOMP] Erreur générale (fallback ZIP): {e_gen_fb}")
                    if callable(progress_callback): progress_callback(0,1); return False, f"ZipFallbackGenericError: {e_gen_fb}"
            
            if operation_cancelled_internally: return False, "CancelledDuringMetadataRead"
            if not (metadata_loaded_successfully and metadata and "items_details" in metadata):
                log_callback("[AIC_HANDLER_DECOMP] AVERT: Métadonnées invalides. Fin anormale."); 
                if callable(progress_callback): progress_callback(0,1); return False, "InvalidMetadata" 
            
            log_callback(f"[AIC_HANDLER_DECOMP] Traitement de {total_items_from_meta} items des métadonnées...")
            files_written_by_custom_logic = set() 

            for item_meta in metadata["items_details"]:
                if cancel_event and cancel_event.is_set(): operation_cancelled_internally = True; break 
                
                original_arcname = item_meta.get("original_name")
                if not original_arcname: 
                    log_callback("[AIC_HANDLER_DECOMP] AVERT: item_meta sans original_name. Ignoré.")
                    processed_items_count += 1
                    if callable(progress_callback) and total_items_from_meta > 0: progress_callback(processed_items_count, total_items_from_meta)
                    continue
                
                output_final_path = os.path.join(output_extract_path, original_arcname)
                log_callback(f"[AIC_HANDLER_DECOMP] Préparation item méta: '{original_arcname}' -> '{output_final_path}'")

                if item_meta.get("status")=="not_found": log_callback(f" Ignoré (not_found): {original_arcname}")
                elif item_meta.get("type_in_archive")=="directory": 
                    if cancel_event and cancel_event.is_set(): operation_cancelled_internally = True; break
                    log_callback(f" Création dossier: {output_final_path}"); os.makedirs(output_final_path,exist_ok=True)
                    files_written_by_custom_logic.add(output_final_path)
                else: 
                    comp_method = item_meta.get("compression_method_used"); is_encrypted = item_meta.get("encrypted", False)
                    member_name_in_zip = original_arcname
                    if comp_method == "MOTEUR_AE_CIFAR10_COLOR": member_name_in_zip += LATENT_FILE_EXTENSION
                    if is_encrypted: member_name_in_zip += ENCRYPTED_FILE_EXTENSION
                    
                    data_from_zip = None; 
                    try: 
                        if cancel_event and cancel_event.is_set(): raise InterruptedError("Annulation avant lecture membre ZIP")
                        data_from_zip = zf.read(member_name_in_zip) 
                    except InterruptedError: operation_cancelled_internally = True; break
                    except KeyError: # ... (gestion KeyError comme avant, mais sans 'continue' direct pour la progression)
                        log_callback(f"[AIC_HANDLER_DECOMP] ERREUR: Membre '{member_name_in_zip}' non trouvé pour '{original_arcname}'.")
                        data_from_zip = None # Sera géré ci-dessous
                    except RuntimeError as e_rt_m: # ... (gestion RuntimeError comme avant)
                        if password_decompress and "password" in str(e_rt_m).lower(): return False, f"PasswordErrorFile:{member_name_in_zip}"
                        log_callback(f"[AIC_HANDLER_DECOMP] Erreur Runtime ZIP lecture '{member_name_in_zip}': {e_rt_m}"); data_from_zip = None
                    
                    if operation_cancelled_internally: break
                    if data_from_zip is None: log_callback(f"[AIC_HANDLER_DECOMP] AVERT: Pas de données lues pour {member_name_in_zip}.")
                    else:
                        log_callback(f"[AIC_HANDLER_DECOMP] Membre '{member_name_in_zip}' lu ({len(data_from_zip)} octets).")
                        data_to_process = data_from_zip
                        if is_encrypted:
                            if cancel_event and cancel_event.is_set(): operation_cancelled_internally = True; break
                            log_callback(f"[AIC_HANDLER_DECOMP] Déchiffrement de '{original_arcname}'...")
                            if not CRYPTOGRAPHY_AVAILABLE: log_callback("[AIC_HANDLER_DECOMP] ERREUR: Crypto non dispo."); data_to_process=None
                            elif not password_decompress: return False, f"PasswordNeededForDecryption:{original_arcname}"
                            else:
                                salt_h=item_meta.get("crypto_salt_hex"); iv_h=item_meta.get("crypto_iv_hex"); tag_h=item_meta.get("crypto_tag_hex")
                                if not (salt_h and iv_h and tag_h): log_callback(f" Infos déchiffrement manquantes pour {original_arcname}."); data_to_process = None
                                else: 
                                    salt=bytes.fromhex(salt_h); iv=bytes.fromhex(iv_h); tag=bytes.fromhex(tag_h)
                                    data_to_process = decrypt_data(data_from_zip, password_decompress, salt, iv, tag, log_callback=log_callback)
                                    if data_to_process is None: return False, f"PasswordErrorDecryption:{original_arcname}"
                        
                        if operation_cancelled_internally: break
                        if data_to_process is not None:
                            final_data_to_write = None; ae_output_already_saved = False
                            if comp_method == "MOTEUR_AE_CIFAR10_COLOR":
                                ae_output_already_saved = True 
                                if AE_ENGINE_LOADED and ensure_ae_models_loaded(log_callback=log_callback):
                                    if cancel_event and cancel_event.is_set(): operation_cancelled_internally = True; break
                                    try:
                                        # ... (logique décodage AE comme avant)
                                        latent_shape_info=item_meta.get("latent_shape_info"); quant_params=item_meta.get("latent_quant_params"); original_dims=item_meta.get("original_image_dims")
                                        if not (latent_shape_info and quant_params): log_callback(f"[AIC_HANDLER_DECOMP] ERREUR: Infos latentes AE pour '{original_arcname}'.")
                                        else:
                                            base, _ = os.path.splitext(original_arcname); save_path = os.path.join(output_extract_path, base + "_reconstructed_cifar_ae.png")
                                            if decode_latent_to_image_ae(data_to_process, latent_shape_info, quant_params, original_dims, save_path, log_callback=log_callback): files_written_by_custom_logic.add(save_path)
                                    except Exception as e_ae_proc: log_callback(f"[AIC_HANDLER_DECOMP] Erreur traitement AE de '{original_arcname}': {e_ae_proc}")
                                else: log_callback(f"[AIC_HANDLER_DECOMP] Moteur AE non chargé pour '{original_arcname}'.")
                            elif CLASSIC_COMPRESSORS_LOADED:
                                if cancel_event and cancel_event.is_set(): operation_cancelled_internally = True; break
                                # ... (logique décompression classique comme avant, appelant les fonctions de classic_compressors)
                                params_from_meta = item_meta.get("compression_params",{})
                                if comp_method == METHOD_BZIP2: final_data_to_write = bzip2_decompress(data_to_process, params_from_meta, log_callback)
                                elif comp_method == METHOD_LZMA: final_data_to_write = lzma_decompress(data_to_process, params_from_meta, log_callback)
                                elif comp_method == METHOD_ZSTD: final_data_to_write = zstd_decompress(data_to_process, params_from_meta, log_callback)
                                elif comp_method == METHOD_BROTLI: final_data_to_write = brotli_decompress(data_to_process, params_from_meta, log_callback)
                                elif comp_method == METHOD_DEFLATE: final_data_to_write = deflate_decompress(data_to_process, params_from_meta, log_callback)
                                elif comp_method == METHOD_STORED: final_data_to_write = stored_decompress(data_to_process, params_from_meta, log_callback)
                                else: final_data_to_write = data_to_process 
                                if final_data_to_write is None and comp_method != METHOD_STORED : log_callback(f"[AIC_HANDLER_DECOMP] Échec décompression {comp_method} pour {original_arcname}.")
                            else: final_data_to_write = data_to_process
                            
                            if final_data_to_write is not None and not ae_output_already_saved:
                                if cancel_event and cancel_event.is_set(): operation_cancelled_internally = True; break
                                os.makedirs(os.path.dirname(output_final_path), exist_ok=True)
                                with open(output_final_path, 'wb') as f_out: f_out.write(final_data_to_write); files_written_by_custom_logic.add(output_final_path)
                
                processed_items_count += 1
                if callable(progress_callback) and total_items_from_meta > 0:
                    progress_callback(processed_items_count, total_items_from_meta)
            
            if operation_cancelled_internally:
                log_callback("[AIC_HANDLER_DECOMP] Décompression AIC annulée en cours de traitement des items.")
                return False, "Cancelled"

            # La boucle de fallback est une sécurité, on peut aussi y mettre des checks d'annulation
            log_callback("[AIC_HANDLER_DECOMP] Phase extraction générale des membres ZIP restants (sécurité)...")
            for member_info in zf.infolist():
                if cancel_event and cancel_event.is_set(): operation_cancelled_internally = True; break
                member_name = member_info.filename; target_disk_path_fallback = os.path.join(output_extract_path, member_name)
                if member_name == METADATA_FILENAME or target_disk_path_fallback in files_written_by_custom_logic: continue
                # ... (logique de la boucle de fallback comme avant)
            if operation_cancelled_internally: return False, "Cancelled"

            if callable(progress_callback) and total_items_from_meta > 0 and processed_items_count >= total_items_from_meta and not operation_cancelled_internally:
                 progress_callback(total_items_from_meta, total_items_from_meta) # Assurer 100%
            log_callback(f"[AIC_HANDLER_DECOMP] Fin de la décompression AIC dans '{output_extract_path}'.")
            return True, "Success"
            
    except InterruptedError: # Attraper notre propre InterruptedError
        log_callback("[AIC_HANDLER_DECOMP] Opération de décompression AIC annulée explicitement.")
        return False, "Cancelled"
    except zipfile.BadZipFile: # ... (comme avant)
        if callable(progress_callback): progress_callback(0,1 if total_items_from_meta == 0 else total_items_from_meta)
        return False, "BadZipFile"
    except RuntimeError as e_global_rt: # ... (comme avant)
         if password_decompress and ("password" in str(e_global_rt).lower()): return False, "PasswordErrorArchive"
         if callable(progress_callback): progress_callback(0,1 if total_items_from_meta == 0 else total_items_from_meta)
         return False, "RuntimeErrorArchive"
    except Exception as e: # ... (comme avant)
        if callable(progress_callback): progress_callback(0,1 if total_items_from_meta == 0 else total_items_from_meta)
        return False, f"UnknownError: {e}"

# Dans aicompress/aic_file_handler.py (à ajouter, par exemple après la fonction decompress_aic)

def extract_archive(archive_path, output_dir, password=None, 
                    log_callback=_default_log, progress_callback=None, cancel_event=None):
    log_callback(f"[AIC_HANDLER] Point d'entrée extraction pour '{os.path.basename(archive_path)}'...")

    if cancel_event and cancel_event.is_set(): return False, "CancelledBeforeStart"
    if not os.path.exists(archive_path):
        log_callback(f"[AIC_HANDLER] ERREUR: Archive '{archive_path}' non trouvée."); return False, "FileNotFound"
    try: os.makedirs(output_dir, exist_ok=True)
    except Exception as e: log_callback(f"[AIC_HANDLER] ERREUR création dossier sortie '{output_dir}': {e}"); return False, "OutputDirError"

    _, extension = os.path.splitext(archive_path); extension = extension.lower()
    analysis_for_extract = "unknown_type"
    if AI_ANALYZER_AVAILABLE:
        analysis_for_extract = analyze_file_content(archive_path, log_callback=log_callback) 

    if extension == DEFAULT_AIC_EXTENSION or analysis_for_extract == "aic_custom_format": 
        return decompress_aic(archive_path, output_dir, password, log_callback, progress_callback, cancel_event)

    elif extension == ".zip" or analysis_for_extract == "zip_archive":
        log_callback(f"[AIC_HANDLER] Format .zip détecté. Utilisation de la logique zipfile interne.")
        success, status = False, "InitErrorZIP"; num_files = 0; processed_files = 0; cancelled = False
        try:
            with zipfile.ZipFile(archive_path, 'r') as zf:
                members = zf.infolist(); num_files = len(members)
                if callable(progress_callback) and num_files > 0: progress_callback(0, num_files)
                for member in members:
                    if cancel_event and cancel_event.is_set(): cancelled=True; status="Cancelled"; break
                    log_callback(f"[AIC_HANDLER] Extraction ZIP: {member.filename}")
                    zf.extract(member, path=output_dir, pwd=password.encode('utf-8') if password else None)
                    processed_files += 1
                    if callable(progress_callback): progress_callback(processed_files, num_files)
                if not cancelled and processed_files == num_files: success, status = True, "Success"
        except Exception as e_zip: status = f"Error: {e_zip}"; success = False
        return success, status

    elif extension == ".rar" or analysis_for_extract == "rar_archive":
        log_callback(f"[AIC_HANDLER] Format .rar détecté. Appel de external_handlers.decompress_rar.")
        return external_handlers.decompress_rar(archive_path, output_dir, password, log_callback, progress_callback, cancel_event)

    elif extension == ".7z" or analysis_for_extract == "7z_archive":
        log_callback(f"[AIC_HANDLER] Format .7z détecté. Appel de external_handlers.decompress_7z.")
        return external_handlers.decompress_7z(archive_path, output_dir, password, log_callback, progress_callback, cancel_event)

    else: # Fallback pour .zip sans extension
        if zipfile.is_zipfile(archive_path):
            log_callback(f"[AIC_HANDLER] Type non reconnu mais semble ZIP, tentative d'extraction...")
            # La logique d'extraction ZIP ci-dessus pourrait être factorisée dans une fonction helper
            return extract_archive(archive_path.replace(extension, ".zip"), output_dir, password, log_callback, progress_callback, cancel_event)
        log_callback(f"[AIC_HANDLER] Format archive non supporté: '{os.path.basename(archive_path)}'."); 
        return False, "UnsupportedFormat"