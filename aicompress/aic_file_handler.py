# aicompress/aic_file_handler.py (Version du 18 Mai - Correction FINALE NameError rarfile)

import os
import zipfile 
import shutil
import json
import numpy as np

# --- Fonction de Log par Défaut ---
def _default_log(message):
    print(message)

# --- Initialisation des Flags de Disponibilité Globaux ---
PIL_AVAILABLE = False
KERAS_AVAILABLE = False
AI_ANALYZER_AVAILABLE = False 
RARFILE_AVAILABLE = False
IMPORTED_RARFILE_MODULE = None # Variable globale pour stocker le module rarfile importé
CRYPTOGRAPHY_AVAILABLE = False
ZSTD_AVAILABLE = False 
CLASSIC_COMPRESSORS_LOADED = False 
AE_ENGINE_LOADED = False 
ORCHESTRATOR_LOADED_SUCCESSFULLY = False
PY7ZR_SUPPORT_AVAILABLE = False 

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
ensure_ae_models_loaded, encode_image_with_ae, decode_latent_to_image_ae = _fb_ensure_ae, _fb_encode_ae, _fb_decode_ae
get_compression_settings = _fb_get_comp_settings
METHOD_STORED,METHOD_DEFLATE,METHOD_BZIP2,METHOD_LZMA,METHOD_ZSTD = "STORED","DEFLATE","BZIP2","LZMA","ZSTD"

# --- Imports Conditionnels et Mise à Jour des Flags et Fonctions ---
try: from PIL import Image, ImageOps; PIL_AVAILABLE = True; _default_log("[AIC_HANDLER] Pillow OK.")
except ImportError: _default_log("AVERT (aic_handler.py): Pillow non trouvée.")
try: import tensorflow as tf; KERAS_AVAILABLE = True; _default_log("[AIC_HANDLER] TensorFlow/Keras OK.")
except ImportError: _default_log("AVERT (aic_handler.py): TensorFlow/Keras non trouvé.")

try:
    from .ai_analyzer import get_file_features as _real_gff, analyze_file_content as _real_afc, AI_ANALYZER_AVAILABLE as _ai_flag
    if _ai_flag: get_file_features, analyze_file_content, AI_ANALYZER_AVAILABLE = _real_gff, _real_afc, True; _default_log("[AIC_HANDLER] ai_analyzer OK.")
    else: _default_log("[AIC_HANDLER] ai_analyzer chargé mais non dispo (interne).")
except ImportError as e: _default_log(f"AVERT (aic_handler.py): Échec import ai_analyzer: {e}")

# --- Bloc d'import rarfile CORRIGÉ ---
try: 
    import rarfile as actual_rarfile_module 
    IMPORTED_RARFILE_MODULE = actual_rarfile_module # Utiliser la variable globale définie en haut
    RARFILE_AVAILABLE = True
    _default_log("[AIC_HANDLER] Bibliothèque rarfile chargée et prête.")
except ImportError: 
    _default_log("AVERTISSEMENT (aic_handler.py): rarfile non installée.")
    # IMPORTED_RARFILE_MODULE reste None, RARFILE_AVAILABLE reste False
# --- FIN Bloc d'import rarfile CORRIGÉ ---

try:
    from .crypto_utils import encrypt_data as _re_cu, decrypt_data as _rd_cu, CRYPTOGRAPHY_AVAILABLE as _cf_cu_flag
    if _cf_cu_flag: encrypt_data, decrypt_data, CRYPTOGRAPHY_AVAILABLE = _re_cu, _rd_cu, True; _default_log("[AIC_HANDLER] crypto_utils OK.")
    else: _default_log("AVERT (aic_handler.py): crypto_utils chargé, mais cryptography indisponible.")
except ImportError as e: _default_log(f"AVERT (aic_handler.py): crypto_utils.py non trouvé: {e}")

try:
    from .classic_compressors import (stored_compress as _cs_s, deflate_compress as _cs_d, bzip2_compress as _cs_b, lzma_compress as _cs_l, zstd_compress as _cs_z,
                                      stored_decompress as _cs_sd, deflate_decompress as _cs_dd, bzip2_decompress as _cs_bd, lzma_decompress as _cs_ld, zstd_decompress as _cs_zd,
                                      METHOD_STORED as _M_S_cl, METHOD_DEFLATE as _M_D_cl, METHOD_BZIP2 as _M_B_cl, METHOD_LZMA as _M_L_cl, METHOD_ZSTD as _M_Z_cl,
                                      ZSTD_AVAILABLE as ZSTD_FLAG_cl, CLASSIC_COMPRESSORS_READY as CC_READY_FLAG_cl)
    if CC_READY_FLAG_cl:
        stored_compress=_cs_s; deflate_compress=_cs_d; bzip2_compress=_cs_b; lzma_compress=_cs_l; zstd_compress=_cs_z
        stored_decompress=_cs_sd; deflate_decompress=_cs_dd; bzip2_decompress=_cs_bd; lzma_decompress=_cs_ld; zstd_decompress=_cs_zd
        METHOD_STORED,METHOD_DEFLATE,METHOD_BZIP2,METHOD_LZMA,METHOD_ZSTD=_M_S_cl,_M_D_cl,_M_B_cl,_M_L_cl,_M_Z_cl
        CLASSIC_COMPRESSORS_LOADED = True; ZSTD_AVAILABLE = ZSTD_FLAG_cl; _default_log("[AIC_HANDLER] classic_compressors OK.")
        if not ZSTD_AVAILABLE: _default_log("[AIC_HANDLER] AVERT: Zstd non dispo via classic_compressors.")
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

def _process_and_add_file_to_aic(zf, file_path_on_disk, arcname_to_use, password_compress, log_callback=_default_log):
    # ... (Code complet de cette fonction, identique à la version précédente)
    item_basename = os.path.basename(file_path_on_disk)
    item_meta = { "original_name": arcname_to_use, "status": "processed", "analysis": "N/A", 
                  "size_original_bytes": 0, "compression_method_used": METHOD_STORED, 
                  "compression_params": {}, "encrypted": False, "crypto_salt_hex": None, 
                  "crypto_iv_hex": None, "crypto_tag_hex": None, "original_image_dims": None, 
                  "latent_quant_params": None, "latent_shape_info": None }
    try: item_meta["size_original_bytes"] = os.path.getsize(file_path_on_disk)
    except OSError as e_size: log_callback(f"[AIC_HANDLER] ERREUR: Taille de {file_path_on_disk}: {e_size}"); item_meta["status"]="read_error_size"; return item_meta
    if AI_ANALYZER_AVAILABLE: item_meta["analysis"] = analyze_file_content(file_path_on_disk, log_callback=log_callback)
    comp_method_name_from_ia, comp_params_dict_from_ia = get_compression_settings(file_path_on_disk, item_meta["analysis"], log_callback=log_callback)
    item_meta["compression_method_used"] = comp_method_name_from_ia; item_meta["compression_params"] = comp_params_dict_from_ia
    final_arcname_for_zip_member = arcname_to_use; data_to_process_further = b""
    if comp_method_name_from_ia == "MOTEUR_AE_CIFAR10_COLOR":
        if AE_ENGINE_LOADED:
            ae_encoding_result = encode_image_with_ae(file_path_on_disk, log_callback=log_callback)
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
        except Exception as e_read: log_callback(f"[AIC_HANDLER] ERREUR lecture fichier {item_basename}: {e_read}"); item_meta["status"]="read_error"; return item_meta
    data_after_internal_comp = data_to_process_further
    if CLASSIC_COMPRESSORS_LOADED:
        if comp_method_name_from_ia == METHOD_BZIP2: data_after_internal_comp = bzip2_compress(data_to_process_further, comp_params_dict_from_ia, log_callback)
        elif comp_method_name_from_ia == METHOD_LZMA: data_after_internal_comp = lzma_compress(data_to_process_further, comp_params_dict_from_ia, log_callback)
        elif comp_method_name_from_ia == METHOD_ZSTD: data_after_internal_comp = zstd_compress(data_to_process_further, comp_params_dict_from_ia, log_callback)
        elif comp_method_name_from_ia == METHOD_DEFLATE: data_after_internal_comp = deflate_compress(data_to_process_further, comp_params_dict_from_ia, log_callback)
        elif comp_method_name_from_ia == METHOD_STORED: data_after_internal_comp = stored_compress(data_to_process_further, comp_params_dict_from_ia, log_callback)
        if data_after_internal_comp is None : 
            log_callback(f"[AIC_HANDLER] Échec {comp_method_name_from_ia}. Utilisation données originales."); data_after_internal_comp = data_to_process_further
            if comp_method_name_from_ia != "MOTEUR_AE_CIFAR10_COLOR": item_meta["compression_method_used"]=METHOD_STORED; item_meta["compression_params"]={}; comp_method_name_from_ia=METHOD_STORED
    elif comp_method_name_from_ia not in ["MOTEUR_AE_CIFAR10_COLOR", METHOD_STORED]:
        log_callback(f"[AIC_HANDLER] AVERT: classic_compressors.py non chargé, {comp_method_name_from_ia} non applicable. Fallback STORED.")
        item_meta["compression_method_used"]=METHOD_STORED; item_meta["compression_params"]={}; comp_method_name_from_ia=METHOD_STORED
    data_to_write_to_zip = data_after_internal_comp
    if password_compress and data_to_write_to_zip and CRYPTOGRAPHY_AVAILABLE:
        encrypted_bundle = encrypt_data(data_to_write_to_zip, password_compress, log_callback=log_callback)
        if encrypted_bundle and encrypted_bundle[0] is not None and encrypted_bundle[1] is not None:
            data_to_write_to_zip,salt,iv,tag=encrypted_bundle; item_meta["encrypted"]=True; item_meta["crypto_salt_hex"]=salt.hex(); item_meta["crypto_iv_hex"]=iv.hex(); item_meta["crypto_tag_hex"]=tag.hex()
            final_arcname_for_zip_member += ENCRYPTED_FILE_EXTENSION
        else: log_callback(f"[AIC_HANDLER] AVERT: Échec chiffrement {item_basename}.")
    if data_to_write_to_zip is not None: zf.writestr(final_arcname_for_zip_member, data_to_write_to_zip, compress_type=zipfile.ZIP_STORED); log_callback(f"[AIC_HANDLER] Données '{item_basename}' écrites: '{final_arcname_for_zip_member}'.")
    else: log_callback(f"[AIC_HANDLER] AVERT: Pas de données à écrire pour {item_basename}.")
    return item_meta

def compress_to_aic(input_paths, output_aic_path, password_compress=None, log_callback=_default_log):
    # ... (Identique à la version précédente)
    log_callback(f"[AIC_HANDLER] Compression AIC vers '{output_aic_path}' (Chiffré: {'Oui' if password_compress else 'Non'})...")
    all_items_metadata = []
    try:
        with zipfile.ZipFile(output_aic_path, 'w') as zf: 
            for item_path_on_disk in input_paths:
                item_basename_for_archive = os.path.basename(item_path_on_disk)
                if not os.path.exists(item_path_on_disk): all_items_metadata.append({"original_name":item_basename_for_archive,"status":"not_found"}); continue
                if os.path.isfile(item_path_on_disk): all_items_metadata.append(_process_and_add_file_to_aic(zf,item_path_on_disk,item_basename_for_archive,password_compress,log_callback))
                elif os.path.isdir(item_path_on_disk):
                    all_items_metadata.append({"original_name":item_basename_for_archive,"type_in_archive":"directory","status":"processed","size_original_bytes":0}) 
                    for root,_,files_in_dir in os.walk(item_path_on_disk):
                        for file_in_d in files_in_dir:
                            full_path=os.path.join(root,file_in_d); arc_path=os.path.join(item_basename_for_archive,os.path.relpath(full_path,item_path_on_disk))
                            all_items_metadata.append(_process_and_add_file_to_aic(zf,full_path,arc_path,password_compress,log_callback))
            metadata_final_content = {"aicompress_version":"1.1-refactor-final","items_details":all_items_metadata,"global_encryption_hint":bool(password_compress and CRYPTOGRAPHY_AVAILABLE)}
            zf.writestr(METADATA_FILENAME,json.dumps(metadata_final_content,indent=4))
            log_callback(f"[AIC_HANDLER] Métadonnées écrites. Compression AIC terminée: '{output_aic_path}'"); return True,"Success"
    except Exception as e: log_callback(f"[AIC_HANDLER] ERREUR MAJEURE compression: {e}"); import traceback; log_callback(f"[AIC_HANDLER] Traceback: {traceback.format_exc()}"); return False,f"Error: {e}"

def decompress_aic(aic_file_path, output_extract_path, password_decompress=None, log_callback=_default_log):
    # ... (Identique à la version précédente, qui appelle les fonctions de classic_compressors et ae_engine)
    log_callback(f"[AIC_HANDLER_DECOMP] >>> Début Décompression AIC: '{aic_file_path}' vers '{output_extract_path}' (Mdp: {'Oui' if password_decompress else 'Non'})")    
    if not os.path.exists(aic_file_path): log_callback(f"[AIC_HANDLER_DECOMP] ERREUR: Archive '{aic_file_path}' non trouvée."); return False, "FileNotFound"
    try:
        os.makedirs(output_extract_path, exist_ok=True); log_callback(f"[AIC_HANDLER_DECOMP] Dossier sortie: {output_extract_path}")
        with zipfile.ZipFile(aic_file_path, 'r') as zf:
            log_callback(f"[AIC_HANDLER_DECOMP] Archive ZIP ouverte. Namelist: {zf.namelist()}")
            metadata=None; metadata_loaded_successfully=False
            try: metadata_str = zf.read(METADATA_FILENAME); metadata = json.loads(metadata_str); log_callback("[AIC_HANDLER_DECOMP] Métadonnées lues."); metadata_loaded_successfully = True
            except Exception as e_meta: 
                log_callback(f"[AIC_HANDLER_DECOMP] ERREUR métadonnées: {e_meta}. Tentative ZIP standard...");
                try: zf.extractall(path=output_extract_path,pwd=password_decompress.encode('utf-8') if password_decompress else None); return True, "SuccessZipFallback"
                except RuntimeError as e_zip_fb:
                    if password_decompress and "password" in str(e_zip_fb).lower(): return False, "PasswordErrorZipFallback"
                    return False, f"ZipFallbackError: {e_zip_fb}"
                except Exception as e_gen_fb: return False, f"ZipFallbackGenericError: {e_gen_fb}"
            if not (metadata_loaded_successfully and metadata and "items_details" in metadata):
                log_callback("[AIC_HANDLER_DECOMP] AVERT: Métadonnées invalides/absentes. Fin anormale."); return False, "InvalidMetadata" 
            log_callback(f"[AIC_HANDLER_DECOMP] Traitement {len(metadata.get('items_details', []))} items des métadonnées...")
            files_written_by_custom_logic = set() 
            for item_meta in metadata["items_details"]:
                original_arcname = item_meta.get("original_name"); 
                if not original_arcname: log_callback("[AIC_HANDLER_DECOMP] AVERT: item_meta sans original_name."); continue
                output_final_path = os.path.join(output_extract_path, original_arcname)
                log_callback(f"[AIC_HANDLER_DECOMP] Prépa item méta: '{original_arcname}' -> '{output_final_path}'")
                if item_meta.get("status")=="not_found": log_callback(f" Ignoré (not_found): {original_arcname}"); continue
                if item_meta.get("type_in_archive")=="directory": log_callback(f" Création dossier: {output_final_path}"); os.makedirs(output_final_path,exist_ok=True); files_written_by_custom_logic.add(output_final_path); continue
                comp_method = item_meta.get("compression_method_used"); is_encrypted = item_meta.get("encrypted", False)
                member_name_in_zip = original_arcname
                if comp_method == "MOTEUR_AE_CIFAR10_COLOR": member_name_in_zip += LATENT_FILE_EXTENSION
                if is_encrypted: member_name_in_zip += ENCRYPTED_FILE_EXTENSION
                data_from_zip = None; 
                try: data_from_zip = zf.read(member_name_in_zip) 
                except KeyError: 
                    if not (comp_method == "MOTEUR_AE_CIFAR10_COLOR" or is_encrypted) and original_arcname in zf.namelist():
                        try: data_from_zip = zf.read(original_arcname); 
                        except Exception as e_rf: log_callback(f"[AIC_HANDLER_DECOMP] Échec lecture fallback '{original_arcname}': {e_rf}"); continue
                    else: log_callback(f"[AIC_HANDLER_DECOMP] ERREUR: Membre '{member_name_in_zip}' non trouvé."); continue
                except RuntimeError as e_rt_m: 
                    if password_decompress and "password" in str(e_rt_m).lower(): return False, f"PasswordErrorFile:{member_name_in_zip}"
                    log_callback(f"[AIC_HANDLER_DECOMP] Erreur Runtime ZIP lecture '{member_name_in_zip}': {e_rt_m}"); continue
                if data_from_zip is None: log_callback(f"[AIC_HANDLER_DECOMP] AVERT: Pas de données lues pour {member_name_in_zip}."); continue
                data_to_process = data_from_zip
                if is_encrypted:
                    if not CRYPTOGRAPHY_AVAILABLE: log_callback("[AIC_HANDLER_DECOMP] ERREUR: Crypto non dispo."); continue
                    if not password_decompress: return False, f"PasswordNeededForDecryption:{original_arcname}"
                    salt=bytes.fromhex(item_meta["crypto_salt_hex"]); iv=bytes.fromhex(item_meta["crypto_iv_hex"]); tag=bytes.fromhex(item_meta["crypto_tag_hex"])
                    data_to_process = decrypt_data(data_from_zip, password_decompress, salt, iv, tag, log_callback=log_callback)
                    if data_to_process is None: return False, f"PasswordErrorDecryption:{original_arcname}"
                final_data_to_write = None; ae_output_already_saved = False
                if comp_method == "MOTEUR_AE_CIFAR10_COLOR":
                    ae_output_already_saved = True 
                    if AE_ENGINE_LOADED and ensure_ae_models_loaded(log_callback=log_callback):
                        try:
                            latent_shape_info=item_meta.get("latent_shape_info"); quant_params=item_meta.get("latent_quant_params"); original_dims=item_meta.get("original_image_dims")
                            if not (latent_shape_info and quant_params): log_callback(f"[AIC_HANDLER_DECOMP] ERREUR: Infos latentes AE pour '{original_arcname}'."); continue
                            base, _ = os.path.splitext(original_arcname); save_path = os.path.join(output_extract_path, base + "_reconstructed_cifar_ae.png")
                            if decode_latent_to_image_ae(data_to_process, latent_shape_info, quant_params, original_dims, save_path, log_callback=log_callback): files_written_by_custom_logic.add(save_path)
                        except Exception as e_ae_proc: log_callback(f"[AIC_HANDLER_DECOMP] Erreur traitement AE de '{original_arcname}': {e_ae_proc}")
                    else: log_callback(f"[AIC_HANDLER_DECOMP] Moteur AE non chargé pour '{original_arcname}'."); continue
                elif CLASSIC_COMPRESSORS_LOADED:
                    comp_params_from_meta = item_meta.get("compression_params", {})
                    if comp_method == METHOD_BZIP2: final_data_to_write = bzip2_decompress(data_to_process, comp_params_from_meta, log_callback)
                    elif comp_method == METHOD_LZMA: final_data_to_write = lzma_decompress(data_to_process, comp_params_from_meta, log_callback)
                    elif comp_method == METHOD_ZSTD: final_data_to_write = zstd_decompress(data_to_process, comp_params_from_meta, log_callback)
                    elif comp_method == METHOD_DEFLATE: final_data_to_write = deflate_decompress(data_to_process, comp_params_from_meta, log_callback)
                    elif comp_method == METHOD_STORED: final_data_to_write = stored_decompress(data_to_process, comp_params_from_meta, log_callback)
                    else: final_data_to_write = data_to_process 
                    if final_data_to_write is None and comp_method != METHOD_STORED : log_callback(f"[AIC_HANDLER_DECOMP] Échec décompression {comp_method} pour {original_arcname}."); continue
                else: final_data_to_write = data_to_process
                if final_data_to_write is not None and not ae_output_already_saved:
                    os.makedirs(os.path.dirname(output_final_path), exist_ok=True)
                    with open(output_final_path, 'wb') as f_out: f_out.write(final_data_to_write); files_written_by_custom_logic.add(output_final_path)
            log_callback("[AIC_HANDLER_DECOMP] Phase extraction générale des membres ZIP restants...")
            for member_info in zf.infolist():
                member_name = member_info.filename; target_disk_path_fallback = os.path.join(output_extract_path, member_name)
                if member_name == METADATA_FILENAME or target_disk_path_fallback in files_written_by_custom_logic: continue
                is_intermediate = False
                if member_name.endswith(LATENT_FILE_EXTENSION) or member_name.endswith(LATENT_FILE_EXTENSION+ENCRYPTED_FILE_EXTENSION) or \
                   (member_name.endswith(ENCRYPTED_FILE_EXTENSION) and not member_name[:-len(ENCRYPTED_FILE_EXTENSION)].endswith(LATENT_FILE_EXTENSION)):
                    orig_cand = member_name.replace(LATENT_FILE_EXTENSION,"").replace(ENCRYPTED_FILE_EXTENSION,"")
                    if os.path.join(output_extract_path,orig_cand) in files_written_by_custom_logic or \
                       os.path.join(output_extract_path, os.path.splitext(orig_cand)[0]+"_reconstructed_cifar_ae.png") in files_written_by_custom_logic: is_intermediate=True
                if is_intermediate: log_callback(f"[AIC_HANDLER_DECOMP_FALLBACK] Ignoré (donnée source déjà traitée): {member_name}"); continue
                try: zf.extract(member_info, path=output_extract_path, pwd=password_decompress.encode('utf-8') if password_decompress else None); log_callback(f"[AIC_HANDLER_DECOMP_FALLBACK] Extrait: {member_name}")
                except RuntimeError as e_rt_extract_all:
                     if password_decompress and "password" in str(e_rt_extract_all).lower(): log_callback(f"[AIC_HANDLER_DECOMP_FALLBACK] Mdp incorrect ZIP membre '{member_name}'.")
                     else: log_callback(f"[AIC_HANDLER_DECOMP_FALLBACK] Erreur runtime extraction '{member_name}': {e_rt_extract_all}")
                except Exception as e_extract_std_member: log_callback(f"[AIC_HANDLER_DECOMP_FALLBACK] Erreur extraction '{member_name}': {e_extract_std_member}")
            log_callback(f"[AIC_HANDLER_DECOMP] Fin de la décompression AIC dans '{output_extract_path}'."); return True, "Success"
    except zipfile.BadZipFile: log_callback(f"[AIC_HANDLER_DECOMP] ERREUR: Not a ZIP: {aic_file_path}"); return False, "BadZipFile"
    except RuntimeError as e_global_rt:
         if password_decompress and ("password" in str(e_global_rt).lower()): return False, "PasswordErrorArchive"
         log_callback(f"[AIC_HANDLER_DECOMP] ERREUR runtime AIC: {e_global_rt}"); return False, "RuntimeErrorArchive"
    except Exception as e: log_callback(f"[AIC_HANDLER_DECOMP] ERREUR majeure décompression AIC: {e}"); import traceback; log_callback(f"[AIC_HANDLER_DECOMP] Traceback: {traceback.format_exc()}"); return False, f"UnknownError: {e}"

# --- Initialisation de external_handlers (doit être à la fin) ---
try:
    from . import external_handlers # Assurez-vous que external_handlers.py est dans le même dossier (aicompress)
    dependencies_for_ext = {
        "log_callback": _default_log,
        "decompress_aic_func": decompress_aic, 
        "analyze_file_content_func": analyze_file_content, 
        "AI_ANALYZER_AVAILABLE_flag": AI_ANALYZER_AVAILABLE,
        "RARFILE_AVAILABLE_flag": RARFILE_AVAILABLE,
        "rarfile_module": IMPORTED_RARFILE_MODULE, # Utiliser la variable globale corrigée
        "DEFAULT_AIC_EXTENSION_const": DEFAULT_AIC_EXTENSION
    }
    external_handlers._initialize_dependencies(dependencies_for_ext)
    _default_log("[AIC_HANDLER] Dépendances pour external_handlers initialisées.")
except ImportError as e_init_ext_handlers:
    _default_log(f"[AIC_HANDLER] AVERT: external_handlers.py non trouvé pour init: {e_init_ext_handlers}")
except AttributeError as e_attr_ext_handlers: # Si _initialize_dependencies n'existe pas
     _default_log(f"[AIC_HANDLER] AVERT: _initialize_dependencies non trouvé dans external_handlers: {e_attr_ext_handlers}")
except Exception as e_init_ext_other:
    _default_log(f"[AIC_HANDLER] AVERT: Erreur init external_handlers: {e_init_ext_other}")