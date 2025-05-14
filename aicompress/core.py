# aicompress/core.py

import os
import zipfile
import shutil
import json
import numpy as np
import bz2 
import lzma 
import zlib

# --- Fonction de Log par Défaut ---
def _default_log(message):
    print(message)

# --- Imports Conditionnels pour les Fonctionnalités ---
try: from PIL import Image, ImageOps; PIL_AVAILABLE = True
except ImportError: PIL_AVAILABLE = False; _default_log("AVERTISSEMENT (core.py): Pillow non trouvée.")

try: import tensorflow as tf; KERAS_AVAILABLE = True
except ImportError: KERAS_AVAILABLE = False; _default_log("AVERTISSEMENT (core.py): TensorFlow/Keras non trouvé.")

AI_ANALYZER_AVAILABLE_FLAG_FROM_MODULE = False 
def _fallback_analyze_file_content(file_path, log_callback=_default_log):
    log_callback(f"[CORE_FALLBACK] Utilisation de analyze_file_content factice.")
    return "analyzer_unavailable"
analyze_file_content = _fallback_analyze_file_content
def _fallback_get_file_features(file_path, log_callback=_default_log):
    log_callback(f"[CORE_FALLBACK] Utilisation de get_file_features factice.")
    return {"type": "analyzer_unavailable", 
            "size_bytes": os.path.getsize(file_path) if os.path.exists(file_path) else 0, 
            "entropy_normalized": 0.0, "error": True}
get_file_features = _fallback_get_file_features
try:
    from .ai_analyzer import (get_file_features as _real_get_file_features,
                              analyze_file_content as _real_analyze_file_content, 
                              AI_ANALYZER_AVAILABLE as _analyzer_status_from_module)
    get_file_features = _real_get_file_features
    analyze_file_content = _real_analyze_file_content
    AI_ANALYZER_AVAILABLE_FLAG_FROM_MODULE = _analyzer_status_from_module 
    if AI_ANALYZER_AVAILABLE_FLAG_FROM_MODULE: _default_log("[CORE] AI Analyzer importé et disponible.")
    else: _default_log("[CORE] AI Analyzer importé mais marqué non disponible par ai_analyzer.py.")
except ImportError as e_imp_analyzer: _default_log(f"AVERTISSEMENT (core.py): Échec import ai_analyzer. Erreur: {e_imp_analyzer}")
except Exception as e_other_analyzer: _default_log(f"AVERTISSEMENT (core.py): Erreur import ai_analyzer. Erreur: {e_other_analyzer}")
AI_ANALYZER_AVAILABLE = AI_ANALYZER_AVAILABLE_FLAG_FROM_MODULE

try: import rarfile; RARFILE_AVAILABLE = True
except ImportError: RARFILE_AVAILABLE = False; _default_log("AVERTISSEMENT (core.py): rarfile non installée.")

try:
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.primitives import hashes; from cryptography.hazmat.backends import default_backend
    CRYPTOGRAPHY_AVAILABLE = True
except ImportError: CRYPTOGRAPHY_AVAILABLE = False; _default_log("AVERTISSEMENT (core.py): 'cryptography' non trouvée.")

try: # NOUVEL IMPORT ZSTANDARD
    import zstandard as zstd 
    ZSTD_AVAILABLE = True
    _default_log("[CORE] Bibliothèque Zstandard (zstd) chargée.")
except ImportError:
    ZSTD_AVAILABLE = False
    _default_log("AVERTISSEMENT (core.py): Bibliothèque 'zstandard' non trouvée. Compression Zstd désactivée.")


# ---- Constantes et Variables Globales ----
METADATA_FILENAME = "aicompress_metadata.json"; DEFAULT_AIC_EXTENSION = ".aic"
LATENT_FILE_EXTENSION = ".aic_latent"; ENCRYPTED_FILE_EXTENSION = ".aic_enc"
CIFAR10_COLOR_ENCODER_PATH = "encoder_cifar10_color_v5.keras"
CIFAR10_COLOR_DECODER_PATH = "decoder_cifar10_color_v5.keras"
cifar10_color_encoder_loaded = None; cifar10_color_decoder_loaded = None
cifar10_models_loaded_successfully = False
AES_KEY_SIZE=32; PBKDF2_ITERATIONS=100_000; SALT_SIZE=16; IV_SIZE=12; TAG_SIZE=16

ORCHESTRATOR_MODEL_PATH = os.path.join(os.path.dirname(__file__), "compression_orchestrator_model.joblib")
ORCHESTRATOR_LABEL_ENCODER_PATH = os.path.join(os.path.dirname(__file__), "orchestrator_label_encoder.joblib")
orchestrator_pipeline = None; orchestrator_label_encoder = None
orchestrator_loaded_successfully = False

def load_orchestrator_model(log_callback=_default_log):
    global orchestrator_pipeline, orchestrator_label_encoder, orchestrator_loaded_successfully
    if orchestrator_loaded_successfully: return True
    if not (os.path.exists(ORCHESTRATOR_MODEL_PATH) and os.path.exists(ORCHESTRATOR_LABEL_ENCODER_PATH)):
        log_callback(f"[CORE_ORCH] ERREUR: Modèle Orchestrateur ou LE non trouvé."); orchestrator_loaded_successfully = False; return False
    try:
        import joblib 
        log_callback(f"[CORE_ORCH] Chargement Pipeline: {ORCHESTRATOR_MODEL_PATH}")
        orchestrator_pipeline = joblib.load(ORCHESTRATOR_MODEL_PATH)
        log_callback(f"[CORE_ORCH] Chargement Label Encoder: {ORCHESTRATOR_LABEL_ENCODER_PATH}")
        orchestrator_label_encoder = joblib.load(ORCHESTRATOR_LABEL_ENCODER_PATH)
        log_callback("[CORE_ORCH] Modèle Orchestrateur et LE chargés."); orchestrator_loaded_successfully = True; return True
    except Exception as e:
        log_callback(f"[CORE_ORCH] ERREUR chargement modèle Orchestrateur: {e}"); import traceback
        log_callback(f"[CORE_ORCH] Traceback: {traceback.format_exc()}"); orchestrator_loaded_successfully = False; return False

if load_orchestrator_model_flag_unused := load_orchestrator_model(_default_log): pass

def _derive_key(password: str, salt: bytes) -> bytes | None:
    if not password: return None
    kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=AES_KEY_SIZE, salt=salt, iterations=PBKDF2_ITERATIONS, backend=default_backend())
    return kdf.derive(password.encode('utf-8'))

def encrypt_data(data: bytes, password: str, log_callback=_default_log) -> tuple | None:
    if not CRYPTOGRAPHY_AVAILABLE: log_callback("[CORE_CRYPTO] Crypto non dispo."); return None
    if not password: return data, None, None, None 
    salt = os.urandom(SALT_SIZE); key = _derive_key(password, salt); iv = os.urandom(IV_SIZE)
    cipher = Cipher(algorithms.AES(key), modes.GCM(iv), backend=default_backend())
    encryptor = cipher.encryptor(); encrypted_data = encryptor.update(data) + encryptor.finalize(); tag = encryptor.tag
    log_callback("[CORE_CRYPTO] Données chiffrées AES-GCM."); return encrypted_data, salt, iv, tag

def decrypt_data(encrypted_data: bytes, password: str, salt: bytes, iv: bytes, tag: bytes, log_callback=_default_log) -> bytes | None:
    if not CRYPTOGRAPHY_AVAILABLE: log_callback("[CORE_CRYPTO] Crypto non dispo."); return None
    if not password: log_callback("[CORE_CRYPTO] Mdp requis pour déchiffrer."); return None
    key = _derive_key(password, salt)
    try:
        cipher = Cipher(algorithms.AES(key), modes.GCM(iv, tag), backend=default_backend())
        decryptor = cipher.decryptor(); decrypted_data = decryptor.update(encrypted_data) + decryptor.finalize()
        log_callback("[CORE_CRYPTO] Données déchiffrées."); return decrypted_data
    except Exception as e: log_callback(f"[CORE_CRYPTO] ÉCHEC DÉCHIFFREMENT: {e}."); return None

def _load_keras_model_pair(encoder_path, decoder_path, model_name_log, log_callback=_default_log):
    if not KERAS_AVAILABLE: log_callback(f"[CORE] Keras/TF non dispo pour {model_name_log}."); return None, None, False
    if not (os.path.exists(encoder_path) and os.path.exists(decoder_path)): log_callback(f"[CORE] ERREUR: Modèles {model_name_log} non trouvés."); return None, None, False
    try:
        log_callback(f"[CORE] Chargement encodeur {model_name_log}: {encoder_path}"); encoder = tf.keras.models.load_model(encoder_path, compile=False)
        log_callback(f"[CORE] Chargement décodeur {model_name_log}: {decoder_path}"); decoder = tf.keras.models.load_model(decoder_path, compile=False)
        log_callback(f"[CORE] Modèles {model_name_log} chargés."); return encoder, decoder, True
    except Exception as e: log_callback(f"[CORE] ERREUR chargement {model_name_log}: {e}"); import traceback; log_callback(f"[CORE] Traceback: {traceback.format_exc()}"); return None, None, False

def ensure_cifar10_color_models_loaded(log_callback=_default_log):
    global cifar10_color_encoder_loaded, cifar10_color_decoder_loaded, cifar10_models_loaded_successfully
    if not cifar10_models_loaded_successfully:
        enc, dec, success = _load_keras_model_pair(CIFAR10_COLOR_ENCODER_PATH, CIFAR10_COLOR_DECODER_PATH, "CIFAR10 Couleur V5", log_callback)
        if success: cifar10_color_encoder_loaded, cifar10_color_decoder_loaded = enc, dec
        cifar10_models_loaded_successfully = success
    return cifar10_models_loaded_successfully

def preprocess_image_for_cifar10_ae(image_path, target_size=(32,32), log_callback=_default_log):
    if not PIL_AVAILABLE: log_callback("[CORE] Pillow non dispo"); return None, None
    try:
        img = Image.open(image_path); original_dims = img.size; img_format = img.format
        if img.mode != 'RGB': img = img.convert('RGB'); log_callback(f"[CORE] Image {image_path} convertie en RGB.")
        img_resized = img.resize(target_size, Image.Resampling.LANCZOS)
        img_array = np.array(img_resized).astype('float32') / 255.0
        img_array_reshaped = np.reshape(img_array, (1, target_size[1], target_size[0], 3))
        log_callback(f"[CORE] Image {image_path} (format:{img_format}, dims:{original_dims}) prétraitée pour AE (taille:{target_size})."); return img_array_reshaped, original_dims
    except Exception as e: log_callback(f"[CORE] Erreur prétraitement {image_path} pour CIFAR10 AE: {e}"); return None, None

def postprocess_and_save_cifar10_ae_output(image_array_norm, output_path, original_target_dims, log_callback=_default_log):
    if not PIL_AVAILABLE: log_callback("[CORE] Pillow non dispo"); return False
    try:
        img_data = image_array_norm.reshape(32, 32, 3); img_data = np.clip(img_data * 255, 0, 255).astype(np.uint8)
        img_to_save = Image.fromarray(img_data, mode='RGB')
        if original_target_dims and (original_target_dims[0] != 32 or original_target_dims[1] != 32):
            log_callback(f"[CORE] Redimensionnement image reconstruite CIFAR10 AE vers {original_target_dims}...")
            img_to_save = img_to_save.resize(original_target_dims, Image.Resampling.LANCZOS)
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir): os.makedirs(output_dir, exist_ok=True)
        img_to_save.save(output_path); log_callback(f"[CORE] Image reconstruite par CIFAR10 AE sauvegardée: {output_path}"); return True
    except Exception as e: log_callback(f"[CORE] Erreur post-traitement/sauvegarde image CIFAR10 AE {output_path}: {e}"); return False

def get_compression_settings(file_path, analysis_result_str_ignored, log_callback=_default_log): 
    global orchestrator_pipeline, orchestrator_label_encoder, orchestrator_loaded_successfully
    global cifar10_models_loaded_successfully 

    log_callback(f"[CORE] Début get_compression_settings pour: '{os.path.basename(file_path)}'")

    if not orchestrator_loaded_successfully:
        if not load_orchestrator_model(log_callback=log_callback): 
            log_callback("[CORE] ERREUR CRITIQUE: Modèle orchestrateur non chargeable. DEFLATE L6 par défaut.")
            return zipfile.ZIP_DEFLATED, 6 

    if not AI_ANALYZER_AVAILABLE: 
        log_callback("[CORE] AI Analyzer non disponible pour extraction de features. DEFLATE L6 par défaut.")
        return zipfile.ZIP_DEFLATED, 6
        
    features = get_file_features(file_path, log_callback=log_callback)
    if features.get("error"):
        log_callback(f"[CORE] Erreur extraction features pour {file_path}. DEFLATE L6 par défaut.")
        return zipfile.ZIP_DEFLATED, 6

    try:
        import pandas as pd 
        input_df = pd.DataFrame([{
            "file_type_analysis": features["type"], 
            "original_size_bytes": features["size_bytes"],
            "entropy_normalized": features["entropy_normalized"],
            "quick_comp_ratio": features["quick_comp_ratio"]
        }])
        log_callback(f"[CORE_ORCH] DataFrame d'entrée pour prédiction: {input_df.to_dict(orient='records')}")
    except ImportError: log_callback("[CORE] Pandas non trouvé. Fallback DEFLATE L6."); return zipfile.ZIP_DEFLATED, 6
    except Exception as e_df: log_callback(f"[CORE] Erreur création DataFrame: {e_df}. Fallback DEFLATE L6."); return zipfile.ZIP_DEFLATED, 6

    try:
        predicted_method_encoded = orchestrator_pipeline.predict(input_df)
        predicted_method_name = orchestrator_label_encoder.inverse_transform(predicted_method_encoded)[0]
        log_callback(f"[CORE_ORCH] Méthode prédite par IA pour '{os.path.basename(file_path)}': {predicted_method_name}")
    except Exception as e_predict: log_callback(f"[CORE_ORCH] Erreur prédiction: {e_predict}. Fallback DEFLATE L6."); return zipfile.ZIP_DEFLATED, 6

    # Interpréter la méthode prédite
    if predicted_method_name == "STORED": return zipfile.ZIP_STORED, None
    elif predicted_method_name.startswith("DEFLATE_L"):
        try: level = int(predicted_method_name.split("DEFLATE_L")[1]); return zipfile.ZIP_DEFLATED, level
        except: log_callback(f"Erreur parsing DEFLATE: {predicted_method_name}"); return zipfile.ZIP_DEFLATED, 6
    elif predicted_method_name.startswith("BZIP2_L"):
        try: level = int(predicted_method_name.split("BZIP2_L")[1]); return "BZIP2", level 
        except: log_callback(f"Erreur parsing BZIP2: {predicted_method_name}"); return "BZIP2", 9 
    elif predicted_method_name.startswith("LZMA_P"):
        try: preset = int(predicted_method_name.split("LZMA_P")[1]); return "LZMA", preset
        except: log_callback(f"Erreur parsing LZMA: {predicted_method_name}"); return "LZMA", 6 
    
    # --- CAS POUR ZSTANDARD (si prédit par l'IA) ---
    elif predicted_method_name.startswith("ZSTD_L"):
        if not ZSTD_AVAILABLE: 
            log_callback(f"[CORE] Zstd prédit mais lib non dispo. Fallback DEFLATE L6.")
            return zipfile.ZIP_DEFLATED, 6
        try: 
            level = int(predicted_method_name.split("ZSTD_L")[1])
            return "ZSTD", level 
        except: 
            log_callback(f"Erreur parsing niveau ZSTD: {predicted_method_name}. Fallback ZSTD L3.")
            return "ZSTD", 3 
    # --- FIN CAS ZSTANDARD ---

    elif predicted_method_name == "MOTEUR_AE_CIFAR10_COLOR":
        if not cifar10_models_loaded_successfully: ensure_cifar10_color_models_loaded(log_callback=log_callback)
        # Vérifier le type réel du fichier ici aussi, car l'IA orchestrateur se base sur les features générales
        analysis_for_ae_check = features["type"] # Utiliser le type déjà analysé
        image_types_for_cifar_ae = ["jpeg_image", "png_image", "bmp_image", "tiff_image", "webp_image"]
        if analysis_for_ae_check not in image_types_for_cifar_ae:
            log_callback(f"[CORE] IA prédit AE, mais type '{analysis_for_ae_check}' non image pour AE. Fallback DEFLATE L1.")
            return zipfile.ZIP_DEFLATED, 1

        if PIL_AVAILABLE and KERAS_AVAILABLE and cifar10_models_loaded_successfully:
            try: 
                with Image.open(file_path) as img: width, height = img.size
                MAX_AE_INPUT_DIM = 256 
                if width <= MAX_AE_INPUT_DIM and height <= MAX_AE_INPUT_DIM: return "MOTEUR_AE_CIFAR10_COLOR", None
                else: log_callback(f"[CORE] IA prédit AE, mais image trop grande. Fallback DEFLATE L1."); return zipfile.ZIP_DEFLATED, 1
            except: log_callback(f"[CORE] IA prédit AE, mais erreur lecture image. Fallback DEFLATE L1."); return zipfile.ZIP_DEFLATED, 1
        else: log_callback(f"[CORE] IA prédit AE, mais dépendances non dispo. Fallback DEFLATE L1."); return zipfile.ZIP_DEFLATED, 1
    else: log_callback(f"[CORE] Méthode prédite '{predicted_method_name}' non gérée. Fallback DEFLATE L6."); return zipfile.ZIP_DEFLATED, 6

# --- Fonction Helper _process_and_add_file_to_aic ---
def _process_and_add_file_to_aic(zf, file_path_on_disk, arcname_to_use, password_compress, log_callback=_default_log):
    item_basename = os.path.basename(file_path_on_disk)
    item_meta = { "original_name": arcname_to_use, "status": "processed", "analysis": "N/A", 
                  "size_original_bytes": os.path.getsize(file_path_on_disk), "compression_method_used": "N/A", 
                  "compression_params": None, "encrypted": False, "crypto_salt_hex": None, 
                  "crypto_iv_hex": None, "crypto_tag_hex": None, "original_image_dims": None, 
                  "latent_quant_params": None, "latent_shape_info": None }

    if AI_ANALYZER_AVAILABLE: 
        item_meta["analysis"] = analyze_file_content(file_path_on_disk, log_callback=log_callback)
    
    comp_method_chosen, comp_params_chosen = get_compression_settings(file_path_on_disk, item_meta["analysis"], log_callback=log_callback)
    item_meta["compression_method_used"] = str(comp_method_chosen)
    item_meta["compression_params"] = comp_params_chosen

    final_arcname_for_zip_member = arcname_to_use
    data_to_process_further = b""
    
    if comp_method_chosen == "MOTEUR_AE_CIFAR10_COLOR":
        image_preprocessed, original_dims = preprocess_image_for_cifar10_ae(file_path_on_disk, log_callback=log_callback)
        if image_preprocessed is not None and cifar10_color_encoder_loaded:
            item_meta["original_image_dims"] = original_dims
            code_latent_float = cifar10_color_encoder_loaded.predict(image_preprocessed)
            min_val,max_val=np.min(code_latent_float),np.max(code_latent_float)
            item_meta["latent_quant_params"]={"min":float(min_val),"max":float(max_val)}
            quant_uint8=np.zeros_like(code_latent_float,dtype=np.uint8)
            if max_val>min_val:norm_l=(code_latent_float-min_val)/(max_val-min_val);quant_uint8=(norm_l*255).astype(np.uint8)
            elif max_val!=0:quant_uint8=np.full_like(code_latent_float,fill_value=np.clip(np.round(max_val),0,255),dtype=np.uint8)
            item_meta["latent_shape_info"]={"shape":list(quant_uint8.shape),"dtype":str(quant_uint8.dtype)}
            data_to_process_further = quant_uint8.tobytes(); final_arcname_for_zip_member = arcname_to_use + LATENT_FILE_EXTENSION
        else: 
            log_callback(f"[CORE] Échec AE CIFAR10 {item_basename}. Fallback DEFLATE."); comp_method_chosen = zipfile.ZIP_DEFLATED; comp_params_chosen = 1
            item_meta["compression_method_used"]=str(comp_method_chosen); item_meta["compression_params"]=comp_params_chosen
    
    if comp_method_chosen != "MOTEUR_AE_CIFAR10_COLOR":
        with open(file_path_on_disk, 'rb') as f_orig: data_to_process_further = f_orig.read()

    data_after_internal_comp = data_to_process_further
    if comp_method_chosen == "BZIP2":
        try:
            level = comp_params_chosen if isinstance(comp_params_chosen,int) else 9
            data_after_internal_comp = bz2.compress(data_to_process_further, compresslevel=level)
        except Exception as e_bz2: log_callback(f"[CORE] Erreur BZIP2 {item_basename}:{e_bz2}."); item_meta["compression_method_used"]=str(zipfile.ZIP_STORED); item_meta["compression_params"]=None
    elif comp_method_chosen == "LZMA":
        try:
            preset = comp_params_chosen if isinstance(comp_params_chosen,int) else 6
            data_after_internal_comp = lzma.compress(data_to_process_further, format=lzma.FORMAT_XZ, preset=preset)
        except Exception as e_lzma: log_callback(f"[CORE] Erreur LZMA {item_basename}:{e_lzma}."); item_meta["compression_method_used"]=str(zipfile.ZIP_STORED); item_meta["compression_params"]=None
    # --- AJOUT DE LA COMPRESSION ZSTD ---
    elif comp_method_chosen == "ZSTD":
        if ZSTD_AVAILABLE:
            try:
                level = comp_params_chosen if isinstance(comp_params_chosen, int) else 3 
                cctx = zstd.ZstdCompressor(level=level,threads=-1)
                data_after_internal_comp = cctx.compress(data_to_process_further)
                log_callback(f"[CORE] {item_basename} compressé avec ZSTD (niveau {level}).")
            except Exception as e_zstd:
                log_callback(f"[CORE] Erreur ZSTD compression pour {item_basename}: {e_zstd}. Pas de compression interne (STORED).")
                item_meta["compression_method_used"] = str(zipfile.ZIP_STORED) 
                item_meta["compression_params"] = None
                data_after_internal_comp = data_to_process_further # Garder les données originales
        else: 
            log_callback(f"[CORE] ZSTD non disponible. Fallback DEFLATE L1 pour {item_basename}.")
            item_meta["compression_method_used"] = str(zipfile.ZIP_DEFLATED)
            item_meta["compression_params"] = 1
            comp_method_chosen = zipfile.ZIP_DEFLATED # Important pour la logique d'écriture zf.write
            data_after_internal_comp = data_to_process_further # Garder les données originales pour DEFLATE par zf.write
    # --- FIN AJOUT ZSTD ---
    
    data_to_write_to_zip = data_after_internal_comp
    if password_compress and data_to_write_to_zip and CRYPTOGRAPHY_AVAILABLE:
        encrypted_bundle = encrypt_data(data_to_write_to_zip, password_compress, log_callback=log_callback)
        if encrypted_bundle and encrypted_bundle[0] is not None:
            data_to_write_to_zip, salt, iv, tag = encrypted_bundle
            item_meta["encrypted"]=True; item_meta["crypto_salt_hex"]=salt.hex(); item_meta["crypto_iv_hex"]=iv.hex(); item_meta["crypto_tag_hex"]=tag.hex()
            final_arcname_for_zip_member += ENCRYPTED_FILE_EXTENSION
            log_callback(f"[CORE] '{item_basename}' chiffré. Nom archive: {final_arcname_for_zip_member}")
        else: log_callback(f"[CORE] AVERT: Échec chiffrement {item_basename}.")
    
    if data_to_write_to_zip:
        # Si la compression interne (AE, BZ2, LZMA, ZSTD) ou le chiffrement a eu lieu,
        # les données sont "préparées" et doivent être stockées sans compression ZIP supplémentaire.
        if comp_method_chosen in ["MOTEUR_AE_CIFAR10_COLOR", "BZIP2", "LZMA", "ZSTD"] or item_meta["encrypted"]:
            zf.writestr(final_arcname_for_zip_member, data_to_write_to_zip, compress_type=zipfile.ZIP_STORED)
            log_callback(f"[CORE] Données (pré-traitées/chiffrées) '{item_basename}' écrites: '{final_arcname_for_zip_member}' (ZIP_STORED).")
        # Si c'est DEFLATE ou STORED (choisi par l'IA) et non chiffré
        elif comp_method_chosen == zipfile.ZIP_STORED and not item_meta["encrypted"]:
            zf.write(file_path_on_disk, arcname=final_arcname_for_zip_member, compress_type=zipfile.ZIP_STORED)
            log_callback(f"[CORE] Fichier '{item_basename}' écrit (ZIP_STORED) sous '{final_arcname_for_zip_member}'.")
        elif comp_method_chosen == zipfile.ZIP_DEFLATED and not item_meta["encrypted"]:
            # comp_params_chosen contient le niveau de DEFLATE
            level = comp_params_chosen if isinstance(comp_params_chosen, int) else 6
            zf.write(file_path_on_disk, arcname=final_arcname_for_zip_member, compress_type=zipfile.ZIP_DEFLATED, compresslevel=level)
            log_callback(f"[CORE] Fichier '{item_basename}' écrit (DEFLATE L{level}) sous '{final_arcname_for_zip_member}'.")
        else: # Cas de fallback ou erreur de logique (ne devrait pas arriver)
            log_callback(f"[CORE] AVERT: Cas d'écriture non géré pour {item_basename}. Tentative writestr avec STORED.")
            zf.writestr(final_arcname_for_zip_member, data_to_write_to_zip, compress_type=zipfile.ZIP_STORED)
    else: 
        log_callback(f"[CORE] AVERT: Pas de données à écrire pour {item_basename}.")
    
    return item_meta

def compress_to_aic(input_paths, output_aic_path, password_compress=None, log_callback=_default_log): # ... (identique à avant, utilisant _process_and_add_file_to_aic)
    log_callback(f"[CORE] Compression AIC vers '{output_aic_path}' (Chiffré: {'Oui' if password_compress else 'Non'})...")
    all_items_metadata = []
    try:
        with zipfile.ZipFile(output_aic_path, 'w') as zf: 
            for item_path_on_disk in input_paths:
                item_basename_for_archive = os.path.basename(item_path_on_disk)
                if not os.path.exists(item_path_on_disk):
                    log_callback(f"[CORE] '{item_path_on_disk}' non trouvé. Ignoré.")
                    all_items_metadata.append({"original_name": item_basename_for_archive, "status": "not_found"})
                    continue
                if os.path.isfile(item_path_on_disk):
                    meta = _process_and_add_file_to_aic(zf, item_path_on_disk, item_basename_for_archive, password_compress, log_callback)
                    all_items_metadata.append(meta)
                elif os.path.isdir(item_path_on_disk):
                    log_callback(f"[CORE] Traitement dossier: {item_basename_for_archive}")
                    dir_meta = {"original_name": item_basename_for_archive, "type_in_archive": "directory", "status": "processed", "size_original_bytes": 0}
                    all_items_metadata.append(dir_meta) 
                    for root, _, files_in_dir in os.walk(item_path_on_disk):
                        for file_in_d in files_in_dir:
                            full_file_path_on_disk = os.path.join(root, file_in_d)
                            arcname_for_subfile = os.path.join(item_basename_for_archive, os.path.relpath(full_file_path_on_disk, item_path_on_disk))
                            meta_subfile = _process_and_add_file_to_aic(zf, full_file_path_on_disk, arcname_for_subfile, password_compress, log_callback)
                            all_items_metadata.append(meta_subfile) 
            metadata_final_content = {"aicompress_version":"1.0-orchestrator-zstd","items_details":all_items_metadata, "global_encryption_hint":bool(password_compress and CRYPTOGRAPHY_AVAILABLE)}
            zf.writestr(METADATA_FILENAME, json.dumps(metadata_final_content, indent=4))
            log_callback(f"[CORE] Métadonnées écrites. Compression AIC terminée: '{output_aic_path}'"); return True, "Success"
    except Exception as e: log_callback(f"[CORE] ERREUR MAJEURE compression: {e}"); import traceback; log_callback(f"[CORE] Traceback: {traceback.format_exc()}"); return False, f"Error: {e}"

def decompress_aic(aic_file_path, output_extract_path, password_decompress=None, log_callback=_default_log):
    # ... (La fonction decompress_aic complète et corrigée de la réponse précédente va ici)
    # ... (Elle doit maintenant gérer la décompression ZSTD si la méthode est "ZSTD")
    log_callback(f"[CORE_DECOMP] >>> Début Décompression AIC: '{aic_file_path}' vers '{output_extract_path}' (Mdp: {'Oui' if password_decompress else 'Non'})")    
    if not os.path.exists(aic_file_path): log_callback(f"[CORE_DECOMP] ERREUR: Archive '{aic_file_path}' non trouvée."); return False, "FileNotFound"
    try:
        os.makedirs(output_extract_path, exist_ok=True); log_callback(f"[CORE_DECOMP] Dossier sortie: {output_extract_path}")
        with zipfile.ZipFile(aic_file_path, 'r') as zf:
            log_callback(f"[CORE_DECOMP] Archive ZIP ouverte. Namelist: {zf.namelist()}")
            metadata=None; metadata_loaded_successfully=False
            try: metadata_str = zf.read(METADATA_FILENAME); metadata = json.loads(metadata_str); log_callback("[CORE_DECOMP] Métadonnées lues."); metadata_loaded_successfully = True
            except Exception as e_meta: 
                log_callback(f"[CORE_DECOMP] ERREUR métadonnées: {e_meta}. Tentative ZIP standard...");
                try: zf.extractall(path=output_extract_path,pwd=password_decompress.encode('utf-8') if password_decompress else None); return True, "SuccessZipFallback"
                except RuntimeError as e_zip_fb:
                    if password_decompress and "password" in str(e_zip_fb).lower(): return False, "PasswordErrorZipFallback"
                    return False, f"ZipFallbackError: {e_zip_fb}"
                except Exception as e_gen_fb: return False, f"ZipFallbackGenericError: {e_gen_fb}"
            if not (metadata_loaded_successfully and "items_details" in metadata):
                log_callback("[CORE_DECOMP] AVERT: Métadonnées invalides/absentes. Fin anormale."); return False, "InvalidMetadata"
            
            log_callback(f"[CORE_DECOMP] Traitement {len(metadata.get('items_details', []))} items des métadonnées...")
            files_written_this_session = set() 

            for item_meta in metadata["items_details"]:
                original_arcname = item_meta.get("original_name")
                if not original_arcname: log_callback("[CORE_DECOMP] AVERT: item_meta sans original_name."); continue
                output_final_path = os.path.join(output_extract_path, original_arcname)
                log_callback(f"[CORE_DECOMP] Prépa item méta: '{original_arcname}' -> '{output_final_path}'")
                if item_meta.get("status")=="not_found": log_callback(f" Ignoré (not_found): {original_arcname}"); continue
                if item_meta.get("type_in_archive")=="directory": log_callback(f" Création dossier: {output_final_path}"); os.makedirs(output_final_path,exist_ok=True); files_written_this_session.add(output_final_path); continue

                comp_method = item_meta.get("compression_method_used"); is_encrypted = item_meta.get("encrypted", False)
                member_name_in_zip = original_arcname
                if comp_method == "MOTEUR_AE_CIFAR10_COLOR": member_name_in_zip += LATENT_FILE_EXTENSION
                if is_encrypted: member_name_in_zip += ENCRYPTED_FILE_EXTENSION
                                
                data_from_zip = None; 
                try: data_from_zip = zf.read(member_name_in_zip) 
                except KeyError: 
                    if not (comp_method == "MOTEUR_AE_CIFAR10_COLOR" or is_encrypted) and original_arcname in zf.namelist():
                        log_callback(f"[CORE_DECOMP] '{member_name_in_zip}' non trouvé, tentative '{original_arcname}'...")
                        try: data_from_zip = zf.read(original_arcname); member_name_in_zip = original_arcname
                        except Exception as e_read_fallback: log_callback(f"[CORE_DECOMP] Échec lecture fallback '{original_arcname}': {e_read_fallback}"); continue
                    else: log_callback(f"[CORE_DECOMP] ERREUR: Membre '{member_name_in_zip}' non trouvé."); continue
                except RuntimeError as e_rt_m: 
                    if password_decompress and "password" in str(e_rt_m).lower(): return False, f"PasswordErrorFile:{member_name_in_zip}"
                    log_callback(f"[CORE_DECOMP] Erreur Runtime ZIP lecture '{member_name_in_zip}': {e_rt_m}"); continue
                if data_from_zip is None: log_callback(f"[CORE_DECOMP] AVERT: Pas de données lues pour {member_name_in_zip}."); continue
                log_callback(f"[CORE_DECOMP] Membre '{member_name_in_zip}' lu ({len(data_from_zip)} octets).")

                data_to_process = data_from_zip
                if is_encrypted:
                    log_callback(f"[CORE_DECOMP] Déchiffrement de '{original_arcname}'...")
                    if not CRYPTOGRAPHY_AVAILABLE: log_callback("[CORE_DECOMP] ERREUR: Crypto non dispo."); continue
                    if not password_decompress: return False, f"PasswordNeededForDecryption:{original_arcname}"
                    salt=bytes.fromhex(item_meta["crypto_salt_hex"]); iv=bytes.fromhex(item_meta["crypto_iv_hex"]); tag=bytes.fromhex(item_meta["crypto_tag_hex"])
                    data_to_process = decrypt_data(data_from_zip, password_decompress, salt, iv, tag, log_callback=log_callback)
                    if data_to_process is None: return False, f"PasswordErrorDecryption:{original_arcname}"
                
                final_data_to_write = None; write_handled_by_ae_postprocess = False
                if comp_method == "MOTEUR_AE_CIFAR10_COLOR":
                    write_handled_by_ae_postprocess = True
                    if not cifar10_models_loaded_successfully and not ensure_cifar10_color_models_loaded(log_callback=log_callback): log_callback(f"[CORE_DECOMP] ERREUR: Modèles AE non dispo pour '{original_arcname}'."); continue
                    try:
                        latent_shape=item_meta["latent_shape_info"]["shape"]; latent_dtype_str=item_meta["latent_shape_info"]["dtype"]
                        quant_params=item_meta["latent_quant_params"]; original_dims=item_meta["original_image_dims"]
                        if not (latent_shape and latent_dtype_str and quant_params): log_callback(f"[CORE_DECOMP] ERREUR: Infos latentes AE pour '{original_arcname}'."); continue
                        arr=np.frombuffer(data_to_process,dtype=np.uint8).reshape(latent_shape)
                        min_v,max_v=quant_params["min"],quant_params["max"];dequant_float=np.zeros_like(arr,dtype=np.float32); 
                        if max_v>min_v: dequant_float=(arr.astype(np.float32)/255.0)*(max_v-min_v)+min_v
                        elif max_v!=0: dequant_float[:]=min_v
                        img_recon_norm = cifar10_color_decoder_loaded.predict(dequant_float)
                        base, _ = os.path.splitext(original_arcname)
                        save_path = os.path.join(output_extract_path, base + "_reconstructed_cifar_ae.png")
                        if postprocess_and_save_cifar10_ae_output(img_recon_norm, save_path, original_dims, log_callback=log_callback):
                             files_written_by_custom_logic.add(save_path)
                    except Exception as e_ae_proc: log_callback(f"[CORE_DECOMP] Erreur traitement AE de '{original_arcname}': {e_ae_proc}")
                elif comp_method == "BZIP2": 
                    try: final_data_to_write = bz2.decompress(data_to_process); log_callback(f" Décompressé BZIP2: {original_arcname}")
                    except Exception as e: log_callback(f" Erreur décomp BZIP2 {original_arcname}: {e}"); continue
                elif comp_method == "LZMA": 
                    try: final_data_to_write = lzma.decompress(data_to_process, format=lzma.FORMAT_XZ); log_callback(f" Décompressé LZMA: {original_arcname}")
                    except Exception as e: log_callback(f" Erreur décomp LZMA {original_arcname}: {e}"); continue
                elif comp_method == "ZSTD": # NOUVELLE DÉCOMPRESSION ZSTD
                    if ZSTD_AVAILABLE:
                        try: dctx = zstd.ZstdDecompressor(); final_data_to_write = dctx.decompress(data_to_process); log_callback(f" Décompressé ZSTD: {original_arcname}")
                        except Exception as e: log_callback(f" Erreur décomp ZSTD {original_arcname}: {e}"); continue
                    else: log_callback(f"[CORE_DECOMP] ZSTD non dispo pour décompresser {original_arcname}."); continue
                elif comp_method in [str(zipfile.ZIP_DEFLATED), str(zipfile.ZIP_STORED)]:
                    if is_encrypted: final_data_to_write = data_to_process # Données déchiffrées prêtes
                    # Si non chiffré, sera extrait par la boucle de fallback.
                else: final_data_to_write = data_to_process 

                if final_data_to_write is not None:
                    log_callback(f"[CORE_DECOMP] Écriture fichier final '{output_final_path}'...")
                    os.makedirs(os.path.dirname(output_final_path), exist_ok=True)
                    with open(output_final_path, 'wb') as f_out: f_out.write(final_data_to_write)
                    files_written_by_custom_logic.add(output_final_path) # Marquer comme écrit par cette logique
            
            # Boucle de Fallback pour les membres qui n'ont pas été écrits par la logique item_meta
            # (typiquement DEFLATE/STORED non chiffrés, ou fichiers de dossiers)
            log_callback("[CORE_DECOMP] Phase extraction générale des membres ZIP restants...")
            for member_info in zf.infolist():
                member_name = member_info.filename
                target_disk_path = os.path.join(output_extract_path, member_name) # Chemin complet sur le disque

                if member_name == METADATA_FILENAME or target_disk_path in files_written_by_custom_logic:
                    log_callback(f"[CORE_DECOMP_FALLBACK] Ignoré (méta ou déjà écrit): {member_name}")
                    continue
                
                # Ignorer les fichiers de données brutes (latent, ou chiffré qui a servi à créer une sortie)
                # car leur "vrai" contenu a déjà été traité et écrit sous le nom original.
                if member_name.endswith(LATENT_FILE_EXTENSION) or \
                   member_name.endswith(LATENT_FILE_EXTENSION + ENCRYPTED_FILE_EXTENSION) or \
                   (member_name.endswith(ENCRYPTED_FILE_EXTENSION) and not member_name[:-len(ENCRYPTED_FILE_EXTENSION)].endswith(LATENT_FILE_EXTENSION)): # Fichier normal chiffré
                    # Vérifier s'il correspond à un item qui a été traité (AE ou déchiffré)
                    original_name_candidate = member_name.replace(LATENT_FILE_EXTENSION, "").replace(ENCRYPTED_FILE_EXTENSION, "")
                    is_source_for_custom_written_file = False
                    for written_path in files_written_by_custom_logic:
                        if os.path.basename(written_path) == original_name_candidate or \
                           os.path.splitext(os.path.basename(written_path))[0].startswith(os.path.splitext(original_name_candidate)[0] + "_reconstructed_"):
                            is_source_for_custom_written_file = True; break
                    if is_source_for_custom_written_file:
                         log_callback(f"[CORE_DECOMP_FALLBACK] Ignoré (donnée source pour fichier déjà écrit): {member_name}")
                         continue
                
                log_callback(f"[CORE_DECOMP_FALLBACK] Tentative extraction standard de: {member_name}")
                try:
                    zf.extract(member_info, path=output_extract_path, pwd=password_decompress.encode('utf-8') if password_decompress else None)
                    log_callback(f"[CORE_DECOMP_FALLBACK] Extrait (standard): {member_name}")
                except RuntimeError as e_rt_ext:
                     if password_decompress and "password" in str(e_rt_ext).lower(): log_callback(f"[CORE_DECOMP_FALLBACK] Mdp incorrect pour ZIP membre '{member_name}'.")
                     else: log_callback(f"[CORE_DECOMP_FALLBACK] Erreur runtime extraction '{member_name}': {e_rt_ext}")
                except Exception as e_ext_std: log_callback(f"[CORE_DECOMP_FALLBACK] Erreur extraction '{member_name}': {e_ext_std}")
            log_callback(f"[CORE_DECOMP] Fin de la décompression AIC dans '{output_extract_path}'."); return True, "Success"
    except zipfile.BadZipFile: log_callback(f"[CORE_DECOMP] ERREUR: Not a ZIP: {aic_file_path}"); return False, "BadZipFile"
    except RuntimeError as e_global_rt:
         if password_decompress and ("password" in str(e_global_rt).lower()): return False, "PasswordErrorArchive"
         log_callback(f"[CORE_DECOMP] ERREUR runtime AIC: {e_global_rt}"); return False, "RuntimeErrorArchive"
    except Exception as e:
        log_callback(f"[CORE_DECOMP] ERREUR majeure décompression AIC: {e}"); import traceback
        log_callback(f"[CORE_DECOMP] Traceback: {traceback.format_exc()}"); return False, f"UnknownError: {e}"


def decompress_rar(rar_file_path, output_extract_path, password=None, log_callback=_default_log):
    if not RARFILE_AVAILABLE: log_callback("[CORE] Erreur : rarfile non dispo."); return False, "RARLibNotAvailable"
    log_callback(f"[CORE] Décompression RAR de '{rar_file_path}'...")
    if not os.path.exists(rar_file_path): log_callback(f"[CORE] Erreur : Fichier RAR '{rar_file_path}' non trouvé."); return False, "FileNotFound"
    try:
        os.makedirs(output_extract_path, exist_ok=True)
        with rarfile.RarFile(rar_file_path, 'r', pwd=password) as rf: rf.extractall(path=output_extract_path)
        log_callback(f"[CORE] Fichier RAR '{rar_file_path}' extrait."); return True, "Success"
    except rarfile.PasswordRequired: log_callback(f"[CORE] ERREUR: Mdp requis pour RAR '{rar_file_path}'."); return False, "PasswordErrorRAR"
    except rarfile.WrongPassword: log_callback(f"[CORE] ERREUR: Mdp incorrect pour RAR '{rar_file_path}'."); return False, "PasswordErrorRAR" 
    except rarfile.BadRarFile as e: log_callback(f"[CORE] ERREUR: Fichier RAR invalide/corrompu ou mauvais mdp '{rar_file_path}'. {e}"); return False, "BadRarFileOrPassword"
    except Exception as e: log_callback(f"[CORE] Erreur décompression RAR : {e}"); return False, f"UnknownErrorRAR: {e}"

def extract_archive(archive_path, output_dir, password=None, log_callback=_default_log):
    log_callback(f"[CORE] Extraction de '{archive_path}'...")
    if not os.path.exists(archive_path): log_callback(f"[CORE] Erreur : Archive '{archive_path}' non trouvée."); return False, "FileNotFound"
    try: os.makedirs(output_dir, exist_ok=True)
    except Exception as e: log_callback(f"[CORE] Erreur création dossier sortie '{output_dir}': {e}"); return False, "OutputDirError"

    _, extension = os.path.splitext(archive_path); extension = extension.lower()
    analysis_for_extract = "unknown_type"; 
    if AI_ANALYZER_AVAILABLE: analysis_for_extract = analyze_file_content(archive_path, log_callback=log_callback) 

    if extension == DEFAULT_AIC_EXTENSION or analysis_for_extract == "aic_custom_format": 
        return decompress_aic(archive_path, output_dir, password_decompress=password, log_callback=log_callback)
    elif extension == ".zip" or analysis_for_extract == "zip_archive":
        log_callback(f"[CORE] Format .zip standard."); success, status = False, "InitErrorZIP"
        try:
            with zipfile.ZipFile(archive_path, 'r') as zf: zf.extractall(path=output_dir, pwd=password.encode('utf-8') if password else None)
            log_callback(f"[CORE] .zip '{archive_path}' extrait."); success, status = True, "Success"
        except RuntimeError as e_rt:
            if "password" in str(e_rt).lower(): status = "PasswordErrorZIP"
            else: status = f"RuntimeErrorZIP: {e_rt}"
            log_callback(f"[CORE] ERREUR: {status} pour ZIP '{archive_path}'.")
        except Exception as e: status = f"UnknownErrorZIP: {e}"; log_callback(f"[CORE] Erreur ZIP: {e}")
        return success, status
    elif extension == ".rar" or analysis_for_extract == "rar_archive":
        return decompress_rar(archive_path, output_dir, password=password, log_callback=log_callback)
    else:
        if zipfile.is_zipfile(archive_path):
            log_callback(f"[CORE] Type non reconnu mais semble ZIP..."); success, status = False, "InitErrorZIPFallback"
            try:
                with zipfile.ZipFile(archive_path, 'r') as zf: zf.extractall(path=output_dir, pwd=password.encode('utf-8') if password else None)
                log_callback(f"[CORE] Extrait (comme ZIP)."); success, status = True, "Success"
            except RuntimeError as e_rt:
                if "password" in str(e_rt).lower(): status = "PasswordErrorZIPFallback"
                else: status = f"RuntimeErrorZIPFallback: {e_rt}"
                log_callback(f"[CORE] ERREUR: {status} pour ZIP (fallback) '{archive_path}'.")
            except Exception as e: status = f"UnknownErrorZIPFallback: {e}"; log_callback(f"[CORE] Échec fallback ZIP: {e}")
            return success, status
        log_callback(f"[CORE] Format archive non supporté: '{archive_path}' (Analyse: {analysis_for_extract}).")
        return False, "UnsupportedFormat"