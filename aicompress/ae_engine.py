# aicompress/ae_engine.py

import os
import numpy as np

# --- Fonction de Log par Défaut pour ce Module ---
def _default_log_ae(message):
    print(message)

# --- Imports Conditionnels et Flags de Disponibilité ---
try:
    from PIL import Image, ImageOps
    PIL_AVAILABLE_AE = True
    _default_log_ae("[AE_ENGINE] Bibliothèque Pillow chargée.")
except ImportError:
    PIL_AVAILABLE_AE = False
    _default_log_ae("AVERTISSEMENT (ae_engine.py): Pillow non trouvée. Moteur AE image désactivé.")

try:
    import tensorflow as tf
    KERAS_AVAILABLE_AE = True
    _default_log_ae("[AE_ENGINE] Bibliothèque TensorFlow/Keras chargée.")
except ImportError:
    KERAS_AVAILABLE_AE = False
    _default_log_ae("AVERTISSEMENT (ae_engine.py): TensorFlow/Keras non trouvé. Moteur AE désactivé.")

# --- Constantes et Variables Globales pour les Modèles AE ---
# (Ces chemins sont relatifs à la racine du projet AICompressProject)
CIFAR10_COLOR_ENCODER_PATH_AE = "encoder_cifar10_color_v5.keras"
CIFAR10_COLOR_DECODER_PATH_AE = "decoder_cifar10_color_v5.keras"

cifar10_color_encoder = None
cifar10_color_decoder = None
cifar10_ae_models_loaded = False

# --- Fonctions de Gestion des Modèles Keras ---
def _load_keras_model_pair_ae(encoder_path, decoder_path, model_name_log, log_callback=_default_log_ae):
    """Charge une paire de modèles encodeur/décodeur Keras."""
    if not KERAS_AVAILABLE_AE:
        log_callback(f"[AE_ENGINE] Keras/TF non dispo pour charger {model_name_log}.")
        return None, None, False

    # Vérifier si les chemins sont absolus, sinon les rendre relatifs au script principal (supposé être à la racine)
    # C'est un peu une rustine, idéalement les chemins seraient gérés par une config
    if not os.path.isabs(encoder_path):
        encoder_path_abs = os.path.join(os.getcwd(), encoder_path) # Suppose que le script est lancé depuis la racine du projet
        log_callback(f"[AE_ENGINE] Chemin relatif encodeur, tentative avec: {encoder_path_abs}")
    else:
        encoder_path_abs = encoder_path

    if not os.path.isabs(decoder_path):
        decoder_path_abs = os.path.join(os.getcwd(), decoder_path)
        log_callback(f"[AE_ENGINE] Chemin relatif décodeur, tentative avec: {decoder_path_abs}")
    else:
        decoder_path_abs = decoder_path

    if not (os.path.exists(encoder_path_abs) and os.path.exists(decoder_path_abs)):
        log_callback(f"[AE_ENGINE] ERREUR: Fichiers modèles {model_name_log} non trouvés aux chemins finaux ({encoder_path_abs} ou {decoder_path_abs}).")
        return None, None, False
    try:
        log_callback(f"[AE_ENGINE] Chargement encodeur {model_name_log}: {encoder_path_abs}")
        encoder = tf.keras.models.load_model(encoder_path_abs, compile=False)
        log_callback(f"[AE_ENGINE] Chargement décodeur {model_name_log}: {decoder_path_abs}")
        decoder = tf.keras.models.load_model(decoder_path_abs, compile=False)
        log_callback(f"[AE_ENGINE] Modèles {model_name_log} chargés avec succès.")
        return encoder, decoder, True
    except Exception as e:
        log_callback(f"[AE_ENGINE] ERREUR lors du chargement des modèles {model_name_log}: {e}")
        import traceback
        log_callback(f"[AE_ENGINE] Traceback: {traceback.format_exc()}")
        return None, None, False

def ensure_cifar10_color_ae_models_loaded(log_callback=_default_log_ae):
    """Charge les modèles CIFAR10 couleur si ce n'est pas déjà fait."""
    global cifar10_color_encoder, cifar10_color_decoder, cifar10_ae_models_loaded
    if cifar10_ae_models_loaded: # Déjà chargés et succès
        return True

    # Si pas chargés ou échec précédent, retenter.
    # Si KERAS_AVAILABLE_AE est False, _load_keras_model_pair_ae retournera False rapidement.
    enc, dec, success = _load_keras_model_pair_ae(
        CIFAR10_COLOR_ENCODER_PATH_AE, 
        CIFAR10_COLOR_DECODER_PATH_AE,
        "CIFAR10 Couleur V5 AE",
        log_callback
    )
    if success:
        cifar10_color_encoder = enc
        cifar10_color_decoder = dec
    cifar10_ae_models_loaded = success # Mettre à jour le statut global
    return cifar10_ae_models_loaded

# --- Fonctions de Prétraitement et Post-traitement Image ---
def preprocess_image_for_ae(image_path, target_size=(32,32), log_callback=_default_log_ae):
    """Prépare une image pour l'encodeur AE (redimensionne, normalise, RGB)."""
    if not PIL_AVAILABLE_AE:
        log_callback("[AE_ENGINE] Pillow non disponible pour le prétraitement d'image."); return None, None
    try:
        img = Image.open(image_path)
        original_dims = img.size # (width, height)
        img_format = img.format

        if img.mode != 'RGB':
            img = img.convert('RGB')
            log_callback(f"[AE_ENGINE] Image {os.path.basename(image_path)} convertie en RGB.")

        img_resized = img.resize(target_size, Image.Resampling.LANCZOS)
        img_array = np.array(img_resized).astype('float32') / 255.0 # Normaliser
        img_array_reshaped = np.reshape(img_array, (1, target_size[1], target_size[0], 3)) # Batch_dim, H, W, C

        log_callback(f"[AE_ENGINE] Image {os.path.basename(image_path)} (format:{img_format}, dims:{original_dims}) prétraitée pour AE (taille:{target_size}).")
        return img_array_reshaped, original_dims
    except Exception as e:
        log_callback(f"[AE_ENGINE] Erreur lors du prétraitement de l'image {os.path.basename(image_path)}: {e}")
        return None, None

def postprocess_and_save_decoded_image(image_array_norm, output_path, original_target_dims, target_ae_size=(32,32), log_callback=_default_log_ae):
    """Dénormalise, redimensionne (optionnel) et sauvegarde l'image reconstruite."""
    if not PIL_AVAILABLE_AE:
        log_callback("[AE_ENGINE] Pillow non disponible pour post-traitement/sauvegarde."); return False
    try:
        img_data = image_array_norm.reshape(target_ae_size[1], target_ae_size[0], 3) # H, W, C
        img_data = np.clip(img_data * 255, 0, 255).astype(np.uint8) # Dénormaliser

        img_to_save = Image.fromarray(img_data, mode='RGB')

        if original_target_dims and (original_target_dims[0] != target_ae_size[0] or original_target_dims[1] != target_ae_size[1]):
            log_callback(f"[AE_ENGINE] Redimensionnement de l'image reconstruite de {target_ae_size} vers {original_target_dims}...")
            img_to_save = img_to_save.resize(original_target_dims, Image.Resampling.LANCZOS)

        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir): # S'assurer que le dossier existe
            os.makedirs(output_dir, exist_ok=True)

        img_to_save.save(output_path) # Pillow déduit le format de l'extension
        log_callback(f"[AE_ENGINE] Image reconstruite par AE sauvegardée : {output_path}")
        return True
    except Exception as e:
        log_callback(f"[AE_ENGINE] Erreur lors du post-traitement/sauvegarde de l'image AE {output_path}: {e}")
        return False

# --- Fonctions d'Encodage et Décodage AE ---
def encode_image_with_cifar10_ae(image_path, log_callback=_default_log_ae) -> dict | None:
    """
    Encode une image en utilisant l'AE CIFAR10 couleur.
    Retourne un dictionnaire avec les données latentes et les métadonnées, ou None si échec.
    """
    if not (KERAS_AVAILABLE_AE and PIL_AVAILABLE_AE):
        log_callback("[AE_ENGINE] Encodeur AE non disponible (Keras ou Pillow manquant).")
        return {"error": True, "error_message": "Keras ou Pillow manquant pour l'AE."}

    if not ensure_cifar10_color_ae_models_loaded(log_callback=log_callback):
        log_callback("[AE_ENGINE] Échec du chargement des modèles AE CIFAR10.")
        return {"error": True, "error_message": "Échec chargement modèles AE."}

    if cifar10_color_encoder is None: # Double vérification
        log_callback("[AE_ENGINE] Encodeur CIFAR10 non chargé en mémoire.")
        return {"error": True, "error_message": "Encodeur AE non chargé."}

    image_preprocessed, original_dims = preprocess_image_for_ae(image_path, target_size=(32,32), log_callback=log_callback)
    if image_preprocessed is None:
        return {"error": True, "error_message": "Échec du prétraitement de l'image."}

    try:
        code_latent_float = cifar10_color_encoder.predict(image_preprocessed)

        min_val, max_val = np.min(code_latent_float), np.max(code_latent_float)
        quant_params = {"min": float(min_val), "max": float(max_val)}

        quantized_latent_uint8 = np.zeros_like(code_latent_float, dtype=np.uint8)
        if max_val > min_val:
            normalized_latent = (code_latent_float - min_val) / (max_val - min_val)
            quantized_latent_uint8 = (normalized_latent * 255).astype(np.uint8)
        elif max_val != 0: # Toutes valeurs identiques non nulles
            quantized_latent_uint8 = np.full_like(code_latent_float, 
                                                 fill_value=np.clip(np.round(max_val),0,255),
                                                 dtype=np.uint8)
        # Si min_val == max_val == 0, quantized_latent_uint8 reste 0, ce qui est correct.

        latent_data_bytes = quantized_latent_uint8.tobytes()
        latent_shape = list(quantized_latent_uint8.shape)
        latent_dtype_str = str(quantized_latent_uint8.dtype)

        log_callback(f"[AE_ENGINE] Code latent pour {os.path.basename(image_path)} généré et quantifié.")
        return {
            "latent_data_bytes": latent_data_bytes,
            "latent_shape_info": {"shape": latent_shape, "dtype": latent_dtype_str},
            "quant_params": quant_params,
            "original_image_dims": original_dims,
            "error": False,
            "error_message": None
        }
    except Exception as e:
        log_callback(f"[AE_ENGINE] Erreur pendant l'encodage AE de {os.path.basename(image_path)}: {e}")
        return {"error": True, "error_message": str(e)}


def decode_latent_to_image_cifar10_ae(latent_data_bytes, latent_shape_info, quant_params, 
                                      original_dims, output_path_suggestion, log_callback=_default_log_ae) -> bool:
    """
    Décode un code latent quantifié en image et la sauvegarde.
    Retourne True si succès, False sinon.
    """
    if not (KERAS_AVAILABLE_AE and PIL_AVAILABLE_AE):
        log_callback("[AE_ENGINE] Décodeur AE non disponible (Keras ou Pillow manquant)."); return False

    if not ensure_cifar10_color_ae_models_loaded(log_callback=log_callback):
        log_callback("[AE_ENGINE] Échec du chargement des modèles AE CIFAR10 pour décodage."); return False

    if cifar10_color_decoder is None:
         log_callback("[AE_ENGINE] Décodeur CIFAR10 non chargé en mémoire."); return False

    try:
        latent_shape = latent_shape_info["shape"]
        latent_dtype_str = latent_shape_info["dtype"]

        if latent_dtype_str == 'uint8':
            quantized_latent_array = np.frombuffer(latent_data_bytes, dtype=np.uint8)
        else:
            log_callback(f"[AE_ENGINE] Dtype latent non supporté: {latent_dtype_str}"); return False

        quantized_latent_array = quantized_latent_array.reshape(latent_shape)

        min_val, max_val = quant_params["min"], quant_params["max"]
        dequantized_latent_float = np.zeros_like(quantized_latent_array, dtype=np.float32)
        if max_val > min_val:
            dequantized_latent_float = (quantized_latent_array.astype(np.float32) / 255.0) * (max_val - min_val) + min_val
        elif max_val != 0: # toutes valeurs identiques non nulles
            dequantized_latent_float[:] = min_val
        # Si min_val == max_val == 0, dequantized_latent_float reste 0.0

        image_reconstructed_norm = cifar10_color_decoder.predict(dequantized_latent_float)

        return postprocess_and_save_decoded_image(image_reconstructed_norm, 
                                                  output_path_suggestion, 
                                                  original_dims, 
                                                  target_ae_size=(32,32), # Taille sur laquelle l'AE a été entraîné
                                                  log_callback=log_callback)
    except Exception as e:
        log_callback(f"[AE_ENGINE] Erreur pendant le décodage AE: {e}")
        import traceback
        log_callback(f"[AE_ENGINE] Traceback: {traceback.format_exc()}")
        return False

if KERAS_AVAILABLE_AE and PIL_AVAILABLE_AE:
    AE_ENGINE_LOADED = True
    _default_log_ae("[AE_ENGINE] Moteur AE marqué comme pleinement opérationnel (Keras et Pillow OK).")
else:
    AE_ENGINE_LOADED = False
    _default_log_ae("[AE_ENGINE] Moteur AE marqué comme NON opérationnel (Keras ou Pillow manquant).")
# Fin de aicompress/ae_engine.py