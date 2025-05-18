# aicompress/orchestrator.py

import os
import zipfile # Pour les constantes ZIP_DEFLATED, ZIP_STORED
import joblib
# Pandas est importé localement dans get_compression_settings car c'est une grosse dépendance

# --- Fonction de Log par Défaut pour ce Module ---
def _default_log_orchestrator(message):
    print(message)

# --- Imports depuis les autres modules d'AICompress ---
# Ces imports sont nécessaires pour que get_compression_settings puisse fonctionner.
# Ils ont leurs propres mécanismes de fallback si les modules sources ne sont pas là.

AI_ANALYZER_READY_ORCH = False # Flag local
get_file_features_orch = lambda fp, lc: {"type":"analyzer_unavailable_in_orch","error":True}
try:
    from .ai_analyzer import get_file_features as _real_gff_orch, \
                               AI_ANALYZER_AVAILABLE as _analyzer_flag_orch
    if _analyzer_flag_orch:
        get_file_features_orch = _real_gff_orch
        AI_ANALYZER_READY_ORCH = True
    _default_log_orchestrator(f"[ORCHESTRATOR] Dépendance ai_analyzer: {'Prêt' if AI_ANALYZER_READY_ORCH else 'Non Prêt'}")
except ImportError:
    _default_log_orchestrator("[ORCHESTRATOR] AVERT: ai_analyzer.py non trouvé.")

CLASSIC_COMPRESSORS_READY_ORCH = False
ZSTD_READY_ORCH = False
METHOD_STORED,METHOD_DEFLATE,METHOD_BZIP2,METHOD_LZMA,METHOD_ZSTD = "S","D","B","L","Z" 
try:
    from .classic_compressors import (METHOD_STORED as _M_S_cl, METHOD_DEFLATE as _M_D_cl, 
                                      METHOD_BZIP2 as _M_B_cl, METHOD_LZMA as _M_L_cl, 
                                      METHOD_ZSTD as _M_Z_cl,
                                      ZSTD_AVAILABLE as ZSTD_FLAG_cl,
                                      CLASSIC_COMPRESSORS_READY as CC_READY_FLAG_cl)
    if CC_READY_FLAG_cl:
        METHOD_STORED,METHOD_DEFLATE,METHOD_BZIP2,METHOD_LZMA,METHOD_ZSTD = _M_S_cl,_M_D_cl,_M_B_cl,_M_L_cl,_M_Z_cl
        ZSTD_READY_ORCH = ZSTD_FLAG_cl
        CLASSIC_COMPRESSORS_READY_ORCH = True
    _default_log_orchestrator(f"[ORCHESTRATOR] Dépendance classic_compressors: {'Prêt' if CLASSIC_COMPRESSORS_READY_ORCH else 'Non Prêt'}")
except ImportError:
    _default_log_orchestrator("[ORCHESTRATOR] AVERT: classic_compressors.py non trouvé.")

AE_ENGINE_READY_ORCH = False
PIL_AVAILABLE_FOR_AE_ORCH = False
KERAS_AVAILABLE_FOR_AE_ORCH = False
ensure_ae_models_loaded_orch = lambda lc: False
try:
    from .ae_engine import (ensure_cifar10_color_ae_models_loaded as _ensure_ae_orch,
                            PIL_AVAILABLE_AE as _pil_ae_orch, 
                            KERAS_AVAILABLE_AE as _keras_ae_orch,
                            AE_ENGINE_LOADED as _ae_loaded_flag_orch)
    if _ae_loaded_flag_orch:
        ensure_ae_models_loaded_orch = _ensure_ae_orch
        PIL_AVAILABLE_FOR_AE_ORCH = _pil_ae_orch
        KERAS_AVAILABLE_FOR_AE_ORCH = _keras_ae_orch
        AE_ENGINE_READY_ORCH = True # Le moteur est chargé et ses dépendances sont OK
    _default_log_orchestrator(f"[ORCHESTRATOR] Dépendance ae_engine: {'Prêt' if AE_ENGINE_READY_ORCH else 'Non Prêt'}")
except ImportError:
    _default_log_orchestrator("[ORCHESTRATOR] AVERT: ae_engine.py non trouvé.")


# --- Chemins et Variables Globales pour le Modèle Orchestrateur ---
_MODULE_DIR_ORCH = os.path.dirname(__file__)
ORCHESTRATOR_MODEL_PATH_ORCH = os.path.join(_MODULE_DIR_ORCH, "compression_orchestrator_model.joblib")
ORCHESTRATOR_LABEL_ENCODER_PATH_ORCH = os.path.join(_MODULE_DIR_ORCH, "orchestrator_label_encoder.joblib")

_orchestrator_pipeline_internal = None
_orchestrator_label_encoder_internal = None
ORCHESTRATOR_IS_READY = False # Flag exporté par ce module

def load_orchestrator_model(log_callback=_default_log_orchestrator):
    """Charge le pipeline du modèle orchestrateur et son encodeur de label."""
    global _orchestrator_pipeline_internal, _orchestrator_label_encoder_internal, ORCHESTRATOR_IS_READY
    
    if ORCHESTRATOR_IS_READY: # Déjà chargé et réussi
        return True

    if not (os.path.exists(ORCHESTRATOR_MODEL_PATH_ORCH) and os.path.exists(ORCHESTRATOR_LABEL_ENCODER_PATH_ORCH)):
        log_callback(f"[ORCHESTRATOR] ERREUR: Modèle ({ORCHESTRATOR_MODEL_PATH_ORCH}) "
                     f"ou Label Encoder ({ORCHESTRATOR_LABEL_ENCODER_PATH_ORCH}) non trouvé.")
        ORCHESTRATOR_IS_READY = False
        return False
    try:
        import joblib # Nécessaire pour charger les modèles sklearn
        log_callback(f"[ORCHESTRATOR] Chargement du Pipeline: {ORCHESTRATOR_MODEL_PATH_ORCH}")
        _orchestrator_pipeline_internal = joblib.load(ORCHESTRATOR_MODEL_PATH_ORCH)
        log_callback(f"[ORCHESTRATOR] Chargement du Label Encoder: {ORCHESTRATOR_LABEL_ENCODER_PATH_ORCH}")
        _orchestrator_label_encoder_internal = joblib.load(ORCHESTRATOR_LABEL_ENCODER_PATH_ORCH)
        log_callback("[ORCHESTRATOR] Modèle Orchestrateur et Label Encoder chargés avec succès.")
        ORCHESTRATOR_IS_READY = True # Mettre à jour le flag
        return True
    except ModuleNotFoundError as e_mnfe: 
        log_callback(f"[ORCHESTRATOR] ERREUR de dépendance lors du chargement du modèle: {e_mnfe} (pandas ou sklearn manquant?)")
        ORCHESTRATOR_IS_READY = False
        return False
    except Exception as e:
        log_callback(f"[ORCHESTRATOR] ERREUR lors du chargement du modèle Orchestrateur: {e}")
        import traceback
        log_callback(f"[ORCHESTRATOR] Traceback: {traceback.format_exc()}")
        ORCHESTRATOR_IS_READY = False
        return False

# Tentative de chargement du modèle lors de l'import du module orchestrator.py
if _initial_orchestrator_load_status_unused := load_orchestrator_model(_default_log_orchestrator):
    pass 


def get_compression_settings(file_path: str, analysis_result_str_ignored: str, log_callback=_default_log_orchestrator) -> tuple:
    """
    Utilise l'IA Orchestrateur pour déterminer la meilleure méthode de compression.
    Retourne (str_method_name_constant, params_dict_for_method).
    """
    log_callback(f"[ORCHESTRATOR] Début get_compression_settings pour: '{os.path.basename(file_path)}'")

    if not ORCHESTRATOR_IS_READY: # Vérifier si le modèle de ce module est prêt
        log_callback("[ORCHESTRATOR] ERREUR CRITIQUE: Modèle orchestrateur non chargé. Fallback: DEFLATE L6.")
        return METHOD_DEFLATE, {"level": 6} 

    if not AI_ANALYZER_READY_ORCH: # Utiliser le flag local initialisé
        log_callback("[ORCHESTRATOR] AI Analyzer non disponible (via orchestrator). Fallback: DEFLATE L6.")
        return METHOD_DEFLATE, {"level": 6}
        
    features = get_file_features_orch(file_path, log_callback=log_callback) # Utiliser la fonction initialisée
    if features.get("error"):
        log_callback(f"[ORCHESTRATOR] Erreur extraction features pour {file_path}. Fallback: DEFLATE L6.")
        return METHOD_DEFLATE, {"level": 6}

    try:
        import pandas as pd 
        input_df = pd.DataFrame([{
            "file_type_analysis": features["type"], 
            "original_size_bytes": features["size_bytes"],
            "entropy_normalized": features["entropy_normalized"],
            "quick_comp_ratio": features["quick_comp_ratio"]
        }])
    except ImportError:
        log_callback("[ORCHESTRATOR] Pandas non trouvé. Impossible de prédire. Fallback DEFLATE L6.")
        return METHOD_DEFLATE, {"level": 6}
    except Exception as e_df:
        log_callback(f"[ORCHESTRATOR] Erreur création DataFrame pour prédiction: {e_df}. Fallback DEFLATE L6.")
        return METHOD_DEFLATE, {"level": 6}

    try:
        predicted_method_encoded = _orchestrator_pipeline_internal.predict(input_df)
        predicted_method_name = _orchestrator_label_encoder_internal.inverse_transform(predicted_method_encoded)[0]
        log_callback(f"[ORCHESTRATOR] Méthode prédite par IA pour '{os.path.basename(file_path)}': {predicted_method_name}")
    except Exception as e_predict:
        log_callback(f"[ORCHESTRATOR] Erreur lors de la prédiction par l'orchestrateur: {e_predict}. Fallback DEFLATE L6.")
        return METHOD_DEFLATE, {"level": 6}
    
    # Interpréter la méthode prédite et retourner les bons paramètres pour core.py/aic_file_handler.py
    if predicted_method_name == METHOD_STORED: 
        return METHOD_STORED, {}
    elif predicted_method_name.startswith(METHOD_DEFLATE + "_L"):
        try: level = int(predicted_method_name.split("_L")[1]); return METHOD_DEFLATE, {"level": level}
        except: log_callback(f"[ORCHESTRATOR] Erreur parsing DEFLATE: {predicted_method_name}"); return METHOD_DEFLATE, {"level": 6}
    elif predicted_method_name.startswith(METHOD_BZIP2 + "_L"):
        try: level = int(predicted_method_name.split("_L")[1]); return METHOD_BZIP2, {"level": level} 
        except: log_callback(f"[ORCHESTRATOR] Erreur parsing BZIP2: {predicted_method_name}"); return METHOD_BZIP2, {"level": 9}
    elif predicted_method_name.startswith(METHOD_LZMA + "_P"):
        try: preset = int(predicted_method_name.split("_P")[1]); return METHOD_LZMA, {"preset": preset}
        except: log_callback(f"[ORCHESTRATOR] Erreur parsing LZMA: {predicted_method_name}"); return METHOD_LZMA, {"preset": 6}
    elif predicted_method_name.startswith(METHOD_ZSTD + "_L"):
        if not ZSTD_READY_ORCH: # Vérifier si Zstd est réellement utilisable via classic_compressors
            log_callback(f"[ORCHESTRATOR] Zstd prédit mais lib non dispo via classic_compressors. Fallback DEFLATE L6.")
            return METHOD_DEFLATE, {"level": 6}
        try: level = int(predicted_method_name.split("_L")[1]); return METHOD_ZSTD, {"level": level} 
        except: log_callback(f"[ORCHESTRATOR] Erreur parsing ZSTD: {predicted_method_name}. Fallback ZSTD L3."); return METHOD_ZSTD, {"level": 3} 
    
    elif predicted_method_name == "MOTEUR_AE_CIFAR10_COLOR":
        if not AE_ENGINE_READY_ORCH: # Vérifier si ae_engine est prêt
            log_callback(f"[ORCHESTRATOR] IA prédit AE, mais moteur AE non opérationnel. Fallback DEFLATE L1.")
            return METHOD_DEFLATE, {"level": 1}
        
        # La vérification de la taille de l'image pour l'AE doit se faire ici ou avant
        # car get_compression_settings doit retourner la décision finale.
        if PIL_AVAILABLE_FOR_AE_ORCH : # ae_engine a déjà vérifié Keras pour son propre flag
            try: 
                from PIL import Image as PILImage # Import local pour éviter dépendance globale si Pillow manque juste ici
                with PILImage.open(file_path) as img: width, height = img.size
                MAX_AE_INPUT_DIM_ORCH = 256 
                if width <= MAX_AE_INPUT_DIM_ORCH and height <= MAX_AE_INPUT_DIM_ORCH: 
                    return "MOTEUR_AE_CIFAR10_COLOR", {}
                else: 
                    log_callback(f"[ORCHESTRATOR] IA prédit AE, mais image trop grande ({width}x{height}). Fallback DEFLATE L1.")
                    return METHOD_DEFLATE, {"level": 1}
            except Exception as e_img_check_orch: 
                log_callback(f"[ORCHESTRATOR] IA prédit AE, mais erreur vérification image ({e_img_check_orch}). Fallback DEFLATE L1.")
                return METHOD_DEFLATE, {"level": 1}
        else: 
            log_callback(f"[ORCHESTRATOR] IA prédit AE, mais Pillow non dispo dans ae_engine. Fallback DEFLATE L1.")
            return METHOD_DEFLATE, {"level": 1}
    else: 
        log_callback(f"[ORCHESTRATOR] Méthode prédite '{predicted_method_name}' non gérée. Fallback DEFLATE L6.")
        return METHOD_DEFLATE, {"level": 6}

# Fin de aicompress/orchestrator.py