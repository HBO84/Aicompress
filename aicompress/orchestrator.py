# aicompress/orchestrator.py (sans Autoencodeur)

import os
import zipfile 
import joblib
# Pandas est importé localement dans get_compression_settings

def _default_log_orchestrator(message):
    print(f"[ORCHESTRATOR] {message}") # Ajout d'un préfixe pour clarté

# --- Imports depuis les autres modules d'AICompress ---
AI_ANALYZER_READY_ORCH = False 
get_file_features_orch = lambda fp, lc: {"type":"analyzer_unavailable_in_orch","error":True}
try:
    from .ai_analyzer import get_file_features as _real_gff_orch, AI_ANALYZER_AVAILABLE as _analyzer_flag_orch
    if _analyzer_flag_orch: 
        get_file_features_orch = _real_gff_orch
        AI_ANALYZER_READY_ORCH = True
    _default_log_orchestrator(f"Dépendance ai_analyzer: {'Prêt' if AI_ANALYZER_READY_ORCH else 'Non Prêt'}")
except ImportError:
    _default_log_orchestrator("AVERTISSEMENT: ai_analyzer.py non trouvé.")

CLASSIC_COMPRESSORS_READY_ORCH = False
ZSTD_READY_ORCH = False
BROTLI_READY_ORCH = False
METHOD_STORED, METHOD_DEFLATE, METHOD_BZIP2, METHOD_LZMA, METHOD_ZSTD, METHOD_BROTLI = \
    "STORED", "DEFLATE", "BZIP2", "LZMA", "ZSTD", "BROTLI" # Noms de base
try:
    from .classic_compressors import (
        METHOD_STORED as _M_S_cl, METHOD_DEFLATE as _M_D_cl, 
        METHOD_BZIP2 as _M_B_cl, METHOD_LZMA as _M_L_cl, 
        METHOD_ZSTD as _M_Z_cl, METHOD_BROTLI as _M_BR_cl, # Constantes de base
        ZSTD_AVAILABLE as ZSTD_FLAG_cl,
        BROTLI_AVAILABLE as BROTLI_FLAG_cl, 
        CLASSIC_COMPRESSORS_READY as CC_READY_FLAG_cl
    )
    if CC_READY_FLAG_cl:
        METHOD_STORED, METHOD_DEFLATE, METHOD_BZIP2, METHOD_LZMA, METHOD_ZSTD, METHOD_BROTLI = \
            _M_S_cl, _M_D_cl, _M_B_cl, _M_L_cl, _M_Z_cl, _M_BR_cl
        ZSTD_READY_ORCH = ZSTD_FLAG_cl
        BROTLI_READY_ORCH = BROTLI_FLAG_cl
        CLASSIC_COMPRESSORS_READY_ORCH = True
    _default_log_orchestrator(f"Dépendance classic_compressors: {'Prêt' if CLASSIC_COMPRESSORS_READY_ORCH else 'Non Prêt'}")
    if CLASSIC_COMPRESSORS_READY_ORCH:
        _default_log_orchestrator(f"  ZSTD: {'OK' if ZSTD_READY_ORCH else 'X'}, Brotli: {'OK' if BROTLI_READY_ORCH else 'X'}")
except ImportError:
    _default_log_orchestrator("AVERTISSEMENT: classic_compressors.py non trouvé.")

# SUPPRESSION des imports et variables globales pour AE_ENGINE
# AE_ENGINE_READY_ORCH = False
# PIL_AVAILABLE_FOR_AE_ORCH = False
# KERAS_AVAILABLE_FOR_AE_ORCH = False
# ensure_ae_models_loaded_orch = lambda lc: False
# try:
#     from .ae_engine import (ensure_cifar10_color_ae_models_loaded as _ensure_ae_orch,
#                             PIL_AVAILABLE_AE as _pil_ae_orch, KERAS_AVAILABLE_AE as _keras_ae_orch,
#                             AE_ENGINE_LOADED as _ae_loaded_flag_orch)
#     if _ae_loaded_flag_orch:
#         ensure_ae_models_loaded_orch = _ensure_ae_orch; PIL_AVAILABLE_FOR_AE_ORCH = _pil_ae_orch
#         KERAS_AVAILABLE_FOR_AE_ORCH = _keras_ae_orch; AE_ENGINE_READY_ORCH = True
#     _default_log_orchestrator(f"Dépendance ae_engine: {'Prêt' if AE_ENGINE_READY_ORCH else 'Non Prêt'}")
# except ImportError:
#     _default_log_orchestrator("AVERTISSEMENT: ae_engine.py non trouvé.")

_MODULE_DIR_ORCH = os.path.dirname(__file__)
ORCHESTRATOR_MODEL_PATH_ORCH = os.path.join(_MODULE_DIR_ORCH, "compression_orchestrator_model.joblib")
ORCHESTRATOR_LABEL_ENCODER_PATH_ORCH = os.path.join(_MODULE_DIR_ORCH, "orchestrator_label_encoder.joblib")
_orchestrator_pipeline_internal = None
_orchestrator_label_encoder_internal = None
ORCHESTRATOR_IS_READY = False 

def load_orchestrator_model(log_callback=_default_log_orchestrator):
    global _orchestrator_pipeline_internal, _orchestrator_label_encoder_internal, ORCHESTRATOR_IS_READY
    if ORCHESTRATOR_IS_READY: return True
    if not (os.path.exists(ORCHESTRATOR_MODEL_PATH_ORCH) and os.path.exists(ORCHESTRATOR_LABEL_ENCODER_PATH_ORCH)):
        log_callback(f"ERREUR: Modèle Orchestrateur ou Label Encoder non trouvé."); ORCHESTRATOR_IS_READY = False; return False
    try:
        import joblib 
        log_callback(f"Chargement du Pipeline: {ORCHESTRATOR_MODEL_PATH_ORCH}")
        _orchestrator_pipeline_internal = joblib.load(ORCHESTRATOR_MODEL_PATH_ORCH)
        log_callback(f"Chargement du Label Encoder: {ORCHESTRATOR_LABEL_ENCODER_PATH_ORCH}")
        _orchestrator_label_encoder_internal = joblib.load(ORCHESTRATOR_LABEL_ENCODER_PATH_ORCH)
        log_callback("Modèle Orchestrateur et Label Encoder chargés avec succès.")
        ORCHESTRATOR_IS_READY = True
        return True
    except ModuleNotFoundError as e_mnfe: 
        log_callback(f"ERREUR de dépendance lors du chargement du modèle: {e_mnfe} (pandas ou sklearn manquant?)")
        ORCHESTRATOR_IS_READY = False
        return False
    except Exception as e:
        log_callback(f"ERREUR lors du chargement du modèle Orchestrateur: {e}")
        # import traceback # Optionnel pour plus de détails
        # log_callback(f"Traceback: {traceback.format_exc()}")
        ORCHESTRATOR_IS_READY = False
        return False

if _initial_orchestrator_load_status_unused_ := load_orchestrator_model(_default_log_orchestrator):
    pass 

def get_compression_settings(file_path: str, analysis_result_str_ignored: str, log_callback=_default_log_orchestrator) -> tuple:
    log_callback(f"Début get_compression_settings pour: '{os.path.basename(file_path)}'")
    
    if not ORCHESTRATOR_IS_READY: 
        log_callback("ERREUR CRITIQUE: Modèle orchestrateur non chargé. Fallback: DEFLATE L6.")
        return METHOD_DEFLATE, {"level": 6} 

    if not AI_ANALYZER_READY_ORCH: 
        log_callback("AI Analyzer non disponible (via orchestrator). Fallback: DEFLATE L6.")
        return METHOD_DEFLATE, {"level": 6}
        
    features = get_file_features_orch(file_path, log_callback=log_callback)
    if features.get("error"):
        log_callback(f"Erreur extraction features pour {os.path.basename(file_path)}. Fallback: DEFLATE L6.")
        return METHOD_DEFLATE, {"level": 6}

    try:
        import pandas as pd 
        input_df = pd.DataFrame([{
            "file_type_analysis": features["type"], 
            "original_size_bytes": features["size_bytes"],
            "entropy_normalized": features["entropy_normalized"],
            "quick_comp_ratio": features.get("quick_comp_ratio", 1.0) # S'assurer que quick_comp_ratio a un fallback
        }])
    except ImportError:
        log_callback("Pandas non trouvé. Impossible de prédire. Fallback DEFLATE L6.")
        return METHOD_DEFLATE, {"level": 6}
    except Exception as e_df:
        log_callback(f"Erreur création DataFrame pour prédiction: {e_df}. Fallback DEFLATE L6.")
        return METHOD_DEFLATE, {"level": 6}

    try:
        predicted_method_encoded = _orchestrator_pipeline_internal.predict(input_df)
        predicted_method_name = _orchestrator_label_encoder_internal.inverse_transform(predicted_method_encoded)[0]
        log_callback(f"Méthode prédite par IA pour '{os.path.basename(file_path)}': {predicted_method_name}")
    except Exception as e_predict:
        log_callback(f"Erreur lors de la prédiction par l'orchestrateur: {e_predict}. Fallback DEFLATE L6.")
        return METHOD_DEFLATE, {"level": 6}
    
    # Interpréter la méthode prédite
    if predicted_method_name == METHOD_STORED: 
        return METHOD_STORED, {}
    elif predicted_method_name.startswith(METHOD_DEFLATE + "_L"):
        try: level = int(predicted_method_name.split("_L")[1]); return METHOD_DEFLATE, {"level": level}
        except: log_callback(f"Erreur parsing DEFLATE: {predicted_method_name}"); return METHOD_DEFLATE, {"level": 6} 
    elif predicted_method_name.startswith(METHOD_BZIP2 + "_L"):
        try: level = int(predicted_method_name.split("_L")[1]); return METHOD_BZIP2, {"level": level} 
        except: log_callback(f"Erreur parsing BZIP2: {predicted_method_name}"); return METHOD_BZIP2, {"level": 9} 
    elif predicted_method_name.startswith(METHOD_LZMA + "_P"):
        try: preset = int(predicted_method_name.split("_P")[1]); return METHOD_LZMA, {"preset": preset}
        except: log_callback(f"Erreur parsing LZMA: {predicted_method_name}"); return METHOD_LZMA, {"preset": 6}
    elif predicted_method_name.startswith(METHOD_ZSTD + "_L"):
        if not ZSTD_READY_ORCH: 
            log_callback(f"ZSTD prédit mais non disponible. Fallback DEFLATE L6.")
            return METHOD_DEFLATE, {"level": 6} 
        try: level = int(predicted_method_name.split("_L")[1]); return METHOD_ZSTD, {"level": level} 
        except: log_callback(f"Erreur parsing ZSTD: {predicted_method_name}"); return METHOD_ZSTD, {"level": 3} 
    elif predicted_method_name.startswith(METHOD_BROTLI + "_L"):
        if not BROTLI_READY_ORCH: 
            log_callback(f"Brotli prédit mais non disponible. Fallback DEFLATE L6.")
            return METHOD_DEFLATE, {"level": 6}
        try: 
            quality = int(predicted_method_name.split("_L")[1]) 
            return METHOD_BROTLI, {"quality": quality} 
        except: 
            log_callback(f"Erreur parsing qualité Brotli: {predicted_method_name}. Fallback BROTLI Q6.")
            return METHOD_BROTLI, {"quality": 6}
    
    # SUPPRESSION du bloc elif pour MOTEUR_AE_CIFAR10_COLOR
    # elif predicted_method_name == "MOTEUR_AE_CIFAR10_COLOR":
    #   ... (logique AE supprimée) ...

    else: # Si la méthode prédite n'est pas reconnue (ou était l'AE, maintenant supprimé)
        log_callback(f"Méthode prédite '{predicted_method_name}' non gérée ou AE (supprimé). Fallback DEFLATE L6.")
        return METHOD_DEFLATE, {"level": 6}

# Fin de aicompress/orchestrator.py