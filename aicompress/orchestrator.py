# aicompress/orchestrator.py

import os
import zipfile 
import joblib
# Pandas est importé localement dans get_compression_settings

def _default_log_orchestrator(message):
    print(message)

AI_ANALYZER_READY_ORCH = False 
get_file_features_orch = lambda fp, lc: {"type":"analyzer_unavailable_in_orch","error":True}
try:
    from .ai_analyzer import get_file_features as _real_gff_orch, AI_ANALYZER_AVAILABLE as _analyzer_flag_orch
    if _analyzer_flag_orch: get_file_features_orch = _real_gff_orch; AI_ANALYZER_READY_ORCH = True
    _default_log_orchestrator(f"[ORCHESTRATOR] Dépendance ai_analyzer: {'Prêt' if AI_ANALYZER_READY_ORCH else 'Non Prêt'}")
except ImportError: _default_log_orchestrator("[ORCHESTRATOR] AVERT: ai_analyzer.py non trouvé.")

CLASSIC_COMPRESSORS_READY_ORCH = False; ZSTD_READY_ORCH = False; BROTLI_READY_ORCH = False
METHOD_S,METHOD_D,METHOD_B,METHOD_L,METHOD_Z,METHOD_BR = "STORED","DEFLATE","BZIP2","LZMA","ZSTD","BROTLI" 
try:
    from .classic_compressors import (
        METHOD_STORED as _M_S_cl, METHOD_DEFLATE as _M_D_cl, 
        METHOD_BZIP2 as _M_B_cl, METHOD_LZMA as _M_L_cl, 
        METHOD_ZSTD as _M_Z_cl, METHOD_BROTLI_L1, METHOD_BROTLI_L6, METHOD_BROTLI_L11, # Juste pour info
        ZSTD_AVAILABLE as ZSTD_FLAG_cl,
        BROTLI_AVAILABLE as BROTLI_FLAG_cl, 
        CLASSIC_COMPRESSORS_READY as CC_READY_FLAG_cl,
        METHOD_BROTLI as _M_BR_cl # Importer la constante de base METHOD_BROTLI
    )
    if CC_READY_FLAG_cl:
        METHOD_STORED,METHOD_DEFLATE,METHOD_BZIP2,METHOD_LZMA,METHOD_ZSTD=_M_S_cl,_M_D_cl,_M_B_cl,_M_L_cl,_M_Z_cl
        METHOD_BROTLI = _M_BR_cl # Utiliser la constante importée
        ZSTD_READY_ORCH = ZSTD_FLAG_cl; BROTLI_READY_ORCH = BROTLI_FLAG_cl
        CLASSIC_COMPRESSORS_READY_ORCH = True
    _default_log_orchestrator(f"[ORCHESTRATOR] Dépendance classic_compressors: {'Prêt' if CLASSIC_COMPRESSORS_READY_ORCH else 'Non Prêt'}")
    if CLASSIC_COMPRESSORS_READY_ORCH:
        _default_log_orchestrator(f"[ORCHESTRATOR]   ZSTD: {'OK' if ZSTD_READY_ORCH else 'X'}, Brotli: {'OK' if BROTLI_READY_ORCH else 'X'}")
except ImportError: _default_log_orchestrator("[ORCHESTRATOR] AVERT: classic_compressors.py non trouvé.")

AE_ENGINE_READY_ORCH = False; PIL_AVAILABLE_FOR_AE_ORCH = False; KERAS_AVAILABLE_FOR_AE_ORCH = False
ensure_ae_models_loaded_orch = lambda lc: False
try:
    from .ae_engine import (ensure_cifar10_color_ae_models_loaded as _ensure_ae_orch,
                            PIL_AVAILABLE_AE as _pil_ae_orch, KERAS_AVAILABLE_AE as _keras_ae_orch,
                            AE_ENGINE_LOADED as _ae_loaded_flag_orch)
    if _ae_loaded_flag_orch:
        ensure_ae_models_loaded_orch = _ensure_ae_orch; PIL_AVAILABLE_FOR_AE_ORCH = _pil_ae_orch
        KERAS_AVAILABLE_FOR_AE_ORCH = _keras_ae_orch; AE_ENGINE_READY_ORCH = True
    _default_log_orchestrator(f"[ORCHESTRATOR] Dépendance ae_engine: {'Prêt' if AE_ENGINE_READY_ORCH else 'Non Prêt'}")
except ImportError: _default_log_orchestrator("[ORCHESTRATOR] AVERT: ae_engine.py non trouvé.")

_MODULE_DIR_ORCH = os.path.dirname(__file__)
ORCHESTRATOR_MODEL_PATH_ORCH = os.path.join(_MODULE_DIR_ORCH, "compression_orchestrator_model.joblib")
ORCHESTRATOR_LABEL_ENCODER_PATH_ORCH = os.path.join(_MODULE_DIR_ORCH, "orchestrator_label_encoder.joblib")
_orchestrator_pipeline_internal = None; _orchestrator_label_encoder_internal = None
ORCHESTRATOR_IS_READY = False 

def load_orchestrator_model(log_callback=_default_log_orchestrator):
    global _orchestrator_pipeline_internal, _orchestrator_label_encoder_internal, ORCHESTRATOR_IS_READY
    if ORCHESTRATOR_IS_READY: return True
    if not (os.path.exists(ORCHESTRATOR_MODEL_PATH_ORCH) and os.path.exists(ORCHESTRATOR_LABEL_ENCODER_PATH_ORCH)):
        log_callback(f"[ORCHESTRATOR] ERREUR: Modèle ou LE non trouvé."); ORCHESTRATOR_IS_READY = False; return False
    try:
        import joblib; _orchestrator_pipeline_internal = joblib.load(ORCHESTRATOR_MODEL_PATH_ORCH)
        _orchestrator_label_encoder_internal = joblib.load(ORCHESTRATOR_LABEL_ENCODER_PATH_ORCH)
        log_callback("[ORCHESTRATOR] Modèle Orchestrateur et LE chargés."); ORCHESTRATOR_IS_READY = True; return True
    except Exception as e: log_callback(f"[ORCHESTRATOR] ERREUR chargement modèle: {e}"); ORCHESTRATOR_IS_READY = False; return False
if _initial_orchestrator_load_status_unused_ := load_orchestrator_model(_default_log_orchestrator): pass 

def get_compression_settings(file_path: str, analysis_result_str_ignored: str, log_callback=_default_log_orchestrator) -> tuple:
    log_callback(f"[ORCHESTRATOR] Début get_compression_settings pour: '{os.path.basename(file_path)}'")
    if not ORCHESTRATOR_IS_READY: log_callback("[ORCHESTRATOR] ERREUR: Modèle non chargé. Fallback."); return METHOD_DEFLATE, {"level": 6} 
    if not AI_ANALYZER_READY_ORCH: log_callback("[ORCHESTRATOR] AI Analyzer non dispo. Fallback."); return METHOD_DEFLATE, {"level": 6}
    features = get_file_features_orch(file_path, log_callback=log_callback)
    if features.get("error"): log_callback(f"[ORCHESTRATOR] Erreur features. Fallback."); return METHOD_DEFLATE, {"level": 6}
    try:
        import pandas as pd 
        input_df = pd.DataFrame([{"file_type_analysis":features["type"], "original_size_bytes":features["size_bytes"], "entropy_normalized":features["entropy_normalized"], "quick_comp_ratio":features["quick_comp_ratio"]}])
    except ImportError: log_callback("[ORCHESTRATOR] Pandas non trouvé. Fallback."); return METHOD_DEFLATE, {"level": 6}
    except Exception as e_df: log_callback(f"[ORCHESTRATOR] Erreur DataFrame: {e_df}. Fallback."); return METHOD_DEFLATE, {"level": 6}
    try:
        predicted_method_encoded = _orchestrator_pipeline_internal.predict(input_df)
        predicted_method_name = _orchestrator_label_encoder_internal.inverse_transform(predicted_method_encoded)[0]
        log_callback(f"[ORCHESTRATOR] Méthode prédite par IA: {predicted_method_name}")
    except Exception as e_predict: log_callback(f"[ORCHESTRATOR] Erreur prédiction: {e_predict}. Fallback."); return METHOD_DEFLATE, {"level": 6}
    
    if predicted_method_name == METHOD_STORED: return METHOD_STORED, {}
    elif predicted_method_name.startswith(METHOD_DEFLATE + "_L"):
        try: level = int(predicted_method_name.split("_L")[1]); return METHOD_DEFLATE, {"level": level}
        except: return METHOD_DEFLATE, {"level": 6} 
    elif predicted_method_name.startswith(METHOD_BZIP2 + "_L"):
        try: level = int(predicted_method_name.split("_L")[1]); return METHOD_BZIP2, {"level": level} 
        except: return METHOD_BZIP2, {"level": 9} 
    elif predicted_method_name.startswith(METHOD_LZMA + "_P"):
        try: preset = int(predicted_method_name.split("_P")[1]); return METHOD_LZMA, {"preset": preset}
        except: return METHOD_LZMA, {"preset": 6}
    elif predicted_method_name.startswith(METHOD_ZSTD + "_L"):
        if not ZSTD_READY_ORCH: return METHOD_DEFLATE, {"level": 6} 
        try: level = int(predicted_method_name.split("_L")[1]); return METHOD_ZSTD, {"level": level} 
        except: return METHOD_ZSTD, {"level": 3} 
    # --- CAS POUR BROTLI ---
    elif predicted_method_name.startswith(METHOD_BROTLI + "_L"): # Ex: "BROTLI_L6"
        if not BROTLI_READY_ORCH: 
            log_callback(f"[ORCHESTRATOR] Brotli prédit mais non disponible. Fallback DEFLATE L6.")
            return METHOD_DEFLATE, {"level": 6}
        try: 
            quality = int(predicted_method_name.split("_L")[1]) 
            return METHOD_BROTLI, {"quality": quality} 
        except: 
            log_callback(f"[ORCHESTRATOR] Erreur parsing qualité Brotli: {predicted_method_name}. Fallback BROTLI Q6.")
            return METHOD_BROTLI, {"quality": 6} # Qualité 6 par défaut pour Brotli
    # --- FIN CAS BROTLI ---
    elif predicted_method_name == "MOTEUR_AE_CIFAR10_COLOR":
        if not AE_ENGINE_READY_ORCH: log_callback(f"[ORCHESTRATOR] IA prédit AE, mais moteur AE non op. Fallback."); return METHOD_DEFLATE, {"level": 1}
        if PIL_AVAILABLE_FOR_AE_ORCH and KERAS_AVAILABLE_FOR_AE_ORCH : 
            try: 
                from PIL import Image as PILImage 
                with PILImage.open(file_path) as img: width, height = img.size
                MAX_AE_INPUT_DIM_ORCH = 256 
                if width <= MAX_AE_INPUT_DIM_ORCH and height <= MAX_AE_INPUT_DIM_ORCH: return "MOTEUR_AE_CIFAR10_COLOR", {}
                else: return METHOD_DEFLATE, {"level": 1}
            except Exception as e_img: log_callback(f"[ORCHESTRATOR] Erreur vérif image AE: {e_img}. Fallback."); return METHOD_DEFLATE, {"level": 1}
        else: return METHOD_DEFLATE, {"level": 1}
    else: return METHOD_DEFLATE, {"level": 6}