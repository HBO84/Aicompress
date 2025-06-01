# aicompress/__init__.py

def _init_log(message): print(f"[AICOMPRESS_INIT] {message}")
_init_log("Initialisation du package AICompress...")

try:
    from .aic_file_handler import ( # Doit importer extract_archive d'ici
        compress_to_aic, 
        decompress_aic,
        extract_archive # <--- S'ASSURER QUE CETTE LIGNE EST LÀ
    )
    _init_log("aic_file_handler.py chargé (fonctions principales).")
except ImportError as e:
    _init_log(f"ERREUR: Impossible d'importer depuis aic_file_handler: {e}")
    def compress_to_aic(*a, **k): _init_log("ERREUR: compress_to_aic indisponible."); return False, "Handler Error"
    def decompress_aic(*a, **k): _init_log("ERREUR: decompress_aic indisponible."); return False, "Handler Error"
    def extract_archive(*a, **k): _init_log("ERREUR: extract_archive indisponible."); return False, "Handler Error"

try:
    from .external_handlers import decompress_rar # decompress_rar est spécifique à external_handlers
    _init_log("external_handlers.py chargé (decompress_rar).")
except ImportError as e:
    _init_log(f"ERREUR: Impossible d'importer decompress_rar: {e}")
    def decompress_rar(*a, **k): _init_log("ERREUR: decompress_rar indisponible."); return False, "Handler Error"

from .aic_file_handler import (
    AI_ANALYZER_AVAILABLE, RARFILE_AVAILABLE, CRYPTOGRAPHY_AVAILABLE,
    ZSTD_AVAILABLE, CLASSIC_COMPRESSORS_LOADED, AE_ENGINE_LOADED,
    ORCHESTRATOR_LOADED_SUCCESSFULLY, DEFAULT_AIC_EXTENSION, PIL_AVAILABLE, KERAS_AVAILABLE
)
from .external_handlers import PY7ZR_AVAILABLE_EXT as PY7ZR_SUPPORT_AVAILABLE

OTA_AVAILABLE = False
def _fb_check_updates(lc): lc("[AICOMPRESS_INIT] OTA non dispo."); return {}
def _fb_download_model(n,i,lc): lc("[AICOMPRESS_INIT] OTA non dispo."); return False
check_for_model_updates = _fb_check_updates
download_and_install_model = _fb_download_model
try:
    from .ota_updater import check_for_model_updates as _real_check_ota, \
                               download_and_install_model as _real_download_ota, \
                               OTA_MODULE_AVAILABLE as OTA_FLAG_FROM_MODULE
    if OTA_FLAG_FROM_MODULE:
        check_for_model_updates = _real_check_ota; download_and_install_model = _real_download_ota
        OTA_AVAILABLE = True; _init_log("Module ota_updater chargé et disponible.")
    else: _init_log("Module ota_updater chargé mais marqué non disponible.")
except ImportError: _init_log("AVERTISSEMENT: Module ota_updater.py non trouvé.")

__all__ = [
    'compress_to_aic', 'decompress_aic', 'extract_archive', 'decompress_rar',
    'AI_ANALYZER_AVAILABLE', 'RARFILE_AVAILABLE', 'CRYPTOGRAPHY_AVAILABLE',
    'ZSTD_AVAILABLE', 'CLASSIC_COMPRESSORS_LOADED', 'AE_ENGINE_LOADED',
    'ORCHESTRATOR_LOADED_SUCCESSFULLY', 'PY7ZR_SUPPORT_AVAILABLE',
    'PIL_AVAILABLE', 'KERAS_AVAILABLE', 'OTA_AVAILABLE',
    'check_for_model_updates', 'download_and_install_model', 'DEFAULT_AIC_EXTENSION'
]
_init_log("Initialisation du package AICompress terminée.")