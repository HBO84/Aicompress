# aicompress/__init__.py

def _init_log(message):
    print(f"[AICOMPRESS_INIT] {message}")
_init_log("Initialisation du package AICompress...")

# Importer les fonctions publiques principales depuis les modules internes
from .aic_file_handler import compress_to_aic, decompress_aic, extract_archive
from .external_handlers import decompress_rar, decompress_7z

# Importer les flags de disponibilité des fonctionnalités pour un accès facile
from .aic_file_handler import (
    AI_ANALYZER_AVAILABLE,
    CRYPTOGRAPHY_AVAILABLE,
    ZSTD_AVAILABLE,
    CLASSIC_COMPRESSORS_LOADED,
    ORCHESTRATOR_LOADED_SUCCESSFULLY,
    PIL_AVAILABLE,
    DEFAULT_AIC_EXTENSION,
    BROTLI_AVAILABLE
)
from .external_handlers import RARFILE_AVAILABLE, PY7ZR_AVAILABLE as PY7ZR_SUPPORT_AVAILABLE

# Importer les fonctions OTA si disponibles
try:
    from .ota_updater import check_for_model_updates, download_and_install_model, OTA_MODULE_AVAILABLE as OTA_AVAILABLE
except ImportError:
    OTA_AVAILABLE = False
    def check_for_model_updates(lc): lc("[AICOMPRESS_INIT] OTA non dispo."); return {}
    def download_and_install_model(n,i,lc): lc("[AICOMPRESS_INIT] OTA non dispo."); return False

# Définir __all__ pour contrôler 'from aicompress import *'
__all__ = [
    'compress_to_aic', 'decompress_aic', 'extract_archive', 'decompress_rar', 'decompress_7z',
    'AI_ANALYZER_AVAILABLE', 'RARFILE_AVAILABLE', 'CRYPTOGRAPHY_AVAILABLE',
    'ZSTD_AVAILABLE', 'CLASSIC_COMPRESSORS_LOADED', 'BROTLI_AVAILABLE',
    'ORCHESTRATOR_LOADED_SUCCESSFULLY', 'PY7ZR_SUPPORT_AVAILABLE',
    'PIL_AVAILABLE', 'OTA_AVAILABLE',
    'check_for_model_updates', 'download_and_install_model',
    'DEFAULT_AIC_EXTENSION'
]
_init_log("Initialisation du package AICompress terminée.")