# aicompress/__init__.py

# Importer les fonctions principales à exposer directement par le package
from .aic_file_handler import compress_to_aic, decompress_aic # Fonctions spécifiques au format .aic
from .external_handlers import extract_archive, decompress_rar # Point d'entrée général pour l'extraction

# Importer les flags de disponibilité des fonctionnalités pour un accès facile
from .aic_file_handler import (
    AI_ANALYZER_AVAILABLE,
    RARFILE_AVAILABLE,
    CRYPTOGRAPHY_AVAILABLE,
    ZSTD_AVAILABLE, # Disponibilité de la lib zstd via classic_compressors
    CLASSIC_COMPRESSORS_LOADED,
    AE_ENGINE_LOADED,
    ORCHESTRATOR_LOADED_SUCCESSFULLY,
    PIL_AVAILABLE, # Flag global Pillow (principalement pour AE)
    KERAS_AVAILABLE # Flag global Keras (principalement pour AE)
)
# Importer le flag de disponibilité de 7zip depuis external_handlers
from .external_handlers import PY7ZR_AVAILABLE_EXT as PY7ZR_SUPPORT_AVAILABLE

# Importer les fonctions OTA si disponibles
OTA_INIT_SUCCESS = False
try:
    from .ota_updater import check_for_model_updates, download_and_install_model
    OTA_AVAILABLE = True # Le module ota_updater lui-même a pu être importé
    OTA_INIT_SUCCESS = True
except ImportError:
    OTA_AVAILABLE = False
    # Définir des fonctions factices si ota_updater n'est pas là
    def check_for_model_updates(log_callback): 
        if callable(log_callback): log_callback("[AICOMPRESS_INIT] OTA non disponible.")
        return {}
    def download_and_install_model(name, info, log_callback):
        if callable(log_callback): log_callback("[AICOMPRESS_INIT] OTA non disponible.")
        return False

# Importer des constantes utiles
from .aic_file_handler import DEFAULT_AIC_EXTENSION

# Optionnel: Définir __all__ pour spécifier ce qui est importé par "from aicompress import *"
# C'est une bonne pratique.
__all__ = [
    'compress_to_aic', 
    'decompress_aic', # Exposer aussi celle-ci si on veut un accès direct au décompresseur .aic
    'extract_archive', 
    'decompress_rar',
    'AI_ANALYZER_AVAILABLE',
    'RARFILE_AVAILABLE',
    'CRYPTOGRAPHY_AVAILABLE',
    'ZSTD_AVAILABLE',
    'CLASSIC_COMPRESSORS_LOADED',
    'AE_ENGINE_LOADED',
    'ORCHESTRATOR_LOADED_SUCCESSFULLY',
    'PY7ZR_SUPPORT_AVAILABLE',
    'PIL_AVAILABLE',
    'KERAS_AVAILABLE',
    'OTA_AVAILABLE',
    'check_for_model_updates',
    'download_and_install_model',
    'DEFAULT_AIC_EXTENSION'
]

_default_log_init = lambda m: print(f"[AICOMPRESS_INIT] {m}")
if OTA_INIT_SUCCESS:
    _default_log_init("Package AICompress initialisé. Fonctions principales et flags de disponibilité exposés.")
else:
    _default_log_init("Package AICompress initialisé (Module OTA non trouvé). Fonctions principales et flags de disponibilité exposés.")