# aicompress/__init__.py (Version sans AE)

def _init_log(message):
    print(f"[AICOMPRESS_INIT] {message}")

_init_log("Initialisation du package AICompress...")

try:
    from .aic_file_handler import (
        compress_to_aic, 
        decompress_aic,
        extract_archive
    )
    _init_log("aic_file_handler.py chargé (fonctions principales).")
except ImportError as e:
    _init_log(f"ERREUR: Impossible d'importer depuis aic_file_handler: {e}")
    def _fb_log(msg="Erreur"):_init_log(msg)
    def compress_to_aic(*a, **k): _fb_log("ERREUR: compress_to_aic non disponible."); return False, "Handler Error"
    def decompress_aic(*a, **k): _fb_log("ERREUR: decompress_aic non disponible."); return False, "Handler Error"
    def extract_archive(*a, **k): _fb_log("ERREUR: extract_archive non disponible."); return False, "Handler Error"

try:
    from .external_handlers import decompress_rar
    _init_log("external_handlers.py chargé (fonction decompress_rar).")
except ImportError as e:
    _init_log(f"ERREUR: Impossible d'importer decompress_rar: {e}")
    def decompress_rar(*a, **k): _init_log("ERREUR: decompress_rar non disponible."); return False, "Handler Error"

# Importer les flags de disponibilité importants
# Ceux liés à l'AE et Keras sont maintenant supprimés
try:
    from .aic_file_handler import (
        AI_ANALYZER_AVAILABLE,
        RARFILE_AVAILABLE,
        CRYPTOGRAPHY_AVAILABLE,
        ZSTD_AVAILABLE, 
        CLASSIC_COMPRESSORS_LOADED,
        # AE_ENGINE_LOADED, # SUPPRIMÉ
        ORCHESTRATOR_LOADED_SUCCESSFULLY,
        PIL_AVAILABLE, # Pillow peut rester si utile pour autre chose
        # KERAS_AVAILABLE, # SUPPRIMÉ
        DEFAULT_AIC_EXTENSION,
        BROTLI_AVAILABLE # Assurez-vous que ce flag est bien défini dans aic_file_handler.py
    )
    _init_log("Flags de disponibilité de aic_file_handler importés.")
except ImportError as e_flags: 
    _init_log(f"AVERTISSEMENT: Impossible d'importer certains flags de aic_file_handler: {e_flags}")
    # Fallbacks pour les flags
    AI_ANALYZER_AVAILABLE = False; RARFILE_AVAILABLE = False; CRYPTOGRAPHY_AVAILABLE = False
    ZSTD_AVAILABLE = False; CLASSIC_COMPRESSORS_LOADED = False; ORCHESTRATOR_LOADED_SUCCESSFULLY = False
    PIL_AVAILABLE = False; BROTLI_AVAILABLE = False
    DEFAULT_AIC_EXTENSION = ".aic"

PY7ZR_SUPPORT_AVAILABLE = False
try:
    from .external_handlers import PY7ZR_AVAILABLE_EXT
    PY7ZR_SUPPORT_AVAILABLE = PY7ZR_AVAILABLE_EXT
    if PY7ZR_SUPPORT_AVAILABLE: _init_log("Support 7-Zip (PY7ZR_AVAILABLE_EXT) importé.")
    else: _init_log("Support 7-Zip non disponible (selon external_handlers).")
except ImportError: _init_log("AVERTISSEMENT: Impossible d'importer PY7ZR_AVAILABLE_EXT.")

OTA_AVAILABLE = False
def _fb_check_updates_init(lc): lc("[AICOMPRESS_INIT] OTA non dispo."); return {}
def _fb_download_model_init(n,i,lc): lc("[AICOMPRESS_INIT] OTA non dispo."); return False
check_for_model_updates = _fb_check_updates_init
download_and_install_model = _fb_download_model_init
try:
    from .ota_updater import check_for_model_updates as _real_check_ota, \
                               download_and_install_model as _real_download_ota, \
                               OTA_MODULE_AVAILABLE
    if OTA_MODULE_AVAILABLE:
        check_for_model_updates = _real_check_ota; download_and_install_model = _real_download_ota
        OTA_AVAILABLE = True; _init_log("Module ota_updater chargé et disponible.")
    else: _init_log("Module ota_updater chargé mais marqué non disponible.")
except ImportError: _init_log("AVERTISSEMENT: Module ota_updater.py non trouvé.")

__all__ = [
    'compress_to_aic', 'decompress_aic', 'extract_archive', 'decompress_rar',
    'AI_ANALYZER_AVAILABLE', 'RARFILE_AVAILABLE', 'CRYPTOGRAPHY_AVAILABLE',
    'ZSTD_AVAILABLE', 'CLASSIC_COMPRESSORS_LOADED', 'BROTLI_AVAILABLE',
    # 'AE_ENGINE_LOADED', # SUPPRIMÉ
    'ORCHESTRATOR_LOADED_SUCCESSFULLY', 'PY7ZR_SUPPORT_AVAILABLE',
    'PIL_AVAILABLE', 
    # 'KERAS_AVAILABLE', # SUPPRIMÉ
    'OTA_AVAILABLE',
    'check_for_model_updates', 'download_and_install_model',
    'DEFAULT_AIC_EXTENSION'
]
_init_log("Initialisation du package AICompress terminée (sans AE).")