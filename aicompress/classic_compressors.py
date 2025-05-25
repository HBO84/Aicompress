# aicompress/classic_compressors.py (Version du 21 Mai - avec toutes les constantes)

import zlib
import bz2
import lzma
import os 

# --- Import conditionnel pour Zstandard ---
try:
    import zstandard as zstd
    ZSTD_AVAILABLE = True
except ImportError:
    ZSTD_AVAILABLE = False

# --- Import conditionnel pour Brotli ---
try:
    import brotli
    BROTLI_AVAILABLE = True
except ImportError:
    BROTLI_AVAILABLE = False

# --- Fonction de Log par Défaut pour ce Module ---
def _default_log_compressors(message):
    # print(f"[CLASSIC_COMPRESSORS_LOG] {message}") # Décommentez pour déboguer ce module
    pass

# --- Constantes pour les Noms de Méthodes de Base ---
# Ces constantes DOIVENT être définies ici
METHOD_STORED = "STORED"
METHOD_DEFLATE = "DEFLATE" 
METHOD_BZIP2 = "BZIP2"     
METHOD_LZMA = "LZMA"       
METHOD_ZSTD = "ZSTD"       
METHOD_BROTLI = "BROTLI"   

# --- Constantes pour les Noms de Méthodes Spécifiques (utilisées par l'IA et create_decision_dataset) ---
METHOD_DEFLATE_L1 = "DEFLATE_L1"; METHOD_DEFLATE_L6 = "DEFLATE_L6"; METHOD_DEFLATE_L9 = "DEFLATE_L9"
METHOD_BZIP2_L9 = "BZIP2_L9"
METHOD_LZMA_P0 = "LZMA_P0"; METHOD_LZMA_P6 = "LZMA_P6"; METHOD_LZMA_P9 = "LZMA_P9"
METHOD_ZSTD_L1 = "ZSTD_L1"; METHOD_ZSTD_L3 = "ZSTD_L3"; METHOD_ZSTD_L9 = "ZSTD_L9"; METHOD_ZSTD_L15 = "ZSTD_L15"
METHOD_BROTLI_L1 = "BROTLI_L1"; METHOD_BROTLI_L6 = "BROTLI_L6"; METHOD_BROTLI_L11 = "BROTLI_L11"


# --- Fonctions de Compression ---
def stored_compress(data_bytes: bytes, params: dict = None, log_callback=_default_log_compressors) -> bytes | None:
    log_callback(f"ClassicCompress: STORED appliqué (taille: {len(data_bytes)})")
    return data_bytes

def deflate_compress(data_bytes: bytes, params: dict = None, log_callback=_default_log_compressors) -> bytes | None:
    level = params.get("level", 6) if params else 6; valid_level = max(0, min(9, level)) if level != -1 else -1 
    try:
        compressed_data = zlib.compress(data_bytes, level=valid_level)
        return compressed_data
    except Exception as e: log_callback(f"[CLASSIC_COMPRESSORS_ERROR] DEFLATE L{level}: {e}"); return None

def bzip2_compress(data_bytes: bytes, params: dict = None, log_callback=_default_log_compressors) -> bytes | None:
    level = params.get("level", 9) if params else 9; valid_level = max(1, min(9, level))
    try:
        compressed_data = bz2.compress(data_bytes, compresslevel=valid_level)
        return compressed_data
    except Exception as e: log_callback(f"[CLASSIC_COMPRESSORS_ERROR] BZIP2 L{valid_level}: {e}"); return None

def lzma_compress(data_bytes: bytes, params: dict = None, log_callback=_default_log_compressors) -> bytes | None:
    preset = params.get("preset", 6) if params else 6; valid_preset = max(0, min(9, preset))
    format_to_use = getattr(lzma, params.get("format", "FORMAT_XZ"), lzma.FORMAT_XZ) if params else lzma.FORMAT_XZ
    try:
        compressed_data = lzma.compress(data_bytes, format=format_to_use, preset=valid_preset)
        return compressed_data
    except Exception as e: log_callback(f"[CLASSIC_COMPRESSORS_ERROR] LZMA P{valid_preset}: {e}"); return None

def zstd_compress(data_bytes: bytes, params: dict = None, log_callback=_default_log_compressors) -> bytes | None:
    if not ZSTD_AVAILABLE: log_callback("[CLASSIC_COMPRESSORS_ERROR] ZSTD non disponible."); return None
    level = params.get("level", 3) if params else 3; valid_level = max(1, min(22, level)) 
    threads = params.get("threads", -1) 
    try:
        cctx = zstd.ZstdCompressor(level=valid_level, threads=threads) 
        compressed_data = cctx.compress(data_bytes)
        return compressed_data
    except Exception as e: log_callback(f"[CLASSIC_COMPRESSORS_ERROR] ZSTD L{valid_level}: {e}"); return None

def brotli_compress(data_bytes: bytes, params: dict = None, log_callback=_default_log_compressors) -> bytes | None:
    if not BROTLI_AVAILABLE: log_callback("[CLASSIC_COMPRESSORS_ERROR] Brotli non disponible."); return None
    quality = params.get("quality", 6) if params else 6; valid_quality = max(0, min(11, quality))
    mode_str = params.get("mode", "generic").upper()
    mode_map = {"GENERIC": brotli.MODE_GENERIC, "TEXT": brotli.MODE_TEXT, "FONT": brotli.MODE_FONT}
    mode_to_use = mode_map.get(mode_str, brotli.MODE_GENERIC)
    try:
        compressed_data = brotli.compress(data_bytes, quality=valid_quality, mode=mode_to_use)
        return compressed_data
    except Exception as e: log_callback(f"[CLASSIC_COMPRESSORS_ERROR] Brotli Q{valid_quality}: {e}"); return None

# --- Fonctions de Décompression ---
def stored_decompress(c_bytes: bytes, p=None, lc=_default_log_compressors) -> bytes | None: return c_bytes
def deflate_decompress(c_bytes: bytes, p=None, lc=_default_log_compressors) -> bytes | None: 
    try: return zlib.decompress(c_bytes)
    except Exception as e: lc(f"[CLASSIC_COMPRESSORS_ERROR] Décompression DEFLATE: {e}"); return None
def bzip2_decompress(c_bytes: bytes, p=None, lc=_default_log_compressors) -> bytes | None:
    try: return bz2.decompress(c_bytes)
    except Exception as e: lc(f"[CLASSIC_COMPRESSORS_ERROR] Décompression BZIP2: {e}"); return None
def lzma_decompress(c_bytes: bytes, p=None, lc=_default_log_compressors) -> bytes | None:
    fmt = getattr(lzma, p.get("format", "FORMAT_XZ"), lzma.FORMAT_XZ) if p else lzma.FORMAT_XZ
    try: return lzma.decompress(c_bytes,format=fmt)
    except Exception as e: lc(f"[CLASSIC_COMPRESSORS_ERROR] Décompression LZMA: {e}"); return None
def zstd_decompress(c_bytes: bytes, p=None, lc=_default_log_compressors) -> bytes | None:
    if not ZSTD_AVAILABLE: lc("[CLASSIC_COMPRESSORS_ERROR] ZSTD non dispo décomp."); return None
    try: dctx=zstd.ZstdDecompressor(); return dctx.decompress(c_bytes)
    except Exception as e: lc(f"[CLASSIC_COMPRESSORS_ERROR] Décompression ZSTD: {e}"); return None
def brotli_decompress(c_bytes: bytes, p=None, lc=_default_log_compressors) -> bytes | None:
    if not BROTLI_AVAILABLE: lc("[CLASSIC_COMPRESSORS_ERROR] Brotli non dispo décomp."); return None
    try: return brotli.decompress(c_bytes)
    except Exception as e: lc(f"[CLASSIC_COMPRESSORS_ERROR] Décompression Brotli: {e}"); return None

CLASSIC_COMPRESSORS_READY = True 
_default_log_compressors(f"[CLASSIC_COMPRESSORS] Module initialisé. ZSTD: {ZSTD_AVAILABLE}, Brotli: {BROTLI_AVAILABLE}")
if not ZSTD_AVAILABLE: _default_log_compressors("[CLASSIC_COMPRESSORS] AVERT: Zstandard n'est pas fonctionnel.")
if not BROTLI_AVAILABLE: _default_log_compressors("[CLASSIC_COMPRESSORS] AVERT: Brotli n'est pas fonctionnel.")

# Fin de aicompress/classic_compressors.py