# aicompress/classic_compressors.py

import zlib
import bz2
import lzma
import os # Juste pour _default_log_compressors si besoin

try:
    import zstandard as zstd
    ZSTD_AVAILABLE = True
except ImportError:
    ZSTD_AVAILABLE = False

def _default_log_compressors(message):
    # print(f"[CLASSIC_COMPRESSORS] {message}") 
    pass 

# --- Constantes pour les Noms de Méthodes (exportées et utilisées par core.py) ---
METHOD_STORED = "STORED"
METHOD_DEFLATE = "DEFLATE" # Nom de base pour la méthode DEFLATE
METHOD_BZIP2 = "BZIP2"     # Nom de base pour la méthode BZIP2
METHOD_LZMA = "LZMA"       # Nom de base pour la méthode LZMA
METHOD_ZSTD = "ZSTD"       # Nom de base pour la méthode ZSTD

# Les noms de méthodes spécifiques avec niveaux/presets (prédits par l'IA)
# seront parsés par get_compression_settings dans core.py pour extraire
# la méthode de base (ex: METHOD_DEFLATE) et le paramètre (ex: level 6).

# --- Fonctions de Compression ---
def stored_compress(data_bytes: bytes, params: dict = None, log_callback=_default_log_compressors) -> bytes | None:
    log_callback(f"ClassicCompress: STORED appliqué (taille: {len(data_bytes)})")
    return data_bytes

def deflate_compress(data_bytes: bytes, params: dict = None, log_callback=_default_log_compressors) -> bytes | None:
    level = params.get("level", 6) if params else 6
    valid_level = max(0, min(9, level)) if level != -1 else -1 
    try:
        compressed_data = zlib.compress(data_bytes, level=valid_level)
        log_callback(f"ClassicCompress: DEFLATE L{valid_level} appliqué. Orig: {len(data_bytes)}, Comp: {len(compressed_data)}")
        return compressed_data
    except Exception as e:
        log_callback(f"[CLASSIC_COMPRESSORS] ERREUR DEFLATE L{valid_level}: {e}") # valid_level ici
        return None

def bzip2_compress(data_bytes: bytes, params: dict = None, log_callback=_default_log_compressors) -> bytes | None:
    level = params.get("level", 9) if params else 9
    valid_level = max(1, min(9, level))
    try:
        compressed_data = bz2.compress(data_bytes, compresslevel=valid_level)
        log_callback(f"ClassicCompress: BZIP2 L{valid_level} appliqué. Orig: {len(data_bytes)}, Comp: {len(compressed_data)}")
        return compressed_data
    except Exception as e:
        log_callback(f"[CLASSIC_COMPRESSORS] ERREUR BZIP2 L{valid_level}: {e}")
        return None

def lzma_compress(data_bytes: bytes, params: dict = None, log_callback=_default_log_compressors) -> bytes | None:
    preset = params.get("preset", 6) if params else 6
    format_to_use = lzma.FORMAT_XZ # Cohérent avec create_decision_dataset et décompression
    valid_preset = max(0, min(9, preset))
    try:
        compressed_data = lzma.compress(data_bytes, format=format_to_use, preset=valid_preset)
        log_callback(f"ClassicCompress: LZMA P{valid_preset} (XZ) appliqué. Orig: {len(data_bytes)}, Comp: {len(compressed_data)}")
        return compressed_data
    except Exception as e:
        log_callback(f"[CLASSIC_COMPRESSORS] ERREUR LZMA P{valid_preset}: {e}")
        return None

def zstd_compress(data_bytes: bytes, params: dict = None, log_callback=_default_log_compressors) -> bytes | None:
    if not ZSTD_AVAILABLE:
        log_callback("[CLASSIC_COMPRESSORS] ZSTD non disponible pour compression.")
        return None
    level = params.get("level", 3) if params else 3
    valid_level = max(1, min(22, level)) 
    try:
        cctx = zstd.ZstdCompressor(level=valid_level, threads=-1) 
        compressed_data = cctx.compress(data_bytes)
        log_callback(f"ClassicCompress: ZSTD L{valid_level} (multi-thread) appliqué. Orig: {len(data_bytes)}, Comp: {len(compressed_data)}")
        return compressed_data
    except Exception as e:
        log_callback(f"[CLASSIC_COMPRESSORS] ERREUR ZSTD L{valid_level}: {e}")
        return None

# --- Fonctions de Décompression ---
def stored_decompress(compressed_data_bytes: bytes, params: dict = None, log_callback=_default_log_compressors) -> bytes | None:
    log_callback(f"ClassicCompress: STORED décompression (taille: {len(compressed_data_bytes)}).")
    return compressed_data_bytes

def deflate_decompress(compressed_data_bytes: bytes, params: dict = None, log_callback=_default_log_compressors) -> bytes | None:
    try:
        decompressed_data = zlib.decompress(compressed_data_bytes)
        log_callback(f"ClassicCompress: DEFLATE décompressé. Comp: {len(compressed_data_bytes)}, Decomp: {len(decompressed_data)}")
        return decompressed_data
    except Exception as e:
        log_callback(f"[CLASSIC_COMPRESSORS] ERREUR Décompression DEFLATE: {e}")
        return None

def bzip2_decompress(compressed_data_bytes: bytes, params: dict = None, log_callback=_default_log_compressors) -> bytes | None:
    try:
        decompressed_data = bz2.decompress(compressed_data_bytes)
        log_callback(f"ClassicCompress: BZIP2 décompressé. Comp: {len(compressed_data_bytes)}, Decomp: {len(decompressed_data)}")
        return decompressed_data
    except Exception as e:
        log_callback(f"[CLASSIC_COMPRESSORS] ERREUR Décompression BZIP2: {e}")
        return None

def lzma_decompress(compressed_data_bytes: bytes, params: dict = None, log_callback=_default_log_compressors) -> bytes | None:
    format_to_use = lzma.FORMAT_XZ 
    try:
        decompressed_data = lzma.decompress(compressed_data_bytes, format=format_to_use)
        log_callback(f"ClassicCompress: LZMA (format {format_to_use}) décompressé. Comp: {len(compressed_data_bytes)}, Decomp: {len(decompressed_data)}")
        return decompressed_data
    except Exception as e:
        log_callback(f"[CLASSIC_COMPRESSORS] ERREUR Décompression LZMA: {e}")
        return None

def zstd_decompress(compressed_data_bytes: bytes, params: dict = None, log_callback=_default_log_compressors) -> bytes | None:
    if not ZSTD_AVAILABLE:
        log_callback("[CLASSIC_COMPRESSORS] ZSTD non disponible pour décompression.")
        return None
    try:
        dctx = zstd.ZstdDecompressor()
        decompressed_data = dctx.decompress(compressed_data_bytes)
        log_callback(f"ClassicCompress: ZSTD décompressé. Comp: {len(compressed_data_bytes)}, Decomp: {len(decompressed_data)}")
        return decompressed_data
    except Exception as e:
        log_callback(f"[CLASSIC_COMPRESSORS] ERREUR Décompression ZSTD: {e}")
        return None
    
    # À la fin de aicompress/classic_compressors.py
# (après toutes les définitions de fonctions et l'import de zstd)

# Ce module est considéré comme "prêt" s'il a pu être importé
# et si ses imports de base (zlib, bz2, lzma) ont réussi, ce qui est implicite.
# Le flag ZSTD_AVAILABLE est spécifique à zstd.
CLASSIC_COMPRESSORS_READY = True 
_default_log_compressors("[CLASSIC_COMPRESSORS] Module marqué comme prêt.")
if not ZSTD_AVAILABLE:
    _default_log_compressors("[CLASSIC_COMPRESSORS] AVERTISSEMENT: Zstandard n'est pas disponible dans ce module.")