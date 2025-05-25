# create_decision_dataset.py
import os
import glob
import csv
import time
import json 
# Les imports directs (zipfile, bz2, lzma, zlib, zstandard, brotli) ne sont plus nécessaires ici
# car nous allons utiliser les wrappers de classic_compressors.py

# Importer les fonctions et flags nécessaires d'AICompress
try:
    from aicompress.ai_analyzer import get_file_features, _default_log_analyzer as log_dataset_script
    AI_ANALYZER_LOADED_FOR_DATASET = True
except ImportError:
    AI_ANALYZER_LOADED_FOR_DATASET = False
    def get_file_features(fp, log_callback=print): 
        log_callback(f"[DATASET_FB] ai_analyzer non chargé. Features factices."); return {"type":"unknown","size_bytes":0,"entropy_normalized":0.0,"error":True}
    def log_dataset_script(message): print(message)
    print("AVERTISSEMENT (create_dataset): ai_analyzer non chargé. Utilisation de features factices.")

try:
    from aicompress.classic_compressors import (
        stored_compress, deflate_compress, bzip2_compress, 
        lzma_compress, zstd_compress, brotli_compress, # Importer la nouvelle fonction brotli
        ZSTD_AVAILABLE as ZSTD_MODULE_AVAILABLE, # Flag de classic_compressors
        BROTLI_AVAILABLE as BROTLI_MODULE_AVAILABLE # NOUVEAU FLAG
    )
    CLASSIC_COMPRESSORS_LOADED_FOR_DATASET = True
except ImportError:
    CLASSIC_COMPRESSORS_LOADED_FOR_DATASET = False
    ZSTD_MODULE_AVAILABLE = False; BROTLI_MODULE_AVAILABLE = False
    def stored_compress(d,p,l): return d
    def deflate_compress(d,p,l): return None # etc. pour tous les fallbacks
    def bzip2_compress(d,p,l): return None
    def lzma_compress(d,p,l): return None
    def zstd_compress(d,p,l): return None
    def brotli_compress(d,p,l): return None # Fallback pour Brotli
    print("AVERTISSEMENT (create_dataset): classic_compressors.py non chargé. Certaines compressions seront indisponibles.")

# --- Configuration ---
SOURCE_FILES_DIR = "./dataset_source_files/"  
OUTPUT_CSV_FILE = "compression_decision_dataset_v2_brotli.csv" # Nouveau nom pour le dataset

# --- Fonctions Wrapper pour appeler les compresseurs de classic_compressors.py ---
# Ces wrappers adaptent la signature pour COMPRESSION_METHODS_SETUP (data, params_dict)
# et passent le log_callback de ce script.

def stored_wrapper_for_dataset(data_bytes, params):
    return stored_compress(data_bytes, params, log_callback=log_dataset_script)
def deflate_wrapper_for_dataset(data_bytes, params):
    return deflate_compress(data_bytes, params, log_callback=log_dataset_script)
def bzip2_wrapper_for_dataset(data_bytes, params):
    return bzip2_compress(data_bytes, params, log_callback=log_dataset_script)
def lzma_wrapper_for_dataset(data_bytes, params):
    return lzma_compress(data_bytes, params, log_callback=log_dataset_script)
def zstd_wrapper_for_dataset(data_bytes, params):
    if not ZSTD_MODULE_AVAILABLE: return None # Vérifier ici aussi pour éviter erreurs si classic_compressors a chargé Zstd mais que la lib manque
    return zstd_compress(data_bytes, params, log_callback=log_dataset_script)

# NOUVEAU WRAPPER POUR BROTLI
def brotli_wrapper_for_dataset(data_bytes, params):
    if not BROTLI_MODULE_AVAILABLE: return None
    return brotli_compress(data_bytes, params, log_callback=log_dataset_script)

# --- Moteurs de Compression à Tester ---
COMPRESSION_METHODS_SETUP = [
    {"name": "STORED",       "func": stored_wrapper_for_dataset,   "params": {}},
    {"name": "DEFLATE_L1",   "func": deflate_wrapper_for_dataset,  "params": {"level": 1}},
    {"name": "DEFLATE_L6",   "func": deflate_wrapper_for_dataset,  "params": {"level": 6}},
    {"name": "DEFLATE_L9",   "func": deflate_wrapper_for_dataset,  "params": {"level": 9}},
    {"name": "BZIP2_L9",     "func": bzip2_wrapper_for_dataset,    "params": {"level": 9}},
    {"name": "LZMA_P0",      "func": lzma_wrapper_for_dataset,     "params": {"preset": 0}},
    {"name": "LZMA_P6",      "func": lzma_wrapper_for_dataset,     "params": {"preset": 6}},
    {"name": "LZMA_P9",      "func": lzma_wrapper_for_dataset,     "params": {"preset": 9}},
    {"name": "ZSTD_L1",      "func": zstd_wrapper_for_dataset,     "params": {"level": 1}},
    {"name": "ZSTD_L3",      "func": zstd_wrapper_for_dataset,     "params": {"level": 3}},
    {"name": "ZSTD_L9",      "func": zstd_wrapper_for_dataset,     "params": {"level": 9}},
    {"name": "ZSTD_L15",     "func": zstd_wrapper_for_dataset,     "params": {"level": 15}},
    # NOUVELLES MÉTHODES BROTLI (qualité 0-11)
    {"name": "BROTLI_L1",    "func": brotli_wrapper_for_dataset,   "params": {"quality": 1}}, # Rapide
    {"name": "BROTLI_L6",    "func": brotli_wrapper_for_dataset,   "params": {"quality": 6}}, # Bon équilibre
    {"name": "BROTLI_L11",   "func": brotli_wrapper_for_dataset,   "params": {"quality": 11}},# Max, lent
]
# Nous n'incluons pas l'AE ici pour l'instant, pour garder ce script axé sur les compresseurs classiques.

# --- Fonction Principale pour Générer le Dataset ---
def generate_dataset():
    log_dataset_script(f"Début de la génération du jeu de données : {OUTPUT_CSV_FILE}")

    if not AI_ANALYZER_LOADED_FOR_DATASET or not CLASSIC_COMPRESSORS_LOADED_FOR_DATASET:
        log_dataset_script("ERREUR: Modules ai_analyzer ou classic_compressors non chargés. Impossible de générer le dataset.")
        return

    if not os.path.exists(SOURCE_FILES_DIR) or not os.listdir(SOURCE_FILES_DIR) and not glob.glob(os.path.join(SOURCE_FILES_DIR, "**", "*"), recursive=True):
        log_dataset_script(f"ERREUR: Dossier source '{SOURCE_FILES_DIR}' vide ou inexistant.")
        log_dataset_script(f"Veuillez le créer et y placer des fichiers variés."); return

    header = ["relative_path", "file_type_analysis", "original_size_bytes", "entropy_normalized", "quick_comp_ratio",
              "best_method", "best_compressed_size_bytes", "best_time_ms"]
    for method_info in COMPRESSION_METHODS_SETUP: # S'assure que toutes les méthodes sont dans le header
        header.extend([f"{method_info['name']}_size_bytes", f"{method_info['name']}_time_ms"])

    with open(OUTPUT_CSV_FILE, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)

        source_filepaths_to_process = []
        log_dataset_script(f"Recherche récursive des fichiers dans : {SOURCE_FILES_DIR}")
        for root, _, files in os.walk(SOURCE_FILES_DIR):
            for filename in files: source_filepaths_to_process.append(os.path.join(root, filename))
        if not source_filepaths_to_process: log_dataset_script(f"ERREUR: Aucun fichier trouvé."); return
        log_dataset_script(f"Trouvé {len(source_filepaths_to_process)} fichiers sources à traiter.")

        for i, filepath in enumerate(source_filepaths_to_process):
            relative_path = os.path.relpath(filepath, SOURCE_FILES_DIR)
            log_dataset_script(f"\nTraitement Fichier {i+1}/{len(source_filepaths_to_process)} : {relative_path}")

            features = get_file_features(filepath, log_callback=log_dataset_script)
            if features.get("error"): log_dataset_script(f"  Erreur features. Ignoré."); continue
            original_size = features["size_bytes"]
            if original_size == 0: log_dataset_script(f"  Fichier vide. Ignoré."); continue
            if original_size < 20 : log_dataset_script(f"  Fichier trop petit ({original_size}o). Ignoré."); continue

            try:
                with open(filepath, 'rb') as f: original_data = f.read()
            except Exception as e: log_dataset_script(f"  Erreur lecture: {e}. Ignoré."); continue

            row_data_prefix = [relative_path, features["type"], original_size, 
                               features["entropy_normalized"], features.get("quick_comp_ratio", 1.0)]
            results_for_file = [] 

            for method_info in COMPRESSION_METHODS_SETUP:
                method_name = method_info["name"]
                compress_func = method_info["func"]
                params = method_info["params"]

                # Vérifier la disponibilité pour ZSTD et BROTLI avant de tester
                if "ZSTD" in method_name and not ZSTD_MODULE_AVAILABLE:
                    log_dataset_script(f"    Méthode {method_name} ignorée (ZSTD non disponible).")
                    results_for_file.append({"name": method_name, "size": original_size, "time": float('inf')})
                    continue
                if "BROTLI" in method_name and not BROTLI_MODULE_AVAILABLE:
                    log_dataset_script(f"    Méthode {method_name} ignorée (Brotli non disponible).")
                    results_for_file.append({"name": method_name, "size": original_size, "time": float('inf')})
                    continue

                # log_dataset_script(f"  Test méthode: {method_name}...") # Peut être trop verbeux
                compressed_data = None; comp_size = original_size; time_ms = float('inf') 

                try:
                    start_time = time.perf_counter()
                    compressed_data = compress_func(original_data, params)
                    end_time = time.perf_counter()
                    if compressed_data is not None:
                        comp_size = len(compressed_data); time_ms = (end_time - start_time) * 1000
                    else: log_dataset_script(f"    Échec compression {method_name} (func a retourné None). Taille=originale.")
                except Exception as e_comp: log_dataset_script(f"    ERREUR méthode {method_name}: {e_comp}")

                results_for_file.append({"name": method_name, "size": comp_size, "time": time_ms})
                if time_ms != float('inf'): log_dataset_script(f"    {method_name}: Taille={comp_size}, Temps={time_ms:.2f} ms")
                else: log_dataset_script(f"    {method_name}: ÉCHEC (Taille={comp_size})")

            if not results_for_file: log_dataset_script(f"  Aucun résultat. Ignoré."); continue

            best_method_info = None
            for res_item in results_for_file:
                if res_item["time"] == float('inf'): continue 
                if best_method_info is None: best_method_info = res_item
                else:
                    if res_item["size"] < best_method_info["size"]: best_method_info = res_item
                    elif res_item["size"] == best_method_info["size"] and res_item["time"] < best_method_info["time"]: best_method_info = res_item

            if best_method_info is None: # Si toutes les méthodes ont échoué
                log_dataset_script(f"  Toutes les méthodes ont échoué pour {relative_path}. Ignoré.")
                continue

            current_row_to_write = list(row_data_prefix)
            current_row_to_write.extend([best_method_info["name"], best_method_info["size"], round(best_method_info["time"], 2)])
            for method_template in COMPRESSION_METHODS_SETUP:
                found_res = next((r for r in results_for_file if r["name"] == method_template["name"]), None)
                if found_res: current_row_to_write.extend([found_res["size"], round(found_res["time"],2) if found_res["time"] != float('inf') else -1.0])
                else: current_row_to_write.extend([original_size, -1.0]) 
            writer.writerow(current_row_to_write)
            log_dataset_script(f"  Meilleur pour {relative_path}: {best_method_info['name']} (Taille: {best_method_info['size']}, Temps: {best_method_info['time']:.2f} ms)")

    log_dataset_script(f"\nJeu de données sauvegardé dans : {OUTPUT_CSV_FILE}")

if __name__ == '__main__':
    if not os.path.exists(SOURCE_FILES_DIR): os.makedirs(SOURCE_FILES_DIR, exist_ok=True); print(f"Dossier source '{SOURCE_FILES_DIR}' créé.")
    if not os.listdir(SOURCE_FILES_DIR) and not glob.glob(os.path.join(SOURCE_FILES_DIR, "**", "*"), recursive=True):
         print(f"Le dossier source '{SOURCE_FILES_DIR}' est vide. Veuillez y placer des fichiers.")
    else:
        generate_dataset()