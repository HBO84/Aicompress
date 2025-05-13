# create_decision_dataset.py
# Script pour générer un jeu de données pour entraîner l'IA "Chef d'Orchestre"

import os
import glob
import csv
import time
import json 
import zipfile 
import bz2
import lzma
import numpy as np
import zlib # Pour une alternative à zipfile.compress si Python < 3.7 ou pour plus de contrôle

# Importer les modules nécessaires d'AICompress
try:
    from aicompress.ai_analyzer import get_file_features, _default_log_analyzer as log
    AI_ANALYZER_LOADED = True
except ImportError:
    AI_ANALYZER_LOADED = False
    def get_file_features(filepath, log_callback=print): 
        log_callback(f"[DATASET_GEN_FALLBACK] ai_analyzer non chargé. Features basiques pour {os.path.basename(filepath)}.")
        if not os.path.exists(filepath): return {"type": "file_not_found", "size_bytes": 0, "entropy_normalized": 0.0, "error": True}
        return {"type": "unknown_type", "size_bytes": os.path.getsize(filepath), "entropy_normalized": 0.5, "error": False}
    def log(message): print(message)
    print("AVERTISSEMENT: aicompress.ai_analyzer n'a pas pu être importé. Utilisation de features factices.")

AE_ENGINE_AVAILABLE = False # On le laisse désactivé pour l'instant pour ce script

# --- Configuration ---
SOURCE_FILES_DIR = "./dataset_source_files/"
OUTPUT_CSV_FILE = "compression_decision_dataset.csv"

# --- Fonctions Wrapper pour les Compresseurs ---
def stored_wrapper(data_bytes, params):
    return data_bytes

def zlib_deflate_wrapper(data_bytes, params):
    level = params.get("level", 6) # Défaut à 6 si non spécifié
    try:
        return zlib.compress(data_bytes, level=level)
    except Exception as e:
        log(f"    Erreur zlib.compress L{level}: {e}")
        return None # Indiquer l'échec

def bzip2_wrapper(data_bytes, params):
    level = params.get("level", 9) # Défaut à 9
    valid_level = max(1, min(9, level))
    try:
        return bz2.compress(data_bytes, compresslevel=valid_level)
    except Exception as e:
        log(f"    Erreur bz2.compress L{valid_level}: {e}")
        return None

def lzma_wrapper(data_bytes, params):
    preset_level = params.get("preset_level", 6) # Défaut à 6
    valid_preset = max(0, min(9, preset_level))
    try:
        # FORMAT_XZ ajoute des en-têtes de conteneur, ce qui est bien pour une "taille de fichier"
        return lzma.compress(data_bytes, format=lzma.FORMAT_XZ, preset=valid_preset)
    except Exception as e:
        log(f"    Erreur lzma.compress P{valid_preset}: {e}")
        return None

# --- Moteurs de Compression à Tester ---
COMPRESSION_METHODS_SETUP = [
    {"name": "STORED",       "func": stored_wrapper,         "params": {}},
    {"name": "DEFLATE_L1",   "func": zlib_deflate_wrapper,   "params": {"level": 1}},
    {"name": "DEFLATE_L6",   "func": zlib_deflate_wrapper,   "params": {"level": 6}},
    {"name": "DEFLATE_L9",   "func": zlib_deflate_wrapper,   "params": {"level": 9}},
    {"name": "BZIP2_L9",     "func": bzip2_wrapper,          "params": {"level": 9}},
    {"name": "LZMA_P0",      "func": lzma_wrapper,           "params": {"preset_level": 0}},
    {"name": "LZMA_P6",      "func": lzma_wrapper,           "params": {"preset_level": 6}},
    {"name": "LZMA_P9",      "func": lzma_wrapper,           "params": {"preset_level": 9}},
    # {"name": "AE_CIFAR10_COLOR", "func": compress_with_ae_cifar10, "params": {"requires_filepath": True}}, # Laissé pour plus tard
]

# --- Fonction Principale pour Générer le Dataset ---
def generate_dataset():
    log(f"Début de la génération du jeu de données : {OUTPUT_CSV_FILE}")
    
    if not AI_ANALYZER_LOADED:
        log("ERREUR: ai_analyzer non chargé. Impossible de générer les features."); return
    if not os.path.exists(SOURCE_FILES_DIR) or not os.listdir(SOURCE_FILES_DIR) and not glob.glob(os.path.join(SOURCE_FILES_DIR, "**", "*"), recursive=True):
        log(f"ERREUR: Dossier source '{SOURCE_FILES_DIR}' vide ou inexistant.");
        log(f"Veuillez le créer et y placer des fichiers variés."); return

    header = ["relative_path", "file_type_analysis", "original_size_bytes", "entropy_normalized", 
              "best_method", "best_compressed_size_bytes", "best_time_ms"]
    for method_info in COMPRESSION_METHODS_SETUP:
        header.extend([f"{method_info['name']}_size_bytes", f"{method_info['name']}_time_ms"])

    with open(OUTPUT_CSV_FILE, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)

        source_filepaths_to_process = []
        log(f"Recherche récursive des fichiers dans : {SOURCE_FILES_DIR}")
        for root, _, files in os.walk(SOURCE_FILES_DIR):
            for filename in files:
                filepath = os.path.join(root, filename)
                source_filepaths_to_process.append(filepath)
        
        if not source_filepaths_to_process:
            log(f"ERREUR: Aucun fichier trouvé récursivement dans {SOURCE_FILES_DIR}."); return

        log(f"Trouvé {len(source_filepaths_to_process)} fichiers sources à traiter.")

        for i, filepath in enumerate(source_filepaths_to_process):
            relative_path = os.path.relpath(filepath, SOURCE_FILES_DIR)
            log(f"\nTraitement Fichier {i+1}/{len(source_filepaths_to_process)} : {relative_path}")

            features = get_file_features(filepath, log_callback=log)
            if features.get("error"): log(f"  Erreur features {relative_path}. Ignoré."); continue
            original_size = features["size_bytes"]
            if original_size == 0: log(f"  Fichier {relative_path} vide. Ignoré."); continue
            # Certains compresseurs peuvent échouer ou mal se comporter avec de très petits fichiers
            if original_size < 20 : log(f"  Fichier {relative_path} très petit ({original_size}o). Ignoré pour éviter erreurs compresseurs."); continue


            try:
                with open(filepath, 'rb') as f: original_data = f.read()
            except Exception as e: log(f"  Erreur lecture {relative_path}: {e}. Ignoré."); continue

            row_data_prefix = [relative_path, features["type"], original_size, features["entropy_normalized"]]
            results_for_file = [] 

            for method_info in COMPRESSION_METHODS_SETUP:
                method_name = method_info["name"]
                compress_func = method_info["func"]
                params = method_info["params"]
                
                # log(f"  Test méthode: {method_name}...") # Peut être trop verbeux
                compressed_data = None
                comp_size = original_size # Par défaut si échec
                time_ms = float('inf')  # Temps infini si échec

                try:
                    start_time = time.perf_counter()
                    # Passer file_path si la fonction de compression en a besoin (ex: AE)
                    # if params.get("requires_filepath"):
                    #     compressed_data = compress_func(None, params, file_path_for_preprocessing=filepath)
                    # else:
                    compressed_data = compress_func(original_data, params)
                    end_time = time.perf_counter()
                    
                    if compressed_data is not None:
                        comp_size = len(compressed_data)
                        time_ms = (end_time - start_time) * 1000
                    else: 
                        log(f"    {method_name} a retourné None (échec compression).")
                except Exception as e_comp:
                    log(f"    ERREUR pendant compression avec {method_name}: {e_comp}")
                
                results_for_file.append({"name": method_name, "size": comp_size, "time": time_ms})
                if time_ms != float('inf'):
                    log(f"    {method_name}: Taille={comp_size}, Temps={time_ms:.2f} ms")
                else:
                    log(f"    {method_name}: ÉCHEC (Taille={comp_size})")

            if not results_for_file: log(f"  Aucun résultat pour {relative_path}. Ignoré."); continue

            # --- LOGIQUE DE SÉLECTION DE LA MEILLEURE MÉTHODE CORRIGÉE ---
            best_method_info = None
            for res_item in results_for_file:
                # On considère un résultat valide si le temps n'est pas infini (pas d'échec)
                if res_item["time"] == float('inf'): 
                    continue # Ignorer les méthodes qui ont échoué

                if best_method_info is None: # Premier résultat valide trouvé
                    best_method_info = res_item
                else:
                    # Critère 1: Meilleure taille de compression
                    if res_item["size"] < best_method_info["size"]:
                        best_method_info = res_item
                    # Critère 2: Si tailles égales, temps de compression plus rapide
                    elif res_item["size"] == best_method_info["size"]:
                        if res_item["time"] < best_method_info["time"]:
                            best_method_info = res_item
            
            if best_method_info is None: # Si toutes les méthodes ont échoué (ne devrait pas arriver avec STORED)
                                         # ou si tous les fichiers étaient trop petits.
                log(f"  Échec de toutes les méthodes de compression valides pour {relative_path}. Ignoré.")
                continue
            
            best_method_name = best_method_info["name"]
            best_size = best_method_info["size"]
            best_time = best_method_info["time"]
            # --- FIN LOGIQUE CORRIGÉE ---
            
            current_row_to_write = list(row_data_prefix)
            current_row_to_write.extend([best_method_name, best_size, round(best_time, 2) if best_time != float('inf') else -1.0])
            
            for method_template in COMPRESSION_METHODS_SETUP:
                found_res = next((r for r in results_for_file if r["name"] == method_template["name"]), None)
                if found_res:
                    current_row_to_write.extend([found_res["size"], round(found_res["time"],2) if found_res["time"] != float('inf') else -1.0])
                else: # Ne devrait pas arriver si results_for_file est bien peuplé
                    current_row_to_write.extend([original_size, -1.0]) 
            
            writer.writerow(current_row_to_write)
            log(f"  Meilleur pour {relative_path}: {best_method_name} (Taille: {best_size}, Temps: {best_time:.2f} ms)")

    log(f"\nJeu de données sauvegardé dans : {OUTPUT_CSV_FILE}")

if __name__ == '__main__':
    if not os.path.exists(SOURCE_FILES_DIR):
        os.makedirs(SOURCE_FILES_DIR, exist_ok=True)
        print(f"Dossier source '{SOURCE_FILES_DIR}' créé.")
    
    # Vérifier si le dossier source est réellement vide (même avec sous-dossiers)
    found_any_file = False
    if os.path.exists(SOURCE_FILES_DIR):
        for _, _, files_in_walk in os.walk(SOURCE_FILES_DIR):
            if files_in_walk:
                found_any_file = True
                break
    
    if not found_any_file:
         print(f"Le dossier source '{SOURCE_FILES_DIR}' est vide ou ne contient que des dossiers vides.")
         print(f"Veuillez y placer une grande variété de fichiers (texte, code, images, binaires, etc.)")
         print(f"dans ce dossier ou ses sous-dossiers pour générer un jeu de données pertinent.")
         print(f"Une fois les fichiers en place, relancez ce script.")
    else:
        generate_dataset()