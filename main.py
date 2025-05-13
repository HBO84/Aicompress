# main.py

from aicompress.core import (
    compress_to_aic,
    decompress_aic, # decompress_aic est utilisé par extract_archive pour les .aic
    extract_archive,
    RARFILE_AVAILABLE,
    AI_ANALYZER_AVAILABLE # Importer pour vérifier si l'analyseur est prêt
)
import os
import shutil

# --- Définition des fichiers et dossiers de test ---

# Fichiers pour la compression AIC : Utilisons certains des fichiers où l'IA a un sens
# Assurez-vous que ces fichiers existent à la racine de AICompressProject
# ou que setup_test_files les crée.
# Pour ce test, nous allons supposer que les fichiers de ai_data_samples
# sont copiés ou créés à la racine pour la simplicité du test de compression.
# Vous pouvez aussi directement compresser le dossier ai_data_samples.

FICHIERS_POUR_AIC_TEST = [
    "doc1.txt", # Fichier texte simple
    "python_code_1.py", # Fichier Python (celui qui était mal classé)
    "english_text_1.txt", # Fichier texte anglais
    "mon_dossier_test", # Un dossier contenant d'autres fichiers
    "ai_data_samples" # Compressons aussi le dossier contenant les échantillons IA
]
NOM_ARCHIVE_AIC = "ma_super_archive_ia.aic"
DOSSIER_EXTRACTION_AIC = "contenu_extrait_de_aic_ia"

# Fichiers pour les tests ZIP et RAR (inchangés)
NOM_ARCHIVE_ZIP_STD = "test_standard.zip"
DOSSIER_EXTRACTION_ZIP_STD = "contenu_extrait_zip_std"

NOM_ARCHIVE_RAR = "test.rar"
DOSSIER_EXTRACTION_RAR = "contenu_extrait_rar"


def setup_test_files():
    """Crée des fichiers et dossiers de base pour les tests si non présents."""
    print("--- Configuration des fichiers de test ---")
    # Fichiers de base
    if not os.path.exists("doc1.txt"):
        with open("doc1.txt", "w") as f: f.write("Ceci est le document un, pour tester AICompress.")
    if not os.path.exists("mon_dossier_test"):
        os.makedirs("mon_dossier_test/sous_dossier_test", exist_ok=True)
        with open("mon_dossier_test/fichier_interne.txt", "w") as f: f.write("Fichier interne dans mon_dossier_test.")
        with open("mon_dossier_test/sous_dossier_test/profond.txt", "w") as f: f.write("Fichier profond dans un sous-dossier.")

    # S'assurer que les fichiers/dossiers pour l'analyse IA existent (ceux de ai_data_samples)
    # Le script setup_etape3_ia.sh les a déjà créés dans ai_data_samples/
    # On s'assure juste que le dossier lui-même est là pour être compressé.
    if not os.path.isdir("ai_data_samples"):
        print("Avertissement : Le dossier 'ai_data_samples' n'a pas été trouvé.")
        print("Veuillez exécuter le script setup_etape3_ia.sh ou le créer manuellement.")
    else:
        # Créer/copier les fichiers spécifiques que nous voulons à la racine pour un test simple de compression de fichiers individuels.
        # python_code_1.py
        if os.path.exists("ai_data_samples/python_code_1.py") and not os.path.exists("python_code_1.py"):
            shutil.copy("ai_data_samples/python_code_1.py", "python_code_1.py")
        elif not os.path.exists("python_code_1.py"):
             with open("python_code_1.py", "w") as f: f.write("def sample_func():\n  pass # Fichier python_code_1.py factice")


        # english_text_1.txt
        if os.path.exists("ai_data_samples/english_text_1.txt") and not os.path.exists("english_text_1.txt"):
            shutil.copy("ai_data_samples/english_text_1.txt", "english_text_1.txt")
        elif not os.path.exists("english_text_1.txt"):
            with open("english_text_1.txt", "w") as f: f.write("This is a sample English text for testing. # Fichier english_text_1.txt factice")

    print("Fichiers de test de base vérifiés/créés.\n")


def cleanup_extraction_directory(dir_path):
    """Supprime un dossier d'extraction et son contenu s'il existe."""
    if os.path.exists(dir_path) and os.path.isdir(dir_path):
        shutil.rmtree(dir_path)
        print(f"Dossier d'extraction '{dir_path}' nettoyé.")

def cleanup_archive_file(file_path):
    """Supprime un fichier archive s'il existe."""
    if os.path.exists(file_path) and os.path.isfile(file_path):
        os.remove(file_path)
        print(f"Fichier archive '{file_path}' nettoyé.")


def main():
    setup_test_files()

    print(f"État de l'analyseur IA : {'Disponible' if AI_ANALYZER_AVAILABLE else 'Non disponible'}")
    print(f"État du support RAR : {'Disponible' if RARFILE_AVAILABLE else 'Non disponible'}\n")

    # --- Test de Compression AIC avec Analyse IA ---
    print("--- TEST DE COMPRESSION AIC (AVEC ANALYSE IA) ---")
    cleanup_archive_file(NOM_ARCHIVE_AIC)

    # Filtrer les fichiers/dossiers qui n'existent pas réellement
    fichiers_sources_valides_aic = []
    for path_item in FICHIERS_POUR_AIC_TEST:
        if os.path.exists(path_item):
            fichiers_sources_valides_aic.append(path_item)
        else:
            print(f"Avertissement (Compression AIC): '{path_item}' non trouvé, sera ignoré.")
    
    if fichiers_sources_valides_aic:
        if compress_to_aic(fichiers_sources_valides_aic, NOM_ARCHIVE_AIC):
            print(f"Archive AIC '{NOM_ARCHIVE_AIC}' créée.")
            if os.path.exists(NOM_ARCHIVE_AIC):
                print(f"Taille de l'archive AIC : {os.path.getsize(NOM_ARCHIVE_AIC)} octets.")
        else:
            print(f"Échec de la création de l'archive AIC '{NOM_ARCHIVE_AIC}'.")
    else:
        print("Aucun fichier source valide trouvé pour la compression AIC.")

    # --- Test de Décompression AIC (et vérification des métadonnées IA) ---
    print("\n--- TEST DE DÉCOMPRESSION AIC (VÉRIFIEZ LES MÉTA-DONNÉES IA DANS LA SORTIE) ---")
    cleanup_extraction_directory(DOSSIER_EXTRACTION_AIC)
    if os.path.exists(NOM_ARCHIVE_AIC):
        # La fonction extract_archive appellera decompress_aic pour les .aic
        # et decompress_aic affiche les métadonnées.
        if extract_archive(NOM_ARCHIVE_AIC, DOSSIER_EXTRACTION_AIC):
            print(f"Décompression AIC réussie. Vérifiez le contenu de '{DOSSIER_EXTRACTION_AIC}'.")
            print("IMPORTANT : Examinez les métadonnées affichées ci-dessus pour voir les résultats de l'analyse IA !")
        else:
            print(f"Échec de l'extraction de l'archive AIC '{NOM_ARCHIVE_AIC}'.")
    else:
        print(f"Archive AIC '{NOM_ARCHIVE_AIC}' non trouvée pour le test d'extraction.")

    # --- Test de Décompression ZIP Standard (inchangé) ---
    print("\n--- TEST DE DÉCOMPRESSION ZIP STANDARD ---")
    cleanup_extraction_directory(DOSSIER_EXTRACTION_ZIP_STD)
    if os.path.exists(NOM_ARCHIVE_ZIP_STD):
        if extract_archive(NOM_ARCHIVE_ZIP_STD, DOSSIER_EXTRACTION_ZIP_STD):
            print(f"Décompression ZIP réussie. Vérifiez le contenu de '{DOSSIER_EXTRACTION_ZIP_STD}'.")
        else:
            print(f"Échec de l'extraction de '{NOM_ARCHIVE_ZIP_STD}'.")
    else:
        print(f"Fichier '{NOM_ARCHIVE_ZIP_STD}' non trouvé. Créez-le manuellement pour ce test (ex: avec 7-Zip ou autre).")

    # --- Test de Décompression RAR (inchangé) ---
    print("\n--- TEST DE DÉCOMPRESSION RAR ---")
    cleanup_extraction_directory(DOSSIER_EXTRACTION_RAR)
    if RARFILE_AVAILABLE:
        if os.path.exists(NOM_ARCHIVE_RAR):
            if extract_archive(NOM_ARCHIVE_RAR, DOSSIER_EXTRACTION_RAR):
                print(f"Décompression RAR réussie. Vérifiez le contenu de '{DOSSIER_EXTRACTION_RAR}'.")
            else:
                print(f"Échec de l'extraction de '{NOM_ARCHIVE_RAR}'.")
        else:
            print(f"Fichier '{NOM_ARCHIVE_RAR}' non trouvé. Créez-le manuellement pour ce test (ex: avec WinRAR ou autre).")
    else:
        print(f"Test de décompression RAR sauté car la bibliothèque 'rarfile' ou l'outil 'unrar' n'est pas disponible/configuré.")
        
    print("\n--- Fin des tests ---")

if __name__ == "__main__":
    main()