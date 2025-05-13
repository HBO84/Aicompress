# test_hash.py
import hashlib
import os

def calculate_sha256_from_python(filepath):
    sha256_hash = hashlib.sha256()
    try:
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    except IOError as e:
        print(f"Erreur lors de la lecture du fichier '{filepath}': {e}")
        return None

# IMPORTANT : Remplacez cette ligne par le chemin correct vers VOTRE fichier
# text_classifier_v1.1.joblib local (celui de référence).
# S'il est dans le même dossier que test_hash.py, vous pouvez juste mettre le nom du fichier.
fichier_modele_reference = "text_classifier_v1.1.joblib" 

if os.path.exists(fichier_modele_reference):
    hash_python = calculate_sha256_from_python(fichier_modele_reference)
    print(f"Checksum SHA256 calculé par Python pour '{fichier_modele_reference}':")
    print(hash_python)
    print("\nMaintenant, dans votre terminal, exécutez la commande suivante sur le MÊME fichier:")
    print(f"sha256sum \"{os.path.abspath(fichier_modele_reference)}\"")
    print("Et comparez les deux résultats.")
else:
    print(f"ERREUR : Le fichier de référence '{fichier_modele_reference}' n'a pas été trouvé.")
    print("Veuillez vérifier le chemin dans la variable 'fichier_modele_reference'.")