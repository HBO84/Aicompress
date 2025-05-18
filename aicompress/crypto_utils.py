# aicompress/crypto_utils.py

import os

# Fonction de Log par Défaut pour ce module
def _default_log_crypto(message):
    print(message)

try:
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.backends import default_backend
    CRYPTOGRAPHY_AVAILABLE = True
    _default_log_crypto("[CRYPTO_UTILS] Bibliothèque 'cryptography' chargée.")
except ImportError:
    CRYPTOGRAPHY_AVAILABLE = False
    _default_log_crypto("AVERTISSEMENT (crypto_utils.py): 'cryptography' non trouvée. Chiffrement/Déchiffrement désactivé.")

# Paramètres Crypto
AES_KEY_SIZE = 32  # AES-256
PBKDF2_ITERATIONS = 100_000 
SALT_SIZE = 16
IV_SIZE = 12 # GCM recommande 12 octets pour l'IV/nonce
TAG_SIZE = 16 # GCM tag size (standard pour AES GCM est 16 octets / 128 bits)

def _derive_key(password: str, salt: bytes) -> bytes | None:
    """Dérive une clé à partir d'un mot de passe et d'un sel en utilisant PBKDF2."""
    if not password: 
        # Ne devrait pas arriver si on chiffre, mais sécurité
        return None 
    if not CRYPTOGRAPHY_AVAILABLE: return None

    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=AES_KEY_SIZE,
        salt=salt,
        iterations=PBKDF2_ITERATIONS,
        backend=default_backend()
    )
    return kdf.derive(password.encode('utf-8'))

def encrypt_data(data: bytes, password: str, log_callback=_default_log_crypto) -> tuple | None:
    """
    Chiffre les données en utilisant AES-GCM.
    Retourne un tuple (encrypted_data_sans_tag, salt, iv, tag) ou None en cas d'échec.
    Si password est None ou vide, retourne (data, None, None, None) -> non chiffré.
    """
    if not password: # Pas de mot de passe, pas de chiffrement
        log_callback("[CRYPTO_UTILS] Aucun mot de passe fourni, données non chiffrées.")
        return data, None, None, None 

    if not CRYPTOGRAPHY_AVAILABLE:
        log_callback("[CRYPTO_UTILS] Bibliothèque Cryptography non disponible. Chiffrement impossible.")
        return None # Indique un échec de chiffrement forcé

    salt = os.urandom(SALT_SIZE)
    key = _derive_key(password, salt)
    if key is None: # Ne devrait pas arriver si password est non vide
         log_callback("[CRYPTO_UTILS] Échec de la dérivation de la clé (pas de mot de passe?).")
         return None

    iv = os.urandom(IV_SIZE) # Nonce pour GCM

    try:
        cipher = Cipher(algorithms.AES(key), modes.GCM(iv), backend=default_backend())
        encryptor = cipher.encryptor()
        # encryptor.authenticate_additional_data(b"associated_data_if_any") # Optionnel
        encrypted_data_payload = encryptor.update(data) + encryptor.finalize()
        tag = encryptor.tag # Obtenir le tag GCM (très important pour l'intégrité et l'authenticité)

        log_callback("[CRYPTO_UTILS] Données chiffrées avec AES-GCM.")
        return encrypted_data_payload, salt, iv, tag
    except Exception as e:
        log_callback(f"[CRYPTO_UTILS] Erreur pendant le chiffrement AES-GCM: {e}")
        return None


def decrypt_data(encrypted_data_payload: bytes, password: str, salt: bytes, iv: bytes, tag: bytes, log_callback=_default_log_crypto) -> bytes | None:
    """
    Déchiffre les données en utilisant AES-GCM.
    Nécessite encrypted_data_payload (sans le tag), salt, iv, et le tag séparément.
    Retourne les données déchiffrées ou None en cas d'échec (ex: mauvais mot de passe, données corrompues).
    """
    if not CRYPTOGRAPHY_AVAILABLE:
        log_callback("[CRYPTO_UTILS] Bibliothèque Cryptography non disponible. Déchiffrement impossible.")
        return None
    if not password: # Si les données étaient chiffrées, un mot de passe est absolument nécessaire
        log_callback("[CRYPTO_UTILS] Mot de passe requis pour déchiffrer les données.")
        return None

    key = _derive_key(password, salt)
    if key is None:
         log_callback("[CRYPTO_UTILS] Échec de la dérivation de la clé pour le déchiffrement.")
         return None

    try:
        # Pour GCM, le tag est fourni à l'initialisation du mode GCM pour le déchiffrement
        cipher = Cipher(algorithms.AES(key), modes.GCM(iv, tag), backend=default_backend())
        decryptor = cipher.decryptor()
        # decryptor.authenticate_additional_data(b"associated_data_if_any") # Doit correspondre à celui du chiffrement

        decrypted_data = decryptor.update(encrypted_data_payload) + decryptor.finalize()
        # Si on arrive ici, le tag a été vérifié et le déchiffrement est réussi
        log_callback("[CRYPTO_UTILS] Données déchiffrées avec succès (AES-GCM).")
        return decrypted_data
    except Exception as e: 
        # cryptography.exceptions.InvalidTag est une exception courante ici si le mdp/clé/tag est mauvais
        log_callback(f"[CRYPTO_UTILS] ÉCHEC DU DÉCHIFFREMENT (AES-GCM): {e}. Mot de passe incorrect ou données corrompues/altérées.")
        return None

# Fin de aicompress/crypto_utils.py