# aicompress/ai_analyzer.py
import os
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import numpy as np
import joblib
import math
from collections import Counter
import shutil  # Pour le nettoyage dans if __name__ == '__main__'
import zlib  # Pour le test de compressibilité rapide

try:
    import magic

    PYTHON_MAGIC_AVAILABLE = True
except ImportError:
    PYTHON_MAGIC_AVAILABLE = False
    print(
        "AVERTISSEMENT (ai_analyzer.py): 'python-magic' non trouvé. Identification binaire limitée."
    )

MODEL_BASE_DIR = os.path.join(os.path.expanduser("~"), ".aicompress")
MODELS_SPECIFIC_DIR = os.path.join(MODEL_BASE_DIR, "models")
TEXT_CLASSIFIER_MODEL_NAME = "text_classifier.joblib"
TEXT_CLASSIFIER_CONFIG_NAME = "text_classifier_config.json"
TEXT_CLASSIFIER_MODEL_PATH = os.path.join(
    MODELS_SPECIFIC_DIR, TEXT_CLASSIFIER_MODEL_NAME
)
TEXT_CLASSIFIER_CONFIG_PATH = os.path.join(
    MODELS_SPECIFIC_DIR, TEXT_CLASSIFIER_CONFIG_NAME
)

DEFAULT_TEXT_CLASSIFIER_VERSION = "1.0"

text_classifier_model = None
current_text_classifier_version = DEFAULT_TEXT_CLASSIFIER_VERSION


def _default_log_analyzer(message):
    print(message)


def ensure_model_dir_exists(log_callback=_default_log_analyzer):
    try:
        os.makedirs(MODELS_SPECIFIC_DIR, exist_ok=True)
    except Exception as e:
        log_callback(
            f"[AI_ANALYZER] Erreur création dossier modèle {MODELS_SPECIFIC_DIR}: {e}"
        )


def get_local_text_classifier_version(log_callback=_default_log_analyzer):
    global current_text_classifier_version
    ensure_model_dir_exists(log_callback=log_callback)
    if os.path.exists(TEXT_CLASSIFIER_CONFIG_PATH):
        try:
            with open(TEXT_CLASSIFIER_CONFIG_PATH, "r") as f:
                config = json.load(f)
            current_text_classifier_version = config.get(
                "version", DEFAULT_TEXT_CLASSIFIER_VERSION
            )
        except Exception as e:
            log_callback(
                f"[AI_ANALYZER] Erreur lecture config version: {e}. Défaut {DEFAULT_TEXT_CLASSIFIER_VERSION}."
            )
            current_text_classifier_version = DEFAULT_TEXT_CLASSIFIER_VERSION
    else:
        current_text_classifier_version = DEFAULT_TEXT_CLASSIFIER_VERSION
    return current_text_classifier_version


def save_local_text_classifier_version(version_str, log_callback=_default_log_analyzer):
    global current_text_classifier_version
    ensure_model_dir_exists(log_callback=log_callback)
    try:
        with open(TEXT_CLASSIFIER_CONFIG_PATH, "w") as f:
            json.dump({"version": version_str}, f)
        current_text_classifier_version = version_str
        log_callback(f"[AI_ANALYZER] Version modèle texte local MàJ: {version_str}")
    except Exception as e:
        log_callback(f"[AI_ANALYZER] Erreur sauvegarde config version: {e}")


def train_and_save_text_classifier(
    version_to_save=DEFAULT_TEXT_CLASSIFIER_VERSION, log_callback=_default_log_analyzer
):
    global text_classifier_model
    ensure_model_dir_exists(log_callback=log_callback)
    log_callback(
        f"[AI_ANALYZER] Entraînement nouveau modèle texte (v {version_to_save})..."
    )
    training_data = [
        (
            "def class import def for if else elif try except finally with as pass return yield",
            "python_script",
        ),
        ("lambda def __init__ self cls args kwargs decorator", "python_script"),
        ("import os sys json re math numpy pandas sklearn", "python_script"),
        ("the a is are was were will be and or but for of to in on at", "english_text"),
        (
            "sentence paragraph word letter text document write read speak",
            "english_text",
        ),
        (
            "hello world good morning example another however therefore because",
            "english_text",
        ),
    ]
    texts, labels = zip(*training_data)
    model = make_pipeline(CountVectorizer(ngram_range=(1, 2)), MultinomialNB())
    model.fit(texts, labels)
    try:
        joblib.dump(model, TEXT_CLASSIFIER_MODEL_PATH)
        log_callback(
            f"[AI_ANALYZER] Modèle texte sauvegardé: {TEXT_CLASSIFIER_MODEL_PATH}"
        )
        save_local_text_classifier_version(version_to_save, log_callback=log_callback)
        text_classifier_model = model
    except Exception as e:
        log_callback(f"[AI_ANALYZER] Erreur sauvegarde modèle texte: {e}")
        text_classifier_model = model
    return text_classifier_model


def load_text_classifier(force_retrain=False, log_callback=_default_log_analyzer):
    global text_classifier_model
    ensure_model_dir_exists(log_callback=log_callback)
    current_version = get_local_text_classifier_version(log_callback=log_callback)
    log_callback(
        f"[AI_ANALYZER] Version locale modèle texte (au chargement): {current_version}"
    )

    if not force_retrain and os.path.exists(TEXT_CLASSIFIER_MODEL_PATH):
        try:
            log_callback(
                f"[AI_ANALYZER] Chargement modèle texte: {TEXT_CLASSIFIER_MODEL_PATH}"
            )
            text_classifier_model = joblib.load(TEXT_CLASSIFIER_MODEL_PATH)
            log_callback("[AI_ANALYZER] Modèle texte chargé.")
            return text_classifier_model
        except Exception as e:
            log_callback(
                f"[AI_ANALYZER] Erreur chargement modèle texte: {e}. Réentraînement."
            )
    return train_and_save_text_classifier(current_version, log_callback=log_callback)


text_classifier_model = load_text_classifier(log_callback=_default_log_analyzer)


def predict_text_content_type(text_content_str, log_callback=_default_log_analyzer):
    global text_classifier_model
    if text_classifier_model is None:
        log_callback(
            "[AI_ANALYZER] Modèle texte non dispo. Tentative chargement/entraînement."
        )
        load_text_classifier(log_callback=log_callback)
        if text_classifier_model is None:
            return "unknown_text_model_unavailable"
    if not isinstance(text_content_str, str) or not text_content_str.strip():
        return "unknown_text_or_empty"
    try:
        return text_classifier_model.predict([text_content_str])[0]
    except Exception as e:
        log_callback(f"[AI_ANALYZER] Erreur prédiction type texte: {e}")
        return "unknown_text_prediction_error"


def analyze_file_content(file_path, log_callback=_default_log_analyzer):
    if not os.path.exists(file_path):
        return "file_not_found"
    if os.path.getsize(file_path) == 0:
        return "empty_file"

    file_type_by_magic = "unknown"  # Valeur par défaut

    if PYTHON_MAGIC_AVAILABLE:
        try:
            mime_type = magic.from_file(file_path, mime=True)
            if mime_type:
                if mime_type.startswith("image/jpeg"):
                    file_type_by_magic = "jpeg_image"
                elif mime_type.startswith("image/png"):
                    file_type_by_magic = "png_image"
                elif mime_type.startswith("image/gif"):
                    file_type_by_magic = "gif_image"
                elif mime_type.startswith("image/webp"):
                    file_type_by_magic = "webp_image"
                elif mime_type.startswith("image/tiff"):
                    file_type_by_magic = "tiff_image"
                elif mime_type.startswith("image/bmp"):
                    file_type_by_magic = "bmp_image"
                elif mime_type.startswith("application/pdf"):
                    file_type_by_magic = "pdf_document"
                elif mime_type.startswith("application/zip"):
                    file_type_by_magic = "zip_archive"
                elif mime_type.startswith("application/x-rar-compressed"):
                    file_type_by_magic = "rar_archive"
                elif mime_type.startswith("application/x-7z-compressed"):
                    file_type_by_magic = "7z_archive"
                elif mime_type.startswith("application/x-tar"):
                    file_type_by_magic = "tar_archive"
                elif mime_type.startswith("application/gzip") or mime_type.startswith(
                    "application/x-gzip"
                ):
                    file_type_by_magic = "gzip_archive"
                elif mime_type.startswith("application/x-bzip2"):
                    file_type_by_magic = "bzip2_archive"
                elif mime_type.startswith("application/x-xz"):
                    file_type_by_magic = "xz_archive"
                elif mime_type.startswith("audio/mpeg"):
                    file_type_by_magic = "mp3_audio"
                elif mime_type.startswith("audio/"):
                    file_type_by_magic = (
                        mime_type.replace("audio/", "").split(";")[0] + "_audio"
                    )
                elif mime_type.startswith("video/"):
                    file_type_by_magic = (
                        mime_type.replace("video/", "").split(";")[0] + "_video"
                    )
                elif mime_type.startswith("text/x-python"):
                    file_type_by_magic = "python_script"
                elif mime_type.startswith("text/html"):
                    file_type_by_magic = "html_document"
                elif mime_type.startswith("text/xml"):
                    file_type_by_magic = "xml_document"
                elif mime_type.startswith("application/json"):
                    file_type_by_magic = "json_data"
                elif mime_type.startswith("text/css"):
                    file_type_by_magic = "css_stylesheet"
                elif mime_type.startswith("text/plain"):
                    file_type_by_magic = "plain_text"
                elif mime_type.startswith("text/"):
                    file_type_by_magic = "generic_text"
                elif mime_type.startswith("application/octet-stream"):
                    file_type_by_magic = "generic_binary"
                else:
                    file_type_by_magic = mime_type.replace("/", "_").split(";")[
                        0
                    ]  # Type générique basé sur MIME
            else:
                file_type_by_magic = (
                    "unknown_mime_none"  # magic.from_file a retourné None
                )

            text_like_types = [
                "plain_text",
                "python_script",
                "generic_text",
                "html_document",
                "xml_document",
                "json_data",
                "css_stylesheet",
            ]
            if file_type_by_magic in text_like_types:
                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        content_sample = f.read(2048)
                    if file_type_by_magic == "python_script":
                        return "python_script"  # Priorité à magic pour .py

                    refined_text_type = predict_text_content_type(
                        content_sample, log_callback=log_callback
                    )
                    # Retourner le type affiné seulement s'il est informatif
                    return (
                        refined_text_type
                        if refined_text_type and "unknown" not in refined_text_type
                        else file_type_by_magic
                    )
                except Exception:
                    return file_type_by_magic  # En cas d'erreur de lecture, retourner le type de magic
            return (
                file_type_by_magic  # Pour les types non textuels identifiés par magic
            )
        except Exception as e:
            log_callback(
                f"[AI_ANALYZER] Avertissement: Erreur python-magic pour {os.path.basename(file_path)}: {e}"
            )
            # Si magic échoue, on passe au fallback ci-dessous

    # Fallback si python-magic n'est pas disponible ou a échoué
    try:
        with open(file_path, "rb") as f_rb:
            content_sample_bytes = f_rb.read(512)
        if not content_sample_bytes:
            return "empty_file"  # Redondant avec la vérif de taille, mais sûr

        if (
            b"\0" in content_sample_bytes[:256]
        ):  # Heuristique plus forte pour binaire (octet nul au début)
            return "unknown_binary_heuristic_null"
        try:
            content_sample_text = content_sample_bytes.decode("utf-8", errors="ignore")
            # Vérifier les caractères non imprimables sur un échantillon plus large si possible
            non_printable = sum(
                1
                for char_code in content_sample_bytes
                if not (32 <= char_code <= 126 or char_code in [9, 10, 13])
            )
            if (
                non_printable / len(content_sample_bytes) > 0.2
            ):  # Si plus de 20% de non-imprimables
                return "unknown_binary_heuristic_nonprint"

            text_type = predict_text_content_type(
                content_sample_text, log_callback=log_callback
            )
            return (
                text_type
                if text_type and "unknown" not in text_type
                else "generic_text_heuristic"
            )
        except UnicodeDecodeError:
            return "unknown_binary_read_error"  # Clairement pas du texte UTF-8
    except Exception:
        return "error_during_fallback_analysis"


def calculate_shannon_entropy(
    file_path, sample_size=10240, log_callback=_default_log_analyzer
):
    if not os.path.exists(file_path):
        return 0.0  # Fichier non trouvé
    file_s = os.path.getsize(file_path)
    if file_s == 0:
        return 0.0  # Fichier vide

    actual_sample_size = min(
        file_s, sample_size
    )  # Ne pas lire plus que la taille du fichier
    try:
        with open(file_path, "rb") as f:
            data = f.read(actual_sample_size)
        if not data:
            return 0.0

        byte_counts = Counter(data)
        total_bytes = len(
            data
        )  # Utiliser len(data) qui est actual_sample_size ou moins
        entropy = 0.0
        for count in byte_counts.values():
            probability = count / total_bytes
            entropy -= probability * math.log2(probability)
        # Normaliser par log2(nombre de symboles possibles), ici 256 octets possibles (0-255)
        # log2(256) = 8
        return entropy / 8.0 if entropy > 0 else 0.0
    except Exception as e:
        log_callback(
            f"[AI_ANALYZER] Erreur calcul entropie pour {os.path.basename(file_path)}: {e}"
        )
        return 0.0


# Dans aicompress/ai_analyzer.py

# ... (gardez les imports : os, json, sklearn, numpy, joblib, math, Counter, magic)
# ... (gardez les constantes MODEL_BASE_DIR, etc.)
# ... (gardez _default_log_analyzer, ensure_model_dir_exists, et les fonctions de gestion du modèle de texte)
# ... (gardez la fonction calculate_shannon_entropy comme avant)
# def calculate_shannon_entropy(file_path, sample_size=10240, log_callback=_default_log_analyzer):
#     # ... (code de la fonction calculate_shannon_entropy) ...
#     # (elle doit être définie AVANT get_file_features si get_file_features l'appelle)


def get_quick_compressibility(
    file_path,
    sample_size=16384,
    compression_level=1,
    log_callback=_default_log_analyzer,
):
    """
    Calcule un ratio de compressibilité rapide sur un échantillon du fichier.
    Retourne un ratio (original_size / compressed_size).
    Un ratio plus élevé signifie plus compressible. Retourne 1.0 si non compressible ou erreur.
    """
    if not os.path.exists(file_path):
        return 1.0
    file_s = os.path.getsize(file_path)
    # Retourner 1.0 pour les fichiers très petits ou vides car le ratio n'est pas significatif
    # ou peut causer des erreurs avec zlib.compress sur des données vides/trop petites.
    if file_s < 20:  # Un seuil un peu plus élevé pour être sûr avec zlib
        # log_callback(f"[AI_ANALYZER] Fichier {os.path.basename(file_path)} trop petit pour test de compressibilité.")
        return 1.0

    actual_sample_size = min(file_s, sample_size)
    try:
        with open(file_path, "rb") as f:
            data_sample = f.read(actual_sample_size)

        if not data_sample:
            return 1.0

        # Utiliser zlib pour le test de compressibilité rapide
        import zlib

        compressed_sample = zlib.compress(data_sample, level=compression_level)

        original_sample_size = len(data_sample)
        compressed_sample_size = len(compressed_sample)

        if (
            compressed_sample_size == 0
        ):  # Peut arriver si data_sample est vide, bien que vérifié avant
            return 1.0
        if (
            compressed_sample_size >= original_sample_size
        ):  # Pas de gain ou augmentation
            return 1.0

        ratio = original_sample_size / compressed_sample_size
        return round(ratio, 4)
    except Exception as e:
        log_callback(
            f"[AI_ANALYZER] Erreur calcul compressibilité rapide pour {os.path.basename(file_path)}: {e}"
        )
        return 1.0


def get_file_features(file_path, log_callback=_default_log_analyzer):
    """Extrait un dictionnaire de caractéristiques pour un fichier."""
    if not os.path.exists(file_path):
        return {
            "type": "file_not_found",
            "size_bytes": 0,
            "entropy_normalized": 0.0,
            "quick_comp_ratio": 1.0,
            "error": True,
        }

    # Utiliser la fonction analyze_file_content existante pour le type
    file_type = analyze_file_content(file_path, log_callback=log_callback)
    file_size = os.path.getsize(file_path)

    # Calculer l'entropie (s'assurer que calculate_shannon_entropy est définie avant cette fonction)
    file_entropy = calculate_shannon_entropy(file_path, log_callback=log_callback)

    # Calculer la compressibilité rapide
    quick_comp_ratio = get_quick_compressibility(file_path, log_callback=log_callback)

    features = {
        "type": file_type,
        "size_bytes": file_size,
        "entropy_normalized": round(file_entropy, 4),
        "quick_comp_ratio": quick_comp_ratio,  # C'était la ligne qui posait problème ou celle d'avant
    }
    log_callback(
        f"[AI_ANALYZER] Features pour {os.path.basename(file_path)}: {features}"
    )
    return features


# Assurez-vous que la fonction calculate_shannon_entropy est bien définie avant get_file_features si vous l'avez copiée ici.
# Sinon, elle devrait déjà être dans votre ai_analyzer.py de la version précédente.
# AI_ANALYZER_AVAILABLE = True doit être à la fin du module.

# Indiquer que l'analyseur est disponible si ce module est importé avec succès
# et si ses dépendances clés (comme sklearn/joblib pour le modèle de texte) sont là.
# Pour l'instant, on le met à True si le module lui-même s'importe.
AI_ANALYZER_AVAILABLE = True

if __name__ == "__main__":
    # ... (la section de test if __name__ == '__main__' comme avant)
    print(f"[AI_ANALYZER TEST] python-magic: {PYTHON_MAGIC_AVAILABLE}")
    print(
        f"[AI_ANALYZER TEST] Modèle texte initial (v {get_local_text_classifier_version()}): {'Oui' if text_classifier_model else 'Non'}"
    )
    test_dir = "temp_analyzer_test_files"
    os.makedirs(test_dir, exist_ok=True)
    file1_path = os.path.join(test_dir, "test1.txt")
    file2_path = os.path.join(test_dir, "test2.py")
    file3_path = os.path.join(test_dir, "test3_random.bin")
    with open(file1_path, "w") as f:
        f.write(
            "this is a simple english text file with some repetition repetition." * 10
        )
    with open(file2_path, "w") as f:
        f.write("import os\ndef hello():\n  print('world')\n# comment" * 10)
    with open(file3_path, "wb") as f:
        f.write(os.urandom(2048))
    print(f"\n--- Test de get_file_features ---")
    for f_path_test in [
        file1_path,
        file2_path,
        file3_path,
        __file__,
    ]:  # Teste lui-même aussi
        if os.path.exists(f_path_test):
            print(
                f"  - Fichier: {os.path.basename(f_path_test)}, Features: {get_file_features(f_path_test)}"
            )
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    print("\nTests de ai_analyzer terminés.")
