# train_decision_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib # Pour sauvegarder le modèle
import os
import numpy as np

# --- Configuration ---
DATASET_CSV_PATH = "compression_decision_dataset_v2_brotli.csv"
# Sauvegarder les modèles dans le dossier 'aicompress' pour qu'ils soient packagés avec
MODEL_DIR = os.path.join("aicompress") 
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "compression_orchestrator_model.joblib")
LABEL_ENCODER_SAVE_PATH = os.path.join(MODEL_DIR, "orchestrator_label_encoder.joblib")
# PREPROCESSOR_SAVE_PATH = os.path.join(MODEL_DIR, "orchestrator_preprocessor.joblib") # Le préprocesseur est dans le pipeline

def train_orchestrator_model():
    print(f"Chargement du jeu de données depuis : {DATASET_CSV_PATH}")
    if not os.path.exists(DATASET_CSV_PATH):
        print(f"ERREUR: Fichier dataset '{DATASET_CSV_PATH}' non trouvé.")
        print("Veuillez d'abord exécuter 'create_decision_dataset.py'.")
        return

    try:
        df = pd.read_csv(DATASET_CSV_PATH)
    except Exception as e:
        print(f"ERREUR: Impossible de charger le fichier CSV '{DATASET_CSV_PATH}': {e}")
        return
        
    print(f"Jeu de données chargé : {df.shape[0]} lignes, {df.shape[1]} colonnes.")
    print("\nAperçu des premières lignes :")
    print(df.head())

    # Nettoyage simple
    df.dropna(subset=['best_method', 'original_size_bytes', 'entropy_normalized', 'file_type_analysis'], inplace=True)
    df = df[df['best_method'] != "N/A"]
    # S'assurer que les colonnes numériques sont bien numériques et sans infini/NaN après chargement
    df['original_size_bytes'] = pd.to_numeric(df['original_size_bytes'], errors='coerce')
    df['entropy_normalized'] = pd.to_numeric(df['entropy_normalized'], errors='coerce')
    df.dropna(subset=['original_size_bytes', 'entropy_normalized'], inplace=True)

    if df.empty:
        print("ERREUR: Le jeu de données est vide après nettoyage ou ne contient pas de 'best_method' valide.")
        return

    print(f"\nNombre de lignes après nettoyage : {df.shape[0]}")
    if df.shape[0] < 10: # Seuil arbitraire pour un dataset minimal
        print("ERREUR: Pas assez de données valides pour l'entraînement après nettoyage.")
        return

    # --- 1. Préparation des Caractéristiques (X) et de la Cible (y) ---
    features_cols = ['file_type_analysis', 'original_size_bytes', 'entropy_normalized', "quick_comp_ratio"]
    target_col = 'best_method'

    if not all(col in df.columns for col in features_cols + [target_col]):
        print(f"ERREUR: Colonnes manquantes dans le CSV. Attendu: {features_cols + [target_col]}")
        print(f"Colonnes présentes: {df.columns.tolist()}")
        return

    X = df[features_cols]
    y_original_labels = df[target_col]

    label_encoder_target = LabelEncoder()
    y = label_encoder_target.fit_transform(y_original_labels)
    
    print(f"\nClasses cibles (méthodes de compression) uniques encodées : {len(label_encoder_target.classes_)}")
    print("Mapping des classes cibles :")
    for i, class_name in enumerate(label_encoder_target.classes_):
        print(f"  {i}: {class_name}")

    # --- 2. Prétraitement des Caractéristiques ---
    numerical_features = ['original_size_bytes', 'entropy_normalized',"quick_comp_ratio"]
    categorical_features = ['file_type_analysis']

    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False) 

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ], 
        remainder='passthrough' 
    )

    # --- 3. Division des Données ---
    # Note: `stratify=y` a été retiré car il causait une erreur avec les petites classes dans votre dataset.
    # Pour un dataset plus grand et équilibré, il serait bon de le réactiver.
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42 # stratify=y (retiré pour l'instant)
        )
    except ValueError as e_split:
        print(f"Erreur lors de la division des données: {e_split}")
        print("Cela peut arriver si le jeu de données est trop petit ou si une classe est trop peu représentée pour le test_size choisi.")
        print("Essayez avec plus de données ou sans stratification.")
        return

    print(f"\nTaille de l'ensemble d'entraînement : {X_train.shape[0]}")
    print(f"Taille de l'ensemble de test : {X_test.shape[0]}")

    if X_train.shape[0] == 0 or X_test.shape[0] == 0:
        print("ERREUR: Ensembles d'entraînement ou de test vides. Vérifiez la taille de votre dataset et le `test_size`.")
        return

    # --- 4. Création du Pipeline avec Prétraitement et Modèle ---
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1)
    # n_jobs=-1 utilise tous les processeurs disponibles pour l'entraînement

    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', model)])

    # --- 5. Entraînement du Modèle ---
    print("\nDébut de l'entraînement du modèle 'Chef d'Orchestre'...")
    try:
        pipeline.fit(X_train, y_train)
        print("Entraînement terminé.")
    except Exception as e_fit:
        print(f"ERREUR pendant l'entraînement du pipeline: {e_fit}")
        import traceback
        print(traceback.format_exc())
        return


    # --- 6. Évaluation du Modèle ---
    print("\n--- Évaluation du Modèle ---")
    try:
        y_pred_train = pipeline.predict(X_train)
        y_pred_test = pipeline.predict(X_test)
    except Exception as e_pred:
        print(f"ERREUR pendant la prédiction: {e_pred}")
        return

    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)

    print(f"Précision (Accuracy) sur l'ensemble d'entraînement : {train_accuracy:.4f}")
    print(f"Précision (Accuracy) sur l'ensemble de test : {test_accuracy:.4f}")

    print("\nRapport de Classification sur l'ensemble de test :")
    target_names_for_report = label_encoder_target.classes_
    # Correction pour classification_report: spécifier les labels pour inclure toutes les classes apprises par LabelEncoder
    labels_for_report = np.arange(len(target_names_for_report))
    
    # S'assurer que y_test et y_pred_test ne contiennent que des labels connus par labels_for_report
    # Cela peut arriver si le split a isolé des classes.
    # Pour une solution robuste, on peut filtrer les labels inconnus ou s'assurer que `labels` contient tous les labels uniques de y_test et y_pred_test
    unique_labels_in_data = np.unique(np.concatenate((y_test, y_pred_test)))
    filtered_target_names = [target_names_for_report[i] for i in unique_labels_in_data if i < len(target_names_for_report)]
    
    # Si on veut un rapport pour TOUTES les classes apprises, même celles avec 0 support dans le test set:
    # Il faut s'assurer que `labels` dans classification_report correspond bien aux classes possibles
    # et que `target_names` a la même longueur.
    # La solution est d'utiliser les labels dérivés de label_encoder_target.classes_
    
    try:
        print(classification_report(y_test, y_pred_test, 
                                    labels=labels_for_report, # Tous les labels que le modèle peut prédire
                                    target_names=target_names_for_report, 
                                    zero_division=0))
    except ValueError as e_report: # Au cas où il y aurait encore un décalage
        print(f"Erreur dans classification_report: {e_report}")
        print("Tentative de rapport avec les labels présents dans y_test et y_pred_test...")
        try:
            # Utiliser seulement les labels présents dans les données de test et prédictions
            present_labels = np.unique(np.concatenate((y_test,y_pred_test)))
            present_target_names = label_encoder_target.inverse_transform(present_labels)
            print(classification_report(y_test, y_pred_test, 
                                        labels=present_labels,
                                        target_names=present_target_names,
                                        zero_division=0))
        except Exception as e_report_fallback:
            print(f"Échec du rapport de classification fallback: {e_report_fallback}")


    try:
        feature_importances = pipeline.named_steps['classifier'].feature_importances_
        one_hot_feature_names = list(pipeline.named_steps['preprocessor']
                                     .named_transformers_['cat']
                                     .get_feature_names_out(categorical_features))
        all_feature_names = numerical_features + one_hot_feature_names
        
        print("\nImportance des Caractéristiques :")
        # S'assurer que le nombre de features correspond
        if len(feature_importances) == len(all_feature_names):
            for score, name in sorted(zip(feature_importances, all_feature_names), key=lambda x: x[0], reverse=True):
                print(f"  {name}: {score:.4f}")
        else:
            print(f"  Avertissement: Le nombre d'importances ({len(feature_importances)}) ne correspond pas au nombre de noms de features ({len(all_feature_names)}).")
            print(f"  Importances brutes: {feature_importances}")

    except Exception as e_feat:
        print(f"  Impossible d'afficher l'importance des features: {e_feat}")

    # --- 7. Sauvegarde du Pipeline et de l'Encodeur de Label ---
    try:
        os.makedirs(MODEL_DIR, exist_ok=True) # S'assurer que le dossier aicompress/ existe
        print(f"\nSauvegarde du pipeline (préprocesseur + modèle) sous : {MODEL_SAVE_PATH}")
        joblib.dump(pipeline, MODEL_SAVE_PATH)
        
        print(f"Sauvegarde de l'encodeur de label sous : {LABEL_ENCODER_SAVE_PATH}")
        joblib.dump(label_encoder_target, LABEL_ENCODER_SAVE_PATH)
        
        print("\nModèle 'Chef d'Orchestre' et composants associés sauvegardés dans le dossier 'aicompress/'.")
        print("Vous pouvez maintenant intégrer ce modèle dans aicompress/core.py (get_compression_settings).")
    except Exception as e_save:
        print(f"ERREUR lors de la sauvegarde des modèles: {e_save}")

if __name__ == '__main__':
    if not os.path.exists(DATASET_CSV_PATH):
        print(f"Le fichier '{DATASET_CSV_PATH}' est introuvable.")
        print("Veuillez d'abord le générer en exécutant 'python create_decision_dataset.py'")
        print("après avoir peuplé le dossier 'dataset_source_files'.")
    else:
        train_orchestrator_model()