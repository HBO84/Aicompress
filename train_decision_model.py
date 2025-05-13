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
DATASET_CSV_PATH = "compression_decision_dataset.csv"
MODEL_SAVE_PATH = os.path.join("aicompress", "compression_orchestrator_model.joblib") # Sauvegarder dans le module
PREPROCESSOR_SAVE_PATH = os.path.join("aicompress", "orchestrator_preprocessor.joblib") # Sauvegarder le préprocesseur

def train_orchestrator_model():
    print(f"Chargement du jeu de données depuis : {DATASET_CSV_PATH}")
    if not os.path.exists(DATASET_CSV_PATH):
        print(f"ERREUR: Fichier dataset '{DATASET_CSV_PATH}' non trouvé.")
        print("Veuillez d'abord exécuter 'create_decision_dataset.py'.")
        return

    df = pd.read_csv(DATASET_CSV_PATH)
    print(f"Jeu de données chargé : {df.shape[0]} lignes, {df.shape[1]} colonnes.")
    print("\nAperçu des premières lignes :")
    print(df.head())

    # Nettoyage simple : supprimer les lignes où la meilleure méthode n'a pas pu être déterminée
    # ou où la taille/temps sont invalides (ex: si on a gardé des -1 ou infini)
    # Ici, on suppose que create_decision_dataset.py a déjà bien filtré.
    # On va juste s'assurer que best_method n'est pas N/A
    df.dropna(subset=['best_method', 'original_size_bytes', 'entropy_normalized', 'file_type_analysis'], inplace=True)
    df = df[df['best_method'] != "N/A"]
    
    if df.empty:
        print("ERREUR: Le jeu de données est vide après nettoyage ou ne contient pas de 'best_method' valide.")
        return

    print(f"\nNombre de lignes après nettoyage simple : {df.shape[0]}")

    # --- 1. Préparation des Caractéristiques (X) et de la Cible (y) ---
    # Caractéristiques à utiliser pour la prédiction
    # 'relative_path' n'est pas une feature, 'best_compressed_size_bytes' et 'best_time_ms' sont des résultats, pas des features.
    # De même pour les colonnes individuelles taille/temps de chaque méthode.
    features_cols = ['file_type_analysis', 'original_size_bytes', 'entropy_normalized']
    target_col = 'best_method'

    X = df[features_cols]
    y_original_labels = df[target_col]

    # Encodage de la cible (les noms des méthodes) en entiers
    label_encoder_target = LabelEncoder()
    y = label_encoder_target.fit_transform(y_original_labels)
    
    print(f"\nClasses cibles (méthodes de compression) uniques encodées : {len(label_encoder_target.classes_)}")
    print("Mapping des classes cibles :")
    for i, class_name in enumerate(label_encoder_target.classes_):
        print(f"  {i}: {class_name}")

    # --- 2. Prétraitement des Caractéristiques ---
    # 'file_type_analysis' est catégorielle -> One-Hot Encoding
    # 'original_size_bytes', 'entropy_normalized' sont numériques -> Standard Scaling
    
    numerical_features = ['original_size_bytes', 'entropy_normalized']
    categorical_features = ['file_type_analysis']

    # Créer le transformateur pour les features numériques
    numerical_transformer = StandardScaler()

    # Créer le transformateur pour les features catégorielles
    # handle_unknown='ignore' : si une nouvelle catégorie apparaît en prédiction, elle sera encodée comme des zéros
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False) 

    # Combiner les transformateurs avec ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ], 
        remainder='passthrough' # Garder les autres colonnes si on en ajoutait, mais ici on ne spécifie que celles à transformer
    )

    # --- 3. Division des Données ---
    X_train, X_test, y_train, y_test, y_train_labels, y_test_labels = train_test_split(
        X, y, y_original_labels, test_size=0.25, random_state=42  # Stratify pour garder la proportion des classes
    )
    print(f"\nTaille de l'ensemble d'entraînement : {X_train.shape[0]}")
    print(f"Taille de l'ensemble de test : {X_test.shape[0]}")

    # --- 4. Création du Pipeline avec Prétraitement et Modèle ---
    # RandomForestClassifier est un bon point de départ
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    # class_weight='balanced' peut aider si certaines méthodes sont beaucoup plus fréquentes que d'autres

    # Pipeline complet
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', model)])

    # --- 5. Entraînement du Modèle ---
    print("\nDébut de l'entraînement du modèle 'Chef d'Orchestre'...")
    pipeline.fit(X_train, y_train)
    print("Entraînement terminé.")

    # --- 6. Évaluation du Modèle ---
    print("\n--- Évaluation du Modèle ---")
    y_pred_train = pipeline.predict(X_train)
    y_pred_test = pipeline.predict(X_test)

    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)

    print(f"Précision (Accuracy) sur l'ensemble d'entraînement : {train_accuracy:.4f}")
    print(f"Précision (Accuracy) sur l'ensemble de test : {test_accuracy:.4f}")

    print("\nRapport de Classification sur l'ensemble de test :")
    # Pour afficher les noms des classes dans le rapport au lieu des entiers :
    target_names_for_report = label_encoder_target.classes_
    print(classification_report(y_test, y_pred_test, target_names=target_names_for_report, zero_division=0))
    
    # print("\nMatrice de Confusion sur l'ensemble de test :")
    # conf_matrix = confusion_matrix(y_test, y_pred_test)
    # print(conf_matrix) # Peut être difficile à lire si beaucoup de classes
    # On pourrait l'afficher avec matplotlib plus tard si besoin

    # Importance des caractéristiques (si RandomForest)
    # Note: L'importance est calculée sur les features après OneHotEncoding
    try:
        feature_importances = pipeline.named_steps['classifier'].feature_importances_
        
        # Obtenir les noms des features après ColumnTransformer
        # Pour les features one-hot encodées, get_feature_names_out() est utile
        one_hot_feature_names = list(pipeline.named_steps['preprocessor']
                                     .named_transformers_['cat']
                                     .get_feature_names_out(categorical_features))
        
        all_feature_names = numerical_features + one_hot_feature_names
        
        print("\nImportance des Caractéristiques :")
        for score, name in sorted(zip(feature_importances, all_feature_names), reverse=True):
            print(f"  {name}: {score:.4f}")
    except Exception as e_feat:
        print(f"  Impossible d'afficher l'importance des features: {e_feat}")


    # --- 7. Sauvegarde du Pipeline (Préprocesseur + Modèle) et de l'Encodeur de Label ---
    print(f"\nSauvegarde du pipeline (préprocesseur + modèle) sous : {MODEL_SAVE_PATH}")
    joblib.dump(pipeline, MODEL_SAVE_PATH)
    
    # Sauvegarder aussi l'encodeur de label pour pouvoir décoder les prédictions plus tard
    label_encoder_save_path = os.path.join(os.path.dirname(MODEL_SAVE_PATH), "orchestrator_label_encoder.joblib")
    joblib.dump(label_encoder_target, label_encoder_save_path)
    print(f"Encodeur de label sauvegardé sous : {label_encoder_save_path}")
    
    # Le préprocesseur est inclus dans le pipeline, mais si on voulait le sauvegarder séparément :
    # print(f"Sauvegarde du préprocesseur sous : {PREPROCESSOR_SAVE_PATH}")
    # joblib.dump(preprocessor, PREPROCESSOR_SAVE_PATH) # Déjà dans le pipeline

    print("\nModèle 'Chef d'Orchestre' et composants associés sauvegardés.")
    print("Vous pouvez maintenant intégrer ce modèle dans aicompress/core.py (get_compression_settings).")

if __name__ == '__main__':
    if not os.path.exists(DATASET_CSV_PATH):
        print(f"Le fichier '{DATASET_CSV_PATH}' est introuvable.")
        print("Veuillez d'abord le générer en exécutant 'python create_decision_dataset.py'")
        print("après avoir peuplé le dossier 'dataset_source_files'.")
    else:
        train_orchestrator_model()