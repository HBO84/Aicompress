# mnist_autoencoder_test.py

import os
# Décommentez la ligne suivante pour forcer l'utilisation du CPU si votre GPU n'est pas configuré
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1' 

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Reshape, Flatten
from tensorflow.keras.models import Model, load_model
import numpy as np
import matplotlib.pyplot as plt

print(f"TensorFlow Version: {tf.__version__}")
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    print(f"GPU(s) détectée(s): {physical_devices}")
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        print("Memory growth activé pour le(s) GPU(s).")
    except RuntimeError as e:
        print(f"Erreur lors de l'activation de memory growth : {e}")
else:
    print("Aucun GPU détecté par TensorFlow. Utilisation du CPU.")


# 1. Charger et préparer les données MNIST
(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

print(f"Forme des données d'entraînement: {x_train.shape}")
print(f"Forme des données de test: {x_test.shape}")

# 2. Définir l'architecture de l'Autoencodeur Convolutif
input_img_shape = (28, 28, 1)
input_img = Input(shape=input_img_shape, name="input_image")
# Encodeur
x = Conv2D(16, (3, 3), activation='relu', padding='same', name="enc_conv1")(input_img)
x = MaxPooling2D((2, 2), padding='same', name="enc_pool1")(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same', name="enc_conv2")(x)
x = MaxPooling2D((2, 2), padding='same', name="enc_pool2")(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same', name="enc_conv3")(x)
encoded_volume = MaxPooling2D((2, 2), padding='same', name="encoded_latent_volume")(x)
latent_shape = encoded_volume.shape[1:]
# Décodeur
x_autoencoder_decode = Conv2D(8, (3, 3), activation='relu', padding='same', name="dec_conv1")(encoded_volume)
x_autoencoder_decode = UpSampling2D((2, 2), name="dec_upsample1")(x_autoencoder_decode)
x_autoencoder_decode = Conv2D(8, (3, 3), activation='relu', padding='same', name="dec_conv2")(x_autoencoder_decode)
x_autoencoder_decode = UpSampling2D((2, 2), name="dec_upsample2")(x_autoencoder_decode)
x_autoencoder_decode = Conv2D(16, (3, 3), activation='relu', padding='same', name="dec_conv3")(x_autoencoder_decode)
x_autoencoder_decode = UpSampling2D((2, 2), name="dec_upsample3")(x_autoencoder_decode)
x_autoencoder_decode = tf.keras.layers.Resizing(28, 28, name="decoder_resize_to_28")(x_autoencoder_decode)
decoded_autoencoder_output = Conv2D(1, (3, 3), activation='sigmoid', padding='same', name="decoder_output_conv")(x_autoencoder_decode)
autoencoder = Model(input_img, decoded_autoencoder_output, name="autoencoder")
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
print("\n--- Architecture de l'Autoencodeur Complet ---")
autoencoder.summary()

# 3. Entraîner l'Autoencodeur
print("\nDébut de l'entraînement de l'autoencodeur...")
EPOCHS = 10 
BATCH_SIZE = 128
history = None
try:
    history = autoencoder.fit(x_train, x_train, epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=True, validation_data=(x_test, x_test))
    print("Entraînement terminé.")
except Exception as e:
    print(f"Erreur pendant l'entraînement : {e}")

if history:
    # 4. Créer les modèles Encodeur et Décodeur séparés
    encoder = Model(input_img, encoded_volume, name="encoder")
    decoder_standalone_input = Input(shape=latent_shape, name="decoder_input_standalone")
    current_layer_output = decoder_standalone_input
    start_decoding = False
    for layer in autoencoder.layers:
        if layer.name == "encoded_latent_volume": start_decoding = True; continue
        if start_decoding:
            current_layer_output = layer(current_layer_output)
            if layer.name == "decoder_output_conv": break
    decoder_standalone_output = current_layer_output
    decoder = Model(decoder_standalone_input, decoder_standalone_output, name="decoder")

    # --- NOUVEAU : Sauvegarde des modèles encodeur et décodeur ---
    ENCODER_MODEL_PATH = "mnist_encoder.keras"
    DECODER_MODEL_PATH = "mnist_decoder.keras"
    print(f"\nSauvegarde du modèle encodeur sous : {ENCODER_MODEL_PATH}")
    encoder.save(ENCODER_MODEL_PATH)
    print(f"Sauvegarde du modèle décodeur sous : {DECODER_MODEL_PATH}")
    decoder.save(DECODER_MODEL_PATH)
    # --- FIN NOUVEAU ---

    # 5. "Compresser", simuler la quantification, et "Décompresser" (comme avant)
    print("\n--- Démonstration Encodage/Décodage et Quantification Simulée (avec modèles originaux en mémoire) ---")
    num_images_to_test = 5 # Réduit pour des tests plus rapides
    test_images_sample = x_test[:num_images_to_test]
    compressed_representation_float = encoder.predict(test_images_sample)
    # ... (toute la logique de quantification et de déquantification, et la prédiction avec le décodeur original) ...
    # (Je vais la copier-coller pour la concision ici, mais elle reste la même que dans votre version précédente)
    min_val = np.min(compressed_representation_float); max_val = np.max(compressed_representation_float)
    quantized_latent_uint8 = np.zeros_like(compressed_representation_float, dtype=np.uint8)
    if max_val > min_val: 
        normalized_latent = (compressed_representation_float - min_val) / (max_val - min_val)
        quantized_latent_uint8 = (normalized_latent * 255).astype(np.uint8)
    elif max_val != 0 : quantized_latent_uint8[:] = np.clip(np.round(max_val), 0, 255).astype(np.uint8)
    
    dequantized_latent_float = np.zeros_like(quantized_latent_uint8, dtype=np.float32)
    if max_val > min_val:
        dequantized_latent_float = (quantized_latent_uint8.astype(np.float32) / 255.0) * (max_val - min_val) + min_val
    elif max_val != 0: dequantized_latent_float[:] = min_val
    
    reconstructed_images_from_quantized_original_model = decoder.predict(dequantized_latent_float)


    # --- NOUVEAU : Chargement des modèles sauvegardés et test ---
    print(f"\n--- Test avec les modèles Encodeur et Décodeur RECHARGÉS depuis disque ---")
    if os.path.exists(ENCODER_MODEL_PATH) and os.path.exists(DECODER_MODEL_PATH):
        print(f"Chargement du modèle encodeur depuis : {ENCODER_MODEL_PATH}")
        loaded_encoder = load_model(ENCODER_MODEL_PATH)
        print(f"Chargement du modèle décodeur depuis : {DECODER_MODEL_PATH}")
        loaded_decoder = load_model(DECODER_MODEL_PATH)

        # Test avec les modèles chargés
        compressed_loaded = loaded_encoder.predict(test_images_sample)
        
        # (Optionnel) Vérifier si le code latent est identique (il devrait l'être)
        # print(f"Codes latents originaux vs chargés sont proches : {np.allclose(compressed_representation_float, compressed_loaded)}")

        # Déquantification du code latent produit par l'encodeur chargé
        min_val_l = np.min(compressed_loaded); max_val_l = np.max(compressed_loaded)
        quant_l_uint8 = np.zeros_like(compressed_loaded, dtype=np.uint8)
        if max_val_l > min_val_l: 
            norm_l = (compressed_loaded - min_val_l) / (max_val_l - min_val_l)
            quant_l_uint8 = (norm_l * 255).astype(np.uint8)
        elif max_val_l != 0 : quant_l_uint8[:] = np.clip(np.round(max_val_l), 0, 255).astype(np.uint8)

        dequant_l_float = np.zeros_like(quant_l_uint8, dtype=np.float32)
        if max_val_l > min_val_l:
            dequant_l_float = (quant_l_uint8.astype(np.float32) / 255.0) * (max_val_l - min_val_l) + min_val_l
        elif max_val_l != 0: dequant_l_float[:] = min_val_l
            
        reconstructed_loaded_model = loaded_decoder.predict(dequant_l_float) # Utiliser le code latent déquantifié
        print("Prédictions avec les modèles chargés terminées.")

        # Visualisation (comparer originales, reconstruites par modèle en mémoire, reconstruites par modèle chargé)
        plt.figure(figsize=(num_images_to_test * 1.8, 7)) # Un peu plus de hauteur pour 4 lignes
        for i in range(num_images_to_test):
            # Originale
            ax = plt.subplot(4, num_images_to_test, i + 1)
            plt.imshow(test_images_sample[i].reshape(28, 28), cmap='gray')
            if i == 0: ax.set_ylabel("Originale", fontsize=7, rotation=0, labelpad=25, ha='right', va='center')
            ax.set_title(f"Img {i+1}", fontsize=8)
            ax.axis('off')

            # Reconstruite (modèle en mémoire, après quantification)
            ax = plt.subplot(4, num_images_to_test, i + 1 + num_images_to_test)
            plt.imshow(reconstructed_images_from_quantized_original_model[i].reshape(28, 28), cmap='gray')
            if i == 0: ax.set_ylabel("Recon.\n(quant,\nmém.)", fontsize=7, rotation=0, labelpad=25, ha='right', va='center')
            ax.axis('off')
            
            # Reconstruite (modèle chargé depuis disque, après quantification)
            ax = plt.subplot(4, num_images_to_test, i + 1 + 2 * num_images_to_test)
            plt.imshow(reconstructed_loaded_model[i].reshape(28, 28), cmap='gray')
            if i == 0: ax.set_ylabel("Recon.\n(quant,\nchargé)", fontsize=7, rotation=0, labelpad=25, ha='right', va='center')
            ax.axis('off')
            
            # Afficher le code latent (juste la première tranche pour la visualisation)
            ax = plt.subplot(4, num_images_to_test, i + 1 + 3 * num_images_to_test)
            if compressed_loaded.shape[-1] > 0: # S'il y a bien des canaux dans le code latent
                plt.imshow(compressed_loaded[i, :, :, 0], cmap='viridis') # Affiche la 1ère "tranche" du code latent 4x4
            if i == 0: ax.set_ylabel("Latent\n(1ère ch.\nchargé)", fontsize=7, rotation=0, labelpad=25, ha='right', va='center')
            ax.axis('off')


        plt.suptitle("Autoencodeur: Comparaison des Reconstructions (Mém vs. Chargé) et Code Latent")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        # plt.savefig("mnist_loaded_model_demo.png")
        plt.show()

    else:
        print("Fichiers de modèles sauvegardés non trouvés. Saut du test de chargement.")
    # --- FIN NOUVEAU ---


    # Afficher la courbe de perte (loss)
    if history:
        plt.figure()
        plt.plot(history.history['loss'], label='Perte (entraînement)')
        plt.plot(history.history['val_loss'], label='Perte (validation)')
        plt.title('Courbe de Perte du Modèle Autoencodeur')
        plt.ylabel('Perte')
        plt.xlabel('Époque')
        plt.legend(loc='upper right')
        # plt.savefig("mnist_loss_curve_autoencoder.png")
        plt.show()
else:
    print("\nL'entraînement de l'autoencodeur a échoué ou n'a pas produit d'historique, les étapes suivantes sont sautées.")

print("\nScript terminé.")