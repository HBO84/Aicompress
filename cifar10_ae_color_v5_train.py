# cifar10_ae_color_v5_train.py

import os
# Décommentez la ligne suivante pour forcer l'utilisation du CPU si le GPU pose problème
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1' 

import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization
from tensorflow.keras.models import Model, load_model # load_model ajouté pour un test optionnel
from tensorflow.keras.datasets import cifar10
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
    print("Aucun GPU détecté. L'entraînement sera sur CPU et sera EXTRÊMEMENT long.")

# 1. Charger et Prétraiter les données CIFAR-10 (en couleur)
print("Chargement des données CIFAR-10...")
(x_train_rgb, _), (x_test_rgb, _) = cifar10.load_data()

x_train = x_train_rgb.astype('float32') / 255.
x_test = x_test_rgb.astype('float32') / 255.

print(f"Forme des données d'entraînement (couleur): {x_train.shape}")
print(f"Forme des données de test (couleur): {x_test.shape}")

# 2. Définir l'Architecture de l'Autoencodeur (V5 - Couleur, Plus Profond/Large)
# (Cette section est identique à la version précédente qui donnait de "bons" résultats)
input_img_shape = (32, 32, 3)
input_img = Input(shape=input_img_shape, name="input_image_32x32_color")
# Encodeur V5
x = Conv2D(64, (3, 3), padding='same', name="enc_conv1")(input_img); x = BatchNormalization(name="enc_bn1")(x); x = tf.keras.layers.ReLU(name="enc_relu1")(x)
x = MaxPooling2D((2, 2), padding='same', name="enc_pool1")(x)
x = Conv2D(128, (3, 3), padding='same', name="enc_conv2")(x); x = BatchNormalization(name="enc_bn2")(x); x = tf.keras.layers.ReLU(name="enc_relu2")(x)
x = MaxPooling2D((2, 2), padding='same', name="enc_pool2")(x)
x = Conv2D(256, (3, 3), padding='same', name="enc_conv3")(x); x = BatchNormalization(name="enc_bn3")(x); x = tf.keras.layers.ReLU(name="enc_relu3")(x)
x = MaxPooling2D((2, 2), padding='same', name="enc_pool3")(x)
encoded_volume = Conv2D(64, (3, 3), activation='relu', padding='same', name="encoded_latent_volume")(x)
latent_shape = encoded_volume.shape[1:]
# Décodeur V5 (Symétrique)
x_dec = Conv2D(64, (3, 3), padding='same', name="dec_conv0")(encoded_volume); x_dec = BatchNormalization(name="dec_bn0")(x_dec); x_dec = tf.keras.layers.ReLU(name="dec_relu0")(x_dec)
x_dec = UpSampling2D((2, 2), name="dec_upsample1")(x_dec)
x_dec = Conv2D(256, (3, 3), padding='same', name="dec_conv1")(x_dec); x_dec = BatchNormalization(name="dec_bn1")(x_dec); x_dec = tf.keras.layers.ReLU(name="dec_relu1")(x_dec)
x_dec = UpSampling2D((2, 2), name="dec_upsample2")(x_dec)
x_dec = Conv2D(128, (3, 3), padding='same', name="dec_conv2")(x_dec); x_dec = BatchNormalization(name="dec_bn2")(x_dec); x_dec = tf.keras.layers.ReLU(name="dec_relu2")(x_dec)
x_dec = UpSampling2D((2, 2), name="dec_upsample3")(x_dec)
decoded_output = Conv2D(3, (3, 3), activation='sigmoid', padding='same', name="decoder_output_conv")(x_dec)

autoencoder_cifar_color = Model(input_img, decoded_output, name="autoencoder_cifar10_color_v5")
autoencoder_cifar_color.compile(optimizer='adam', loss='mse')
print("\n--- Architecture de l'Autoencodeur CIFAR-10 Couleur (V5) ---")
# autoencoder_cifar_color.summary() # Vous pouvez décommenter pour revoir le résumé

# 3. Entraîner l'Autoencodeur OU Charger un modèle pré-entraîné
# Mettez TRAIN_MODEL = False si vous avez déjà un modèle sauvegardé et voulez juste évaluer
TRAIN_MODEL = True 
EPOCHS = 100  # Ou le nombre d'époques que vous aviez utilisé pour de "bons" résultats
BATCH_SIZE = 256 
AUTOENCODER_MODEL_SAVE_PATH = "autoencoder_cifar10_color_v5.keras" # Modèle complet
ENCODER_SAVE_PATH = "encoder_cifar10_color_v5.keras"
DECODER_SAVE_PATH = "decoder_cifar10_color_v5.keras"

history = None
if TRAIN_MODEL:
    print("\nDébut de l'entraînement de l'autoencodeur (V5) pour CIFAR-10 Couleur...")
    try:
        history = autoencoder_cifar_color.fit(x_train, x_train,
                                    epochs=EPOCHS,
                                    batch_size=BATCH_SIZE,
                                    shuffle=True,
                                    validation_data=(x_test, x_test))
        print("Entraînement terminé.")
        autoencoder_cifar_color.save(AUTOENCODER_MODEL_SAVE_PATH) # Sauvegarder l'AE complet
        print(f"Autoencodeur complet sauvegardé: {AUTOENCODER_MODEL_SAVE_PATH}")

        # Créer et Sauvegarder les modèles Encodeur et Décodeur séparés
        encoder_cifar10_color = Model(input_img, encoded_volume, name="encoder_cifar10_color_v5")
        decoder_input_standalone = Input(shape=latent_shape, name="decoder_input_cifar10_color_v5")
        current_layer_output = decoder_input_standalone
        start_decoding = False
        for layer in autoencoder_cifar_color.layers:
            if layer.name == "encoded_latent_volume": start_decoding = True; continue
            if start_decoding:
                current_layer_output = layer(current_layer_output)
                if layer.name == "decoder_output_conv": break 
        decoder_output_standalone = current_layer_output
        decoder_cifar10_color = Model(decoder_input_standalone, decoder_output_standalone, name="decoder_cifar10_color_v5")
        
        print(f"Sauvegarde encodeur: {ENCODER_SAVE_PATH}"); encoder_cifar10_color.save(ENCODER_SAVE_PATH)
        print(f"Sauvegarde décodeur: {DECODER_SAVE_PATH}"); decoder_cifar10_color.save(DECODER_SAVE_PATH)
        print("Modèles encodeur et décodeur V5 pour CIFAR-10 couleur sauvegardés.")

    except Exception as e:
        print(f"Erreur pendant l'entraînement ou la sauvegarde des modèles CIFAR V5: {e}")
        import traceback; print(traceback.format_exc())
        history = None # Assurer que history est None si l'entraînement échoue
else: # Charger un modèle pré-entraîné si TRAIN_MODEL est False
    if os.path.exists(AUTOENCODER_MODEL_SAVE_PATH):
        print(f"Chargement de l'autoencodeur pré-entraîné depuis {AUTOENCODER_MODEL_SAVE_PATH}...")
        autoencoder_cifar_color = load_model(AUTOENCODER_MODEL_SAVE_PATH)
        print("Autoencodeur pré-entraîné chargé.")
        # Vous pourriez aussi charger encoder et decoder séparément si vous les avez sauvegardés ainsi
    else:
        print(f"Modèle pré-entraîné {AUTOENCODER_MODEL_SAVE_PATH} non trouvé. Veuillez entraîner d'abord.")
        autoencoder_cifar_color = None # Pour éviter des erreurs plus loin

# 4. Évaluer et Visualiser les résultats
if autoencoder_cifar_color is not None: # S'assurer que le modèle est chargé ou entraîné
    print("\nPrédiction sur le jeu de test pour évaluation...")
    decoded_imgs = autoencoder_cifar_color.predict(x_test)

    # === DÉBUT DES AJOUTS POUR PSNR ET SSIM ===
    # S'assurer que x_test et decoded_imgs sont des tf.Tensor pour les fonctions tf.image
    x_test_tf = tf.convert_to_tensor(x_test, dtype=tf.float32)
    decoded_imgs_tf = tf.convert_to_tensor(decoded_imgs, dtype=tf.float32)

    # max_val est 1.0 car nos images sont normalisées dans la plage [0, 1]
    psnr_values = tf.image.psnr(x_test_tf, decoded_imgs_tf, max_val=1.0)
    ssim_values = tf.image.ssim(x_test_tf, decoded_imgs_tf, max_val=1.0)
    
    avg_psnr = np.mean(psnr_values.numpy())
    avg_ssim = np.mean(ssim_values.numpy())

    print(f"\n--- Métriques d'Évaluation sur le Jeu de Test ---")
    print(f"PSNR Moyen: {avg_psnr:.2f} dB")
    print(f"SSIM Moyen: {avg_ssim:.4f}")
    # === FIN DES AJOUTS POUR PSNR ET SSIM ===

    # Visualisation
    n = 10  # Nombre d'images à afficher
    plt.figure(figsize=(20, 5)) 
    for i in range(n):
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(x_test[i]) 
        ax.set_title("Originale", fontsize=8)
        ax.get_xaxis().set_visible(False); ax.get_yaxis().set_visible(False)

        ax = plt.subplot(2, n, i + 1 + n)
        # Clipper les valeurs au cas où, pour un affichage correct avec imshow pour des flottants [0,1]
        plt.imshow(np.clip(decoded_imgs[i], 0.0, 1.0)) 
        ax.set_title("Reconstruite", fontsize=8)
        ax.get_xaxis().set_visible(False); ax.get_yaxis().set_visible(False)
    
    # Mettre à jour le titre pour inclure les métriques
    plt.suptitle(f"CIFAR-10 Couleur (V5): Reconstructions\nPSNR: {avg_psnr:.2f} dB, SSIM: {avg_ssim:.4f}")
    plt.tight_layout(rect=[0,0,1,0.93]) 
    plt.savefig("cifar10_color_reconstructions_v5_with_metrics.png")
    print("Image des reconstructions (avec métriques) sauvegardée.")
    plt.show()

    # Afficher la courbe de perte (loss), seulement si l'entraînement a eu lieu (history n'est pas None)
    if history and history.history: # Vérifier aussi que history.history n'est pas vide
        plt.figure()
        if 'loss' in history.history: plt.plot(history.history['loss'], label='Perte (entraînement)')
        if 'val_loss' in history.history: plt.plot(history.history['val_loss'], label='Perte (validation)')
        plt.title('Courbe de Perte (CIFAR-10 Couleur Autoencodeur V5)')
        plt.ylabel('Perte (MSE)'); plt.xlabel('Époque'); plt.legend(loc='upper right')
        plt.savefig("cifar10_color_loss_curve_v5_with_metrics.png")
        print("Courbe de perte sauvegardée.")
        plt.show()
    elif TRAIN_MODEL: # Si on a essayé d'entraîner mais history est vide
        print("L'historique de l'entraînement n'est pas disponible pour afficher la courbe de perte.")

else:
    print("\nModèle autoencodeur non disponible (ni entraîné ni chargé). Visualisation et évaluation sautées.")

print("\nScript cifar10_ae_color_v5_train.py terminé.")