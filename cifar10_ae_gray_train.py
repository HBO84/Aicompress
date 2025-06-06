# cifar10_ae_gray_train.py

import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf

# AJOUT: Importer BatchNormalization
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    UpSampling2D,
    BatchNormalization,
)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.datasets import cifar10
import numpy as np
import matplotlib.pyplot as plt

print(f"TensorFlow Version: {tf.__version__}")
# ... (code de détection GPU et memory growth comme avant) ...
physical_devices = tf.config.list_physical_devices("GPU")
if physical_devices:
    print(f"GPU(s) détectée(s): {physical_devices}")
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
else:
    print("Aucun GPU détecté. L'entraînement sera sur CPU et peut être TRES long.")

# 1. Charger et Prétraiter les données CIFAR-10 (inchangé)
print("Chargement données CIFAR-10...")
(x_train_rgb, _), (x_test_rgb, _) = cifar10.load_data()
print("Conversion en niveaux de gris...")
x_train_gray = tf.image.rgb_to_grayscale(x_train_rgb).numpy()
x_test_gray = tf.image.rgb_to_grayscale(x_test_rgb).numpy()
x_train = x_train_gray.astype("float32") / 255.0
x_test = x_test_gray.astype("float32") / 255.0
print(f"Forme entraînement: {x_train.shape}, Forme test: {x_test.shape}")

# 2. Définir l'Architecture de l'Autoencodeur (V4.3b - BatchNormalization)

input_img_shape = (32, 32, 1)
input_img = Input(shape=input_img_shape, name="input_image_32x32")

# Encodeur V4.3b
x = Conv2D(64, (3, 3), padding="same", name="enc_conv1")(input_img)
x = BatchNormalization(name="enc_bn1")(x)
x = tf.keras.layers.ReLU(name="enc_relu1")(x)
x = MaxPooling2D((2, 2), padding="same", name="enc_pool1")(x)  # 16x16x64

x = Conv2D(32, (3, 3), padding="same", name="enc_conv2")(x)
x = BatchNormalization(name="enc_bn2")(x)
x = tf.keras.layers.ReLU(name="enc_relu2")(x)
x = MaxPooling2D((2, 2), padding="same", name="enc_pool2")(x)  # 8x8x32

x = Conv2D(16, (3, 3), padding="same", name="enc_conv3")(x)
x = BatchNormalization(name="enc_bn3")(x)
x = tf.keras.layers.ReLU(name="enc_relu3")(x)
encoded_volume = MaxPooling2D((2, 2), padding="same", name="encoded_latent_volume")(
    x
)  # 4x4x16

print(f"Forme du volume encodé (code latent): {encoded_volume.shape}")
latent_shape = encoded_volume.shape[1:]

# Décodeur V4.3b (Symétrique)
x_dec = Conv2D(16, (3, 3), padding="same", name="dec_conv0")(
    encoded_volume
)  # Couche initiale pour correspondre à la sortie de l'encodeur avant UpSampling
x_dec = BatchNormalization(name="dec_bn0")(x_dec)
x_dec = tf.keras.layers.ReLU(name="dec_relu0")(x_dec)

x_dec = UpSampling2D((2, 2), name="dec_upsample1")(x_dec)  # 8x8x16
x_dec = Conv2D(32, (3, 3), padding="same", name="dec_conv1")(x_dec)
x_dec = BatchNormalization(name="dec_bn1")(x_dec)
x_dec = tf.keras.layers.ReLU(name="dec_relu1")(x_dec)

x_dec = UpSampling2D((2, 2), name="dec_upsample2")(x_dec)  # 16x16x32
x_dec = Conv2D(64, (3, 3), padding="same", name="dec_conv2")(x_dec)
x_dec = BatchNormalization(name="dec_bn2")(x_dec)
x_dec = tf.keras.layers.ReLU(name="dec_relu2")(x_dec)

x_dec = UpSampling2D((2, 2), name="dec_upsample3")(x_dec)  # 32x32x64
decoded_output = Conv2D(
    1, (3, 3), activation="sigmoid", padding="same", name="decoder_output_conv"
)(
    x_dec
)  # 32x32x1

autoencoder_cifar_gray = Model(
    input_img, decoded_output, name="autoencoder_cifar10_gray_v4_3b"
)
autoencoder_cifar_gray.compile(optimizer="adam", loss="mse")
print("\n--- Architecture de l'Autoencodeur CIFAR-10 Gris (V4.3b) ---")
autoencoder_cifar_gray.summary()

# 3. Entraîner l'Autoencodeur
print("\nDébut de l'entraînement de l'autoencodeur (V4.3b) pour CIFAR-10 (gris)...")
EPOCHS = (
    100  # Augmenter encore, la BN peut aider à entraîner plus longtemps sans diverger.
)
# Soyez patient, ou réduisez si c'est trop long pour un test.
BATCH_SIZE = 256  # Augmenter la taille du batch peut aussi aider avec la BN
history = None
try:
    history = autoencoder_cifar_gray.fit(
        x_train,
        x_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=True,
        validation_data=(x_test, x_test),
    )
    print("Entraînement terminé.")

    # 4. Créer et Sauvegarder les modèles Encodeur et Décodeur séparés
    encoder_cifar10_gray = Model(
        input_img, encoded_volume, name="encoder_cifar10_gray_v4_3b"
    )

    decoder_input_standalone = Input(
        shape=latent_shape, name="decoder_input_cifar10_gray_v4_3b"
    )
    current_layer_output = decoder_input_standalone
    start_decoding = False
    for layer in autoencoder_cifar_gray.layers:
        if layer.name == "encoded_latent_volume":
            start_decoding = True
            continue
        if start_decoding:
            current_layer_output = layer(current_layer_output)
            if layer.name == "decoder_output_conv":
                break
    decoder_output_standalone = current_layer_output
    decoder_cifar10_gray = Model(
        decoder_input_standalone,
        decoder_output_standalone,
        name="decoder_cifar10_gray_v4_3b",
    )

    # ... (Summaries et sauvegarde des modèles avec les nouveaux noms _v4_3b)
    print("\n--- Architecture de l'Encodeur CIFAR-10 Gris (V4.3b) ---")
    encoder_cifar10_gray.summary()
    print("\n--- Architecture du Décodeur CIFAR-10 Gris (V4.3b) ---")
    decoder_cifar10_gray.summary()

    ENCODER_SAVE_PATH = "encoder_cifar10_gray_v4_3b.keras"
    DECODER_SAVE_PATH = "decoder_cifar10_gray_v4_3b.keras"
    print(f"\nSauvegarde encodeur: {ENCODER_SAVE_PATH}")
    encoder_cifar10_gray.save(ENCODER_SAVE_PATH)
    print(f"Sauvegarde décodeur: {DECODER_SAVE_PATH}")
    decoder_cifar10_gray.save(DECODER_SAVE_PATH)
    print("Modèles V4.3b sauvegardés.")

except Exception as e:
    print(f"Erreur entraînement/sauvegarde modèles V4.3b: {e}")
    import traceback

    print(traceback.format_exc())

# 5. Évaluer et Visualiser les résultats (si entraînement OK)
if (
    history
    and "decoder_cifar10_gray" in locals()
    and "encoder_cifar10_gray" in locals()
):
    # ... (Visualisation comme avant, mais avec les nouveaux noms de fichiers pour les images sauvegardées) ...
    print("\nPrédiction et visualisation (V4.3b)...")
    decoded_imgs = autoencoder_cifar_gray.predict(x_test)
    n = 10
    plt.figure(figsize=(20, 4))
    for i in range(n):
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(x_test[i].reshape(32, 32), cmap="gray")
        ax.set_title("Originale", fontsize=8)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs[i].reshape(32, 32), cmap="gray")
        ax.set_title("Reconstruite", fontsize=8)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.suptitle("CIFAR-10 Gris (V4.3b + BN): Originales vs. Reconstruites")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("cifar10_gray_reconstructions_v4_3b.png")
    print("Image reconstructions sauvegardée: cifar10_gray_reconstructions_v4_3b.png")
    plt.show()
    plt.figure()
    plt.plot(history.history["loss"], label="Perte (train)")
    plt.plot(history.history["val_loss"], label="Perte (val)")
    plt.title("Courbe Perte (CIFAR-10 Gris Autoencodeur V4.3b + BN)")
    plt.ylabel("Perte")
    plt.xlabel("Époque")
    plt.legend(loc="upper right")
    plt.savefig("cifar10_gray_loss_curve_v4_3b.png")
    print("Courbe perte sauvegardée: cifar10_gray_loss_curve_v4_3b.png")
    plt.show()
else:
    print("\nEntraînement/sauvegarde échoué, visualisation sautée.")
print("\nScript cifar10_ae_gray_train.py (V4.3b) terminé.")
