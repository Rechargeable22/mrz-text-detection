import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, \
    BatchNormalization, Dropout, Lambda
from matplotlib import pyplot as plt

from src.image_loading import sample_image_generator
from src.training_data_generator import sample_generator


def encoder_block(x, filters, kernel_size=3):
    x = Conv2D(filters=filters, kernel_size=kernel_size, activation='relu', kernel_initializer='he_normal',
               padding='same')(x)
    x = Conv2D(filters=filters, kernel_size=kernel_size, activation='relu', kernel_initializer='he_normal',
               padding='same')(x)
    p = MaxPooling2D(2)(x)
    return p, x


def decoder_block(x, skip_connection_features, filters, kernel_size=3):
    x = Conv2DTranspose(filters=filters, kernel_size=3, strides=2, padding='same')(x)
    # Blog about deconvolution vs conv_transpose https: // distill.pub / 2016 / deconv - checkerboard /
    x = concatenate([x, skip_connection_features])
    x = Conv2D(filters=filters, kernel_size=kernel_size, activation='relu', kernel_initializer='he_normal',
               padding='same')(x)
    x = Conv2D(filters=filters, kernel_size=kernel_size, activation='relu', kernel_initializer='he_normal',
               padding='same')(x)
    return x


def unet(img_channels):
    # Shallow U-Net which is enough for detecting angle brackets as they are a relatively small feature
    inputs = Input((None, None, img_channels))
    x = inputs

    p_1, skip_1 = encoder_block(x, 16, 3)
    p_2, skip_2 = encoder_block(p_1, 32, 3)

    # bottleneck
    b = Conv2D(filters=64, kernel_size=3, activation='relu', kernel_initializer='he_normal', padding='same')(p_2)
    b = Conv2D(filters=64, kernel_size=3, activation='relu', kernel_initializer='he_normal', padding='same')(b)

    up_2 = decoder_block(b, skip_2, 32, 3)
    up_1 = decoder_block(up_2, skip_1, 16, 3)

    out = Conv2D(filters=2, kernel_size=1, activation='softmax', padding='same')(up_1)
    model = Model(inputs=inputs, outputs=out)
    return model


if __name__ == '__main__':
    keras.backend.clear_session()

    # Build model
    img_size = (512, 512)
    num_classes = 2
    epochs = 2
    steps_per_epoch = 5  # 15
    # model_name = "model_2_large_augmented_noise_15"
    # model_name = "model_shallow_30"
    # model_name = "model_shallow_flexible_input"
    model_name = "refactor"
    path_trained_model = f"../data/trained_models/{model_name}.h5"
    model = unet(1)
    model.summary()

    train_dataset = tf.data.Dataset.from_generator(sample_generator, output_signature=(
        tf.TensorSpec(shape=(*img_size, 1), dtype=tf.float32), tf.TensorSpec(shape=(*img_size, 2), dtype=tf.float32)))
    train_dataset = train_dataset.batch(4)
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    val_dataset = tf.data.Dataset.from_generator(sample_image_generator, output_signature=(
        tf.TensorSpec(shape=(*img_size, 1), dtype=tf.float32), tf.TensorSpec(shape=(*img_size, 2), dtype=tf.float32)))
    val_dataset = val_dataset.batch(4)
    val_dataset = val_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    model.compile(optimizer="adam", loss="binary_crossentropy",
                  metrics=[tf.keras.metrics.Accuracy(), tf.keras.metrics.IoU(num_classes=2, target_class_ids=[0])])

    # model = keras.models.load_model(path_trained_model)
    model_history = model.fit(train_dataset, epochs=epochs, steps_per_epoch=steps_per_epoch,
                              validation_data=val_dataset)
    model.save(path_trained_model)

    if True:
        print(model_history.history.keys())
        fig, ax = plt.subplots(3, 1, figsize=(10, 10))
        ax[0].set_title('Loss')
        ax[0].plot(model_history.history['loss'], label='train')
        ax[0].plot(model_history.history['val_loss'], label='validation')
        ax[0].set_xlabel('Epoch')
        ax[0].set_ylabel('Binary Crossentropy Loss')
        ax[0].legend()

        ax[1].set_title('IoU')
        ax[1].plot(model_history.history['io_u'], label='train')
        ax[1].plot(model_history.history['val_io_u'], label='validation')
        ax[1].set_xlabel('Epoch')
        ax[1].set_ylabel('IoU')
        ax[1].legend()

        ax[2].set_title('Accuracy')
        ax[2].plot(model_history.history['accuracy'], label='train')
        ax[2].plot(model_history.history['val_accuracy'], label='validation')
        ax[2].set_xlabel('Epoch')
        ax[2].set_ylabel('Accuracy')
        ax[2].legend()
        plt.savefig(f"../data/plots/{model_name}_loss.png")
        plt.show()

    fig, ax = plt.subplots(3, 3, figsize=(10, 10))
    for img, label in train_dataset.take(2):
        pred = model.predict(img)
        for ix in range(2):
            ax[ix][0].imshow(img[ix, :, :], cmap="gray")
            ax[ix][0].set_title('photo')
            ax[ix][1].imshow(label[ix, :, :, 0], cmap='gray')
            ax[ix][1].set_title('label')
            ax[ix][2].imshow(pred[ix, :, :, 0] >= pred[ix, :, :, 1], cmap='gray')
            # ax[ix][2].imshow(pred[ix, :, :, 0], cmap='gray')
            ax[ix][2].set_title('prediction')

    for img, label in val_dataset.take(2):
        pred = model.predict(img)
        for ix in range(0, 1):
            ax[ix + 2][0].imshow(img[ix, :, :], cmap="gray")
            ax[ix + 2][0].set_title('photo')
            ax[ix + 2][1].imshow(label[ix, :, :, 0], cmap='gray')
            ax[ix + 2][1].set_title('label')
            ax[ix + 2][2].imshow(pred[ix, :, :, 0] >= pred[ix, :, :, 1], cmap='gray')
            # ax[ix + 2][2].imshow(pred[ix, :, :, 0], cmap='gray')
            ax[ix + 2][2].set_title('prediction')
    fig.tight_layout()
    plt.savefig(f"../data/plots/{model_name}_samples.png")
    plt.show()
