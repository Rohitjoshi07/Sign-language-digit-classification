from model_files.model_functions import model_cnn, get_data, save_model
import  os
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping


def main():
    # data
    trainx, valx, trainy, valy = get_data("None")

    # model
    model = model_cnn()

    # datagenerator
    '''
    datagen = ImageDataGenerator(
        rotation_range=14,
        width_shift_range=0.12,
        height_shift_range=0.12,
        zoom_range=0.12,

                )
    datagen.fit(trainx)

    '''
    # checkpoints
    checkpoints = []
    if not os.path.exists('Data/Checkpoints/'):
        os.makedirs('Data/Checkpoints/')
    checkpoints.append(ModelCheckpoint('Data/Checkpoints/best_weights.h5', monitor='val_loss', verbose=0,
                                       save_best_only=True, save_weights_only=True, mode='auto', save_freq='epoch'))

    checkpoints.append(TensorBoard(log_dir='Data/Checkpoints/./logs', histogram_freq=1, write_graph=True,
                                   write_images=True, embeddings_freq=0, embeddings_layer_names=None,
                                   embeddings_metadata=None))

    earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
    history = model.fit(trainx, trainy, epochs=100, validation_data=(valx, valy),
                        callbacks=earlyStopping)

    print("completed training!!!!!!!")
    # save model
    save_model(model)

    return history

if __name__=='__main__':
    history = main()