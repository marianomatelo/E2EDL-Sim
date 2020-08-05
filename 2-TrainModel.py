from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Input, concatenate
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, CSVLogger, EarlyStopping
import os
from Generator import DriveDataGenerator
from Cooking import checkAndCreateDir
import h5py


if __name__ == '__main__':

    print('=====================================================')
    print('                 STARTING TRAINING                   ')
    print('=====================================================')

    COOKED_DATA_DIR = 'data_cooked/'
    MODEL_OUTPUT_DIR = 'model'

    batch_size = 32

    train_dataset = h5py.File(os.path.join(COOKED_DATA_DIR, 'train.h5'), 'r')
    eval_dataset = h5py.File(os.path.join(COOKED_DATA_DIR, 'eval.h5'), 'r')
    test_dataset = h5py.File(os.path.join(COOKED_DATA_DIR, 'test.h5'), 'r')

    num_train_examples = train_dataset['image'].shape[0]
    num_eval_examples = eval_dataset['image'].shape[0]
    num_test_examples = test_dataset['image'].shape[0]

    # Reads data in chunks
    data_generator = DriveDataGenerator(rescale=1./255., horizontal_flip=True, brighten_range=0.4)
    train_generator = data_generator.flow\
        (train_dataset['image'], train_dataset['previous_state'], train_dataset['label'], batch_size=batch_size, zero_drop_percentage=0.95, roi=[76,135,0,255])
    eval_generator = data_generator.flow\
        (eval_dataset['image'], eval_dataset['previous_state'], eval_dataset['label'], batch_size=batch_size, zero_drop_percentage=0.95, roi=[76,135,0,255])

    [sample_batch_train_data, sample_batch_test_data] = next(train_generator)

    # Build network architecture
    image_input_shape = sample_batch_train_data[0].shape[1:]
    state_input_shape = sample_batch_train_data[1].shape[1:]
    activation = 'relu'

    #Create the convolutional stacks
    pic_input = Input(shape=image_input_shape)

    img_stack = Conv2D(16, (3, 3), name="convolution0", padding='same', activation=activation)(pic_input)
    img_stack = MaxPooling2D(pool_size=(2,2))(img_stack)
    img_stack = Conv2D(32, (3, 3), activation=activation, padding='same', name='convolution1')(img_stack)
    img_stack = MaxPooling2D(pool_size=(2, 2))(img_stack)
    img_stack = Conv2D(32, (3, 3), activation=activation, padding='same', name='convolution2')(img_stack)
    img_stack = MaxPooling2D(pool_size=(2, 2))(img_stack)
    img_stack = Flatten()(img_stack)
    img_stack = Dropout(0.2)(img_stack)

    #Inject the state input
    state_input = Input(shape=state_input_shape)
    merged = concatenate([img_stack, state_input])

    # Add a few dense layers to finish the model
    merged = Dense(64, activation=activation, name='dense0')(merged)
    merged = Dropout(0.2)(merged)
    merged = Dense(10, activation=activation, name='dense2')(merged)
    merged = Dropout(0.2)(merged)
    merged = Dense(1, name='output')(merged)

    adam = Nadam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model = Model(inputs=[pic_input, state_input], outputs=merged)
    model.compile(optimizer=adam, loss='mse')

    # If the model is near a minimum and the learning rate is too high, then the model will circle around that minimum without ever reaching it.
    # This callback will allow us to reduce the learning rate when the validation loss stops improving, allowing us to reach the optimal point.
    plateau_callback = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.0001, verbose=1)

    checkpoint_filepath = os.path.join(MODEL_OUTPUT_DIR, 'models', '{0}_model.{1}-{2}.h5'.
                                       format('model', '{epoch:02d}', '{val_loss:.7f}'))

    checkAndCreateDir(checkpoint_filepath)

    # This callback will save the model each time the validation loss improves.
    checkpoint_callback = ModelCheckpoint(checkpoint_filepath, save_best_only=True, verbose=1)

    # Logs each iteration
    csv_callback = CSVLogger(os.path.join(MODEL_OUTPUT_DIR, 'training_log.csv'))

    # This monitor will detect when the validation loss stops improving, and will stop the training process when that occurs.
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10, verbose=1)

    # List of callbacks to use
    callbacks = [plateau_callback, csv_callback, checkpoint_callback, early_stopping_callback]

    # Train the model
    history = model.fit(train_generator, steps_per_epoch=num_train_examples//batch_size, epochs=500, callbacks=callbacks,\
                       validation_data=eval_generator, validation_steps=num_eval_examples//batch_size, verbose=2)

    print('=====================================================')
    print('                TRAINING COMPLETED                   ')
    print('=====================================================')