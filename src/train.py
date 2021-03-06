import utils
from data import Dataset, DataLoader
from model import IMG_WIDTH, IMG_HEIGHT, build_model1, build_model2, build_pre_train_model

from keras.optimizers import Adam
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, TensorBoard
from pprint import pprint


def train(batch_size=64, epochs=25):
    run_id = utils.generate_run_id()
    tb = TensorBoard(log_dir='../logs/{}'.format(run_id))
    model_path = '../models/{}.hdf5'.format(run_id)
    model_checkpoint = ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True)

    #model = build_model1()
    #model = build_pre_train_model(run_id="2017-10-07-11-33-19")
    model = build_model1()
    model.compile(optimizer=Adam(lr=1e-4), loss='mse', metrics=['mae'])
    model.summary()

    ds = Dataset()
    X_train, X_val, y_train, y_val = ds.train_val_split()
    train_gen = DataLoader(X_train, y_train, IMG_WIDTH, IMG_HEIGHT, batch_size)
    val_gen = DataLoader(X_val, y_val, IMG_WIDTH, IMG_HEIGHT, batch_size)

    model.fit_generator(train_gen, steps_per_epoch=len(train_gen), epochs=epochs,
                        validation_data=val_gen,
                        validation_steps=len(val_gen),
                        workers=8,
                        callbacks=[tb, model_checkpoint])
    evaluate(model_path)


def evaluate(model_path, batch_size=128):
    print("Evaluating model {}".format(model_path))
    model = load_model(model_path)
    ds = Dataset()
    X, y = ds.get_test_data()

    gen = DataLoader(X, y, IMG_WIDTH, IMG_HEIGHT, batch_size)
    metrics = model.evaluate_generator(gen, steps=len(gen), workers=8)
    pprint(dict(zip(model.metrics_names, metrics)))


if __name__ == '__main__':
    train()
    # evaluate('../models/2017-10-07-00-00-03.hdf5')
