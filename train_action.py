import os
import csv
import time
import json
import numpy as np
import seaborn as sns
from keras.metrics import AUC
import matplotlib.pyplot as plt
from keras.models import Sequential
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from keras.callbacks import ReduceLROnPlateau
from keras.layers import LSTM, Dense, Dropout, Input, Flatten

epochs = 50
n_hidden = 16
window_size = 10
num_keypoints = 17
batch_size = 1
classes = ["idle", "run", "walk"]
filename = 'LSTM_Action_WithoutLSTM_Dense128'
k_aug = 0

def augmentation(X, y, X_out, y_out, k, method):
    from tsaug import AddNoise
    X_aug = []
    print('[INFO] ' + method + '...')
    for i in range(int(len(X) * k)):
        X_aug.append(AddNoise(scale=0.01).augment(X[i]))
    print('[DONE] ' + method + '.')

    return np.vstack((X_out, X_aug)), np.append(y_out, y)

def get_training_data():
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.preprocessing import LabelEncoder


    X = np.load('normalized/Norm_MinMax_Dist3_%02d_X.npy' % window_size)
    y = np.load('normalized/Norm_MinMax_Dist3_%02d_y.npy' % window_size)

    print("[INFO] Augmenting...")

    X_augmented, y_augmented = X, y
    if k_aug:
        X_augmented, y_augmented = augmentation(X, y, X_augmented, y_augmented, k_aug, 'AddNoise')

    print(X_augmented.shape)
    le = LabelEncoder()
    le.fit(y_augmented)
    le_y = np.array(le.transform(y_augmented))
    le_y = le_y.reshape((-1, 1))

    ohe = OneHotEncoder()
    ohe.fit(le_y)

    y_augmented = ohe.transform(le_y).toarray()

    return X_augmented, y_augmented


X, y = get_training_data()


def trainer(dir, k, X_train, X_test, y_train, y_test):
    global mean_val_acc, mean_val_auc, mean_val_loss, mean_time, mean_cm
    dirname = os.path.join(dir, 'k' + str(k))
    os.mkdir(dirname)

    model = Sequential(name=filename)
    # model.add(LSTM(n_hidden, input_shape=(window_size-1, num_keypoints),
    #                 name='lstm_0',
    #                 activation='tanh',
    #                 return_sequences=False
    #                 ))
    model.add(Input(shape=(window_size-1, num_keypoints)))
    model.add(Flatten())
    model.add(Dense(128, activation='tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(3, activation='softmax'))
    model.compile(optimizer='adam',
                    loss='categorical_crossentropy', metrics=['accuracy', AUC(name="auc")])
    model.summary()

    rlr = ReduceLROnPlateau(
                monitor="val_loss", verbose=1, patience=3, factor=0.3, min_lr=1e-6)

    time_start = time.ctime()
    start_time = time.time()
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                        validation_data=(X_test, y_test),
                        callbacks=[rlr],
                        )
    end_time = time.time()

    model.save(dirname + '/' + filename + ".h5")
    eval_loss, eval_acc, eval_auc = model.evaluate(X_test, y_test)
    mean_val_acc += eval_acc
    mean_val_auc += eval_auc
    mean_val_loss += eval_loss
    mean_time += end_time - start_time
    print("\nTraining at %s" % time_start)
    print("Total run-time: %f seconds" % (end_time - start_time))
    print("Loss of the model is - ", eval_loss)
    print("Accuracy of the model is - ", eval_acc*100, "%")
    print("AUC of the model is - ", eval_auc*100, "%")

    # Save history of model
    # Save it under the form of a json file
    json.dump(str(history.history), open(dirname + '/history.json', 'w'))

    # Analysis after Model Training
    epochs_arr = [i for i in range(epochs)]
    train_acc = history.history['accuracy']
    train_auc = history.history['auc']
    train_loss = history.history['loss']

    val_acc = history.history['val_accuracy']
    val_auc = history.history['val_auc']
    val_loss = history.history['val_loss']

    plt.plot(epochs_arr, train_acc, 'go-', label='Training Accuracy')
    plt.plot(epochs_arr, val_acc, 'ro-', label='Validation Accuracy')
    plt.title('Training & Validation Accuracy')
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.savefig(dirname + '/' + "Acc_plot.png")
    plt.clf()

    plt.plot(epochs_arr, train_auc, 'go-', label='Training AUC')
    plt.plot(epochs_arr, val_auc, 'ro-', label='Validation AUC')
    plt.title('Training & Validation AUC')
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("AUC")
    plt.savefig(dirname + '/' + "AUC_plot.png")
    plt.clf()

    plt.plot(epochs_arr, train_loss, 'g-o', label='Training Loss')
    plt.plot(epochs_arr, val_loss, 'r-o', label='Validation Loss')
    plt.title('Testing Accuracy & Loss')
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig(dirname + '/' + "Loss_plot.png")
    plt.clf()

    y_true = np.argmax(y_test, axis=1)
    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=1)

    cm = confusion_matrix(y_true, y_pred)
    cm = cm / cm.astype(np.float).sum(axis=1)
    mean_cm = mean_cm + cm
    g = sns.heatmap(cm, annot=True, fmt='.2f', cmap="Blues",
                    xticklabels=classes, yticklabels=classes)
    g.set_xticklabels(g.get_xticklabels(), ha='center', rotation=0)
    g.set_yticklabels(g.get_yticklabels(), rotation=0)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    figure = g.get_figure()
    figure.savefig(dirname + '/' + "CM_plot.png", facecolor='w', transparent=False)
    plt.clf()

    with open(dir + '/logs.csv', 'a') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([eval_acc, eval_auc, eval_loss, cm,
                         '%f (s)' % (end_time - start_time)])


kf = KFold(n_splits=10, random_state=2021, shuffle=True)
k = 0
str_time = time.strftime('%Y%m%d_%H%M%S')
dirname = os.path.join('results', str_time + '_' + filename)
if not os.path.isdir(dirname):
    os.mkdir(dirname)

# Write header

with open(dirname + '/logs.csv', 'w') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['val_acc', 'val_auc', 'val_loss', 'cm', 'runtime'])

mean_val_acc = mean_val_auc = mean_val_loss = mean_time = 0
mean_cm = np.array([[0]*len(classes)]*len(classes))

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    trainer(dirname, k, X_train, X_test, y_train, y_test)
    k += 1

with open(dirname + '/logs.csv', 'a') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow([mean_val_acc/k, mean_val_auc/k,
                     mean_val_loss/k, mean_cm/k, mean_time/k])
print(filename)
print("Mean run-time:", mean_time/k)
print("Mean Loss of the model is - ", mean_val_loss/k)
print("Mean Accuracy of the model is - ", (mean_val_acc*100)/k)
print("Mean AUC of the model is - ", (mean_val_auc*100)/k)
print("Mean CM of the model is - ", mean_cm/k)