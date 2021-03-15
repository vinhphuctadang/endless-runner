import os
import time
import json
import numpy as np
from utils import *
import seaborn as sns
from constants import *
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from tensorflow.keras.metrics import AUC
from tensorflow.keras.layers import Dense
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.layers import LSTM, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


class model_tools:

    def __init__(self, X=None, y=None, test_size=0.3, model_name='lstm', model_path=None, **kwargs):
        if X is not None:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=2021)
            self.X_train = X_train
            self.X_test = X_test
            self.y_train = y_train
            self.y_test = y_test

        if model_path:
            self.model = load_model(model_path)
        else:
            n_hidden = 16
            model = Sequential(name=model_name)
            if 'lstm' in model_name:
                model.add(LSTM(n_hidden, input_shape=(
                    window_size, num_keypoints)))
                # model.add(Flatten())
                # model.add(Dense(32, activation='relu'))
                model.add(Dropout(0.5))
            model.add(Dense(n_classes, activation='softmax'))
            self.model = model
            self.model.compile(optimizer='adam',
                               loss='categorical_crossentropy', metrics=['accuracy', AUC(name="auc")])

        model.summary()
        self.model_name = model_name

    def fit_and_save_model(self, es=False, mc=False, rlr=False, log=False):
        print('\n[INFO] training network...')
        callbacks = []
        if es:
            callbacks.append(EarlyStopping(monitor='val_accuracy',
                                           mode='auto', verbose=1, patience=patience, min_delta=.0001, baseline=None,
                                           restore_best_weights=True))
        if mc:
            callbacks.append(ModelCheckpoint(model_save_path,
                                             monitor='val_accuracy', mode='auto', verbose=1, save_best_only=True, save_weights_only=False,))
        if rlr:
            callbacks.append(ReduceLROnPlateau(
                monitor="val_loss", verbose=1, patience=3, factor=0.3, min_lr=1e-6))
        if log:
            if os.path.exists(log_path):
                os.remove(log_path)
            callbacks.append(CSVLogger(log_path, append=True, separator=';'))
        time_start = time.ctime()
        start_time = time.time()

        history = self.model.fit(self.X_train, self.y_train, epochs=epochs, batch_size=batch_size,
                                 validation_data=(self.X_test, self.y_test),
                                 callbacks=callbacks,
                                 )
        end_time = time.time()

        print("\n[INFO] evaluating network...")
        eval_loss, eval_acc, eval_auc = self.model.evaluate(
            self.X_test, self.y_test)

        print("\n[INFO] saving model...")
        self.model.save(model_save_path)
        if log:
            f = open(log_path, 'a')
            f.write('\nTraining at %s\n' % time_start)
            table = PrettyTable(['Features', 'Value'])
            table.align['Features'] = 'l'
            table.align['Value'] = 'l'
            table.add_row(['val_acc', eval_acc])
            table.add_row(['val_auc', eval_auc])
            table.add_row(['val_loss', eval_loss])
            table.add_row(['run-time', '%f (s)' %
                           (end_time - start_time)])
            f.write("{}\n".format(table))
            f.close()

        f = open(cfg_path, 'w')
        f.write('model_name=%s\n' % self.model_name)
        f.write('num_class=%d\n' % n_classes)
        f.write('classes={}\n'.format(classes))
        f.write('epochs=%d\n' % epochs)
        f.write('batch_size=%d\n' % batch_size)
        f.write('window_size=%d\n' % window_size)
        f.write('num_keypoints=%d\n' % num_keypoints)
        if es:
            f.write('patience=%d\n\n' % patience)
        f.write('model_save_path=%s\n' % model_save_path)
        f.write('log_path=%s\n' % log_path)
        f.write('cfg_path=%s\n\n' % cfg_path)
        f.close()
        print("Training at %s" % time_start)
        print(table)

        # Save history of model
        # Save it under the form of a json file
        json.dump(str(history.history), open(history_json_save_path, 'w'))

        # Analysis after Model Training
        epoch_arr = [i for i in range(epochs)]
        train_acc = history.history['accuracy']
        train_auc = history.history['auc']
        train_loss = history.history['loss']

        val_acc = history.history['val_accuracy']
        val_auc = history.history['val_auc']
        val_loss = history.history['val_loss']

        self.save_fig(epoch_arr, 'Accuracy', acc_plot_path, train_acc, val_acc)
        self.save_fig(epoch_arr, 'AUC', auc_plot_path, train_auc, val_auc)
        self.save_fig(epoch_arr, 'Loss', loss_plot_path, train_loss, val_loss)

        y_true = np.argmax(self.y_test, axis=1)
        y_pred = self.model.predict(self.X_test)
        y_pred = np.argmax(y_pred, axis=1)

        cm = confusion_matrix(y_true, y_pred)
        cm = cm / cm.astype(np.float).sum(axis=1)
        g = sns.heatmap(cm, annot=True, fmt='.2f', cmap="Blues",
                        xticklabels=classes, yticklabels=classes)
        g.set_xticklabels(g.get_xticklabels(), ha='center', rotation=0)
        g.set_yticklabels(g.get_yticklabels(), rotation=0)
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        figure = g.get_figure()
        figure.savefig(cm_plot_path, facecolor='w', transparent=False)
        os.system("say Your experiment has finished. Please collect your result")

    def save_fig(self, epoch_arr, name, filepath, train, val):
        plt.plot(epoch_arr, train, 'g-o', label='Training ' + name)
        plt.plot(epoch_arr, val, 'r-o', label='Validation ' + name)
        plt.title('Testing Accuracy & ' + name)
        plt.legend()
        plt.xlabel("Epochs")
        plt.ylabel(name)
        plt.savefig(filepath, facecolor='w', transparent=False)
        plt.clf()

    def plt_fig(self):
        plt.rcParams['figure.figsize'] = (18, 18)
        # plt.subplots_adjust(hspace=-0.05)
        plt.subplot(2, 2, 1)
        plt.axis('off')

        acc_plot = plt.imread(acc_plot_path)
        plt.imshow(acc_plot)
        plt.subplot(2, 2, 2)
        plt.axis('off')

        auc_plot = plt.imread(auc_plot_path)
        plt.imshow(auc_plot)
        plt.subplot(2, 2, 3)
        plt.axis('off')

        loss_plot = plt.imread(loss_plot_path)
        plt.imshow(loss_plot)

        plt.subplot(2, 2, 4)
        plt.axis('off')

        cm_plot = plt.imread(cm_plot_path)
        plt.imshow(cm_plot)

    def summary(self):
        self.model.summary()
