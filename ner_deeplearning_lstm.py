import time
import os
import json
import random
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.callbacks import ModelCheckpoint
from keras.layers import Merge
from keras.layers import TimeDistributed
from keras.layers import Dense
from keras.layers.core import Activation
from sklearn.metrics import confusion_matrix, accuracy_score
from keras.models import model_from_json
from keras.preprocessing.sequence import pad_sequences
from collections import Counter
# filepath = "data/"
model_file_name = "model.json"
weight_filename = "weights-improvement-47-0.8332.hdf5"

class NER_Model_Builder:
    def __init__(self):
        start_time = time.time()
        self.char_to_int={'+': 10, 'x': 50, '2': 17, 'r': 44, 'l': 38, ',': 11, '/': 14, 'o': 41, 'n': 40, 'p': 42, 'c': 29, '#': 2, 'y': 51, 'j': 36, '&': 5, '*': 9, ' ': 1, 't': 46, 'd': 30, 'a': 27, 'f': 32, 'q': 43, '5': 20, 'v': 48, 'undetected': 0, 'h': 34, 's': 45, '$': 3, ':': 25, '3': 18, '%': 4, 'z': 52, '0': 15, '8': 23, '1': 16, '7': 22, '6': 21, '(': 7, 'm': 39, 'b': 28, "'": 6, '9': 24, 'e': 31, '.': 13, 'u': 47, '4': 19, '@': 26, 'i': 35, 'k': 37, 'w': 49, 'g': 33, ')': 8, '-': 12}
        # if os.path.isfile(filepath + "char_to_int") and os.stat(filepath + "char_to_int").st_size > 0:
        #     self.char_to_int = json.load(open(filepath + "char_to_int"))
        self.int_to_char = {'12': '-', '47': 'u', '38': 'l', '21': '6', '26': '@', '48': 'v', '13': '.', '51': 'y', '16': '1', '28': 'b', '20': '5', '10': '+', '30': 'd', '42': 'p', '43': 'q', '27': 'a', '15': '0', '5': '&', '49': 'w', '22': '7', '0': 'undetected', '52': 'z', '25': ':', '33': 'g', '24': '9', '18': '3', '3': '$', '36': 'j', '17': '2', '37': 'k', '29': 'c', '19': '4', '8': ')', '1': ' ', '31': 'e', '7': '(', '32': 'f', '11': ',', '6': "'", '45': 's', '41': 'o', '9': '*', '44': 'r', '14': '/', '4': '%', '35': 'i', '2': '#', '50': 'x', '34': 'h', '40': 'n', '23': '8', '46': 't', '39': 'm'}

        self.ind2label = {'1': 'o', '2': 'skill', '3': 'role'}
        self.label2ind = {'skill': 2, 'o': 1, 'role': 3}
        self.maxlen = 121

        self.batch_size = 32
        self.max_features = len(self.char_to_int)
        self.embedding_size = 128
        self.hidden_size = 32
        self.out_size = len(self.label2ind) + 1
        self.max_label = max(self.label2ind.values()) + 1
        self.num_epochs = 100
        print("initialisation time for NER Builder: %s seconds." % (time.time() - start_time))



    def get_training_data(self):
        all_x = []
        all_y = []
        skills_list = json.load(open("skills_data"))
        all_x,all_y = self.convert_list_string_to_char_vec(skills_list,"skill",all_x,all_y)
        role_list = json.load(open("role_data"))
        all_x, all_y = self.convert_list_string_to_char_vec(role_list, "role", all_x, all_y)
        others = json.load(open("other_data"))
        others = random.sample(others, int(len(skills_list) * 1.5))
        all_x, all_y = self.convert_list_string_to_char_vec(others, "o", all_x, all_y)
        return all_x,all_y

    def convert_list_string_to_char_vec(self,list_string,tag_label,input_x,input_y):
        for ls in list_string:
            if self.maxlen:
                if len(ls)>self.maxlen:
                    new_ls = ls[:self.maxlen]
                else:
                    new_ls = ls
            else:
                new_ls = ls
            new_x = [self.char_to_int[tmp_s.lower()] if tmp_s.lower() in self.char_to_int.keys() else self.char_to_int["undetected"] for tmp_s in new_ls.strip()]
            input_x.append(new_x)
            input_y.append([tag_label]*len(new_ls))
        return input_x,input_y

    def encode(self, x, n):
        result = np.zeros(n)
        result[x] = 1
        return result

    def train_model(self):
        program_start_time = time.time()
        all_x,all_y = self.get_training_data()
        print("data fetch time: %s seconds." % (time.time() - program_start_time))
        (X_train_f, X_test_f, X_train_b, X_test_b, y_train, y_test) = self.convert_training_data_to_pad_format(all_x,all_y)
        print('Training and testing tensor shapes:')
        print(X_train_f.shape, X_test_f.shape, X_train_b.shape, X_test_b.shape, y_train.shape, y_test.shape)
        print("data split time: %s seconds." % (time.time() - program_start_time))
        model = self.get_model_structure()
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        callbacks_list = self.get_callback_list()
        print("model fetch time: %s seconds." % (time.time() - program_start_time))
        model.fit([X_train_f, X_train_b], y_train, epochs=self.num_epochs, batch_size=self.batch_size,
                  callbacks=callbacks_list, shuffle=True, validation_data=([X_test_f, X_test_b], y_test))
        print("model fit time: %s seconds." % (time.time() - program_start_time))
        score = model.evaluate([X_test_f, X_test_b], y_test, batch_size=self.batch_size)
        print('Raw test score:', score)
        pr = model.predict_classes([X_train_f, X_train_b])
        yh = y_train.argmax(2)
        fyh, fpr = self.score(yh, pr)
        print('Training accuracy:', accuracy_score(fyh, fpr))
        print('Training confusion matrix:')
        print(confusion_matrix(fyh, fpr))
        pr = model.predict_classes([X_test_f, X_test_b])
        yh = y_test.argmax(2)
        fyh, fpr = self.score(yh, pr)
        print('Testing accuracy:', accuracy_score(fyh, fpr))
        print('Testing confusion matrix:')
        print(confusion_matrix(fyh, fpr))
        if not ((os.path.isfile(model_file_name) and os.stat(model_file_name).st_size > 0)):
            self.save_model(model)

    def save_model(self,model):
        model_json = model.to_json()
        with open(model_file_name, "w") as json_file:
            json_file.write(model_json)

    def get_callback_list(self):
        w_filename =  "weights-improvement-{epoch:02d}-{val_loss:.4f}.hdf5"
        checkpoint = ModelCheckpoint(w_filename, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        return [checkpoint]

    def convert_training_data_to_pad_format(self,all_x,all_y):
        X_enc = all_x
        X_enc_reverse = [[c for c in reversed(x)] for x in X_enc]
        max_label = max(self.label2ind.values()) + 1
        y_enc = [[0] * (self.maxlen - len(ey)) + [self.label2ind[c] for c in ey] for ey in all_y]
        y_enc = [[self.encode(c, max_label) for c in ey] for ey in y_enc]
        X_enc_f = pad_sequences(X_enc, maxlen=self.maxlen)
        X_enc_b = pad_sequences(X_enc_reverse, maxlen=self.maxlen)
        y_enc = pad_sequences(y_enc, maxlen=self.maxlen)
        return train_test_split(X_enc_f, X_enc_b, y_enc, random_state=42)

    def get_model_structure(self):
        if os.path.isfile(model_file_name) and os.stat(model_file_name).st_size > 0:
            json_file = open(model_file_name, 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            model = model_from_json(loaded_model_json)
            if os.path.isfile(weight_filename) and os.stat(weight_filename).st_size > 0:
                model.load_weights(weight_filename)
        else:
            #forward mode
            model_forward = Sequential()
            model_forward.add(Embedding(self.max_features, self.embedding_size, input_length=self.maxlen, mask_zero=True))
            model_forward.add(LSTM(self.hidden_size, return_sequences=True))
            #backward model
            model_backward = Sequential()
            model_backward.add(Embedding(self.max_features, self.embedding_size, input_length=self.maxlen, mask_zero=True))
            model_backward.add(LSTM(self.hidden_size, return_sequences=True))
            #bidirectional lstm embedded model
            model = Sequential()
            model.add(Merge([model_forward, model_backward], mode='concat'))
            model.add(TimeDistributed(Dense(self.out_size)))
            model.add(Activation('softmax'))
        return model

    def score(self, yh, pr):
        coords = [np.where(yhh > 0)[0][0] for yhh in yh]
        yh = [yhh[co:] for yhh, co in zip(yh, coords)]
        ypr = [prr[co:] for prr, co in zip(pr, coords)]
        fyh = [c for row in yh for c in row]
        fpr = [c for row in ypr for c in row]
        return fyh, fpr
    def get_categories_of_unmapped(self,X_enc):
        try:
            detector_op = {}
            X_enc_reverse = [[c for c in reversed(x)] for x in X_enc]
            X_enc_f = pad_sequences(X_enc, maxlen=self.maxlen)
            X_enc_b = pad_sequences(X_enc_reverse, maxlen=self.maxlen)
            pred = self.loaded_model.predict_classes([X_enc_f, X_enc_b])
            for i, p in enumerate(pred):
                tmp_p = p[(self.maxlen - len(X_enc[i])):]
                new_p = Counter([self.ind2label[str(m)] for m in tmp_p])
                input_q = "".join([self.int_to_char[str(t)] for t in X_enc[i]])
                if "o" in new_p.keys() and (("skill" in new_p.keys() and new_p["o"] == new_p["skill"]) or (
                        "role" in new_p.keys() and new_p["o"] == new_p["role"])):
                    output_q = "o"
                else:
                    output_q = max(new_p, key=new_p.get)
                detector_op[input_q] = output_q
            return detector_op
        except Exception as e:
            print("NER_Detector get_categories_of_unmapped error : ", str(e))
            raise Exception(str(e))

    def convert_txt_to_vec(self,txt):
        try:
            new_x = []
            if len(txt)>self.maxlen:
                txt = txt[:self.maxlen]
            for c in txt:
                try:
                    new_x.append(self.char_to_int[c])
                except:
                    new_x.append(self.char_to_int["undetected"])
            return new_x
        except Exception as e:
            print("NER_Detector convert_txt_to_vec error : ",str(e))
            raise Exception(str(e))
    def categorize_txt(self,query_list):
        json_file = open(os.path.join('model.json'), 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.loaded_model = model_from_json(loaded_model_json)
        self.loaded_model.load_weights(weight_filename)
        self.loaded_model.compile(loss='categorical_crossentropy', optimizer='adam')
        X_enc = []
        for query in query_list:
            q = query.lower().replace('''"''', '''''').replace("_", " ").strip()
            X_enc.append(self.convert_txt_to_vec(q))
        tmp_op = self.get_categories_of_unmapped(X_enc)
        return tmp_op
if __name__ == "__main__":
    obj = NER_Model_Builder()
    # obj.train_model()
    print(obj.categorize_txt(['java']))