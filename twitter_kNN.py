# Import the libraries

import tensorflow as tf
import numpy as np
import nltk
import csv

from nltk.data import path
# append your path for nltk data
path.append("C:\\Users\\andri\\AppData\\Roaming\\nltk_data")

# Load the data

file_path = '.\Data\\train.csv' # path for the data set
X, y2 = [], []

with open(file_path, 'rt') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='"')
    next(reader, None) # Skip header
    
    for row in reader:      
        y2.append(row[1])
        X.append(row[2])

y_real = []        
for i in y2:
    y_real.append(int(i))

# Making vector y one_hot
y = [] # one hot y
for i in range(len(y_real)):
    if y_real[i] == 0:
        y.append([1, 0])
    else:
        y.append([0, 1])

# Number of 0 and 1 classes
pozitivan, negativan = 0, 0

for y_ch in y_real:
    if y_ch == 0:
        negativan += 1
    else:
        pozitivan += 1

print('br. pozitivnih: ', pozitivan, '\nbr. negativnih: ', negativan)

# Split data into training, validtaiton and test set

train_len = int((len(X)/20) * 0.6)
validatioon_len = int((len(X)/20)*0.2 + train_len)
test_len = 2 * validatioon_len

X_train = X[:train_len]
Y_train = y[:train_len]

X_valid = X[train_len:validatioon_len]
Y_valid = y[train_len:validatioon_len]

X_test = X[validatioon_len:test_len]
Y_test = y[validatioon_len:test_len]

# Split sentences to tokens
from nltk.tokenize import sent_tokenize, word_tokenize

X_train_sent = []
X_valid_sent = []
X_test_sent = []

def split_to_sent(sent_array, x_array):
    for i in range(len(x_array)):
        sent_array.append(sent_tokenize(x_array[i]))

split_to_sent(X_train_sent, X_train)
split_to_sent(X_valid_sent, X_valid)
split_to_sent(X_test_sent, X_test)

# Split sentences into words

import re
from nltk.tokenize import regexp_tokenize

X_train_word, X_valid_word, X_test_word = [], [], []

# Funkcija za pronalazenje svih pozicija karaktera ch u stringu s
def findOccurences(s, ch):
    return [i for i, letter in enumerate(s) if letter == ch]

def clean_data(data_set_to_split, data_set):
    """
    data_set_to_split - set with sentences to split into words
    data_set - set with words
    """
    
    line = [] # jedan twit
    occurences = [] # lista sa pozicijama karaktera '@' u datoj recenici
    occurences_and = [] # lista sa pozicijama karaktera '&' u datoj recenici
    http_index = [] # lista sa pozicijama podstringa 'http' u datoj recenici
    usernames = [] # list of usernames and strings starting with '&' to remove
    http_list = [] # list of links to remove
    
    for x in data_set_to_split:
        duzina = len(x)
        for i in range(duzina):
            string = str(x[i]).strip()
            
            # Remove usernames and links
            occurences = findOccurences(string, '@')
            occurences_and = findOccurences(string, '&')
            http_index = [m.start() for m in re.finditer('(?=http)', string)]
            
            if occurences or occurences_and or http_index: # if any of the lists is not empty
                if occurences_and:
                    for index in occurences_and: # indexes of '&'
                        stop_index = string.find(' ' or '\n', index) # finds the first occurence of ' ' or '\n'
                        char_and = str(string[index:stop_index])
                        usernames.append(char_and)
                    occurences_and = []
                if occurences:
                    for index in occurences: # indexes of '@'
                        stop_index = string.find(' ' or '\n', index)
                        user_name = str(string[index:stop_index]) # find twitter username: @blah
                        usernames.append(user_name)
                    occurences = []
                if http_index:
                    for index in http_index:
                        stop_index = string.find(' ' or '\n', index)
                        link = str(string[index:stop_index])
                        http_list.append(link)
                    http_index = []

                for username_link in usernames or http_list:
                    if username_link in string:
                        string = string.replace(username_link, '')
                usernames = []
                http_list = []
            line.extend(regexp_tokenize(string, "[\w']+"))
        data_set.append(line)
        line = []    

## Train
clean_data(X_train_sent, X_train_word)
    
## Validation
clean_data(X_valid_sent, X_valid_word)

## Test
clean_data(X_test_sent, X_test_word)

from string import punctuation
from nltk.corpus import stopwords

# Get a list of stopwords for english
stopword_list = set(stopwords.words('english'))
stopwords_punctuation_list = set(stopword_list).union(set(punctuation))

# Removing punctuation and words that doesn't give huge meaning to sentence

from nltk.tokenize import wordpunct_tokenize, regexp_tokenize

X_final_train = []
X_final_valid = []
X_final_test = []

def token(word_array, final_array):
    for x in word_array:
        x = [w.lower() for w in x if w not in stopwords_punctuation_list and not w.isdigit() and len(w) > 1 and not w[0].isdigit() and len(w) > 2]
        final_array.append(x)
        
token(X_train_word, X_final_train)
token(X_valid_word, X_final_valid)
token(X_test_word, X_final_test)

# Postavljanje reci na korene

from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
from nltk.stem import WordNetLemmatizer

porter = LancasterStemmer()

def trim_to_root(final_array):
    for x in final_array:
        duzina = len(x)
        for i in range(duzina):
            x[i] = porter.stem(x[i])

trim_to_root(X_final_train)
trim_to_root(X_final_valid)
trim_to_root(X_final_test)

# Dodavanje reci iz trening seta u recnik i odredjivanje da li je rec 'pozitivna' ili 'negativna'

class_val = 0
vrednosti = {}

for x in X_final_train:
    duzina_x = len(x)
    for i in range(duzina_x):
        if x[i] not in vrednosti:
            vrednosti.setdefault(x[i], 0)
        else:
            if Y_train[class_val][0] == 1 and Y_train[class_val][1] == 0: # negativan[0,1], [1,0]
                vrednosti[x[i]] -= 1
            else:
                vrednosti[x[i]] += 1
    class_val += 1

# Predstavljanje svakog tvita u setu preko urenjenog para [pos, neg] 
# [pos, neg] - gde je pos broj pozitivnih reci u tvitu, a neg broj negativnih reci u tvitu

X_train_cor = [] # X sa koordinatama
X_valid_cor = []
X_test_cor = []

def assign_coord(final_array, cor_array):
    pos = 0
    neg = 0
    for x in final_array:
        duzina_x = len(x)
        for i in range(duzina_x):
            if x[i] not in vrednosti:
                pass
            elif vrednosti[x[i]] == 0:
                pass
            elif vrednosti[x[i]] > 0:
                pos += 1
            else:
                neg += 1
        cor_array.append([pos, neg])
        pos, neg = 0, 0

assign_coord(X_final_train, X_train_cor)
assign_coord(X_final_valid, X_valid_cor)
assign_coord(X_final_test, X_test_cor)

# Za trening i validaciju ne uzimaju se tvitovi za koje ne mozemo da odredimo klasu
# Brisanje tvita cija je vrednost [0,0], nema ni pozitivnih ni negativnih reci

X_train_final, X_valid_final, X_test_final = X_train_cor, [], []

def clear_tweet(coord_array, non_zero_array):
    for x in coord_array:
        if x[0] != 0 and x[1] != 0:
            non_zero_array.append(x)

clear_tweet(X_valid_cor, X_valid_final)
clear_tweet(X_test_cor, X_test_final)

# ======================================== #
#                   KNN                    #
# ======================================== #

# algoritam ce proci kroz k-ove iz k_valid i zapamtiti najbolju preciznost u acc_valid i k u K
k_valid = (3, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15) # skloni parne brojeve za ovaj data set
K_valid = 0 # k iz k_valid koje daje najbojle rezultate ce postati K_valid
acc_valid = 0

# input train vector
x1 = tf.placeholder(dtype=tf.float32, shape=[None, 2])

# input validation/test vector
x2 = tf.placeholder(dtype=tf.float32, shape=[2])

# number of classes k
K = tf.placeholder(dtype=tf.int32)

# Calculate L2 norm

# Euklidovo rastojanje
distance = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(x1, x2)), axis=1)) # 300x1
# weighted distance
w_distance = 1.0/distance

_, indices = tf.nn.top_k(tf.negative(w_distance), K)

k_nn_labels = tf.gather(Y_train, indices) # vraca vektor lepo mapiranih labela sa njihovim indeksima

predict = tf.argmax(tf.reduce_sum(k_nn_labels, axis=0), axis = 0)

###############
# Validation #
##############

with tf.Session() as sess:
    
    for k in k_valid:
        
        print('Validatioin for k = ', k)
        
        accuracy = 0.0

        for i in range(len(X_valid_final)):

            # nasa koju smo nasli
            pred_y = sess.run(predict, feed_dict={x1: X_train_final, x2: X_valid_final[i], K:k})

            # proveriti da li je pred_y broj
            if not pred_y.dtype == 'int64':
                pred_y = Y_train[i][tf.reduce_max(Y_train[i], axis=0).eval()]
            
            # realna klasa
            true_y = tf.argmax(Y_valid[i], axis=0).eval() # eval vrati poziciju najveceg elementa u Y_np_valid

            match = pred_y == true_y

            print("[Validation %3d] Prediction: %d, True Class: %d, Match: %d" % (i, pred_y, true_y, match))
            
            #print('X_np_train ', X_np_train[i], ' klasa: ', Y_train[i])
            #print('shape: ', )
            #print('\ndistanca: \n', sess.run(distance, feed_dict={x1: X_np_train, x2: X_np_valid[i], K:k}))
            #print('\nw_distanca: \n', sess.run(w_distance, feed_dict={x1: X_np_train, x2: X_np_valid[i], K:k}))
            #print('*'*25)
            
            if match:
                accuracy += 1.0 / len(X_valid_final)

            if accuracy > acc_valid:
                acc_valid = accuracy
                K_valid = k

        print('accuracy for k = ', k, ' -> ', accuracy)
    print('The best accuracy', acc_valid, ' is with k = ', K_valid)

#########
# Test #
########

# Test accuracy with best k from validatin in test set
with tf.Session() as sess:
    
    accuracy = 0.0
    
    for i in range(len(X_test_final)):

        # nasa koju smo nasli
        pred_y = sess.run(predict, feed_dict={x1: X_train_final, x2: X_test_final[i], K:K_valid})

        # proveriti da li je pred_y broj, ako nije postaviti da bude one klase za koju je distance dobio 0 # isdigit()
        if not pred_y.dtype == 'int64':
            pred_y = Y_train[i][tf.reduce_max(Y_train[i], axis=0).eval()]

        # realna klasa
        true_y = tf.argmax(Y_train[i], axis=0).eval() # Y_np_test treba da bude one_hot

        match = pred_y == true_y

        print("[Test %3d] Prediction: %d, True Class: %d, Match: %d" % (i, pred_y, true_y, match))

        if match:
            accuracy += 1.0 / len(X_test_final)

    print('For k = %d accuaracy on:' % (K_valid))
    print(' - validation set is ', acc_valid)
    print(' - test set is ', accuracy)

    print('Difference in accuracy: ', acc_valid - accuracy) if acc_valid > accuracy else print('difference in accuracy: ', accuracy - acc_valid)