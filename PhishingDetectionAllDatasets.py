import re # Regular Expression
import numpy as np
import pandas as pd
from joblib import parallel_backend
from keras.models import Sequential
from keras.layers import Dense
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from urllib.parse import urlparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, recall_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from tld import get_tld
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import keras.backend as K
from sklearn.metrics import accuracy_score, precision_score, recall_score

print("Dataset 1: 651191 entries\n"
      "Dataset 2: 11430 entries\n"
      "Dataset 3: 549344 entries")

DatasetIndex = 0
while DatasetIndex not in [1, 2, 3]:
    DatasetIndex = int(input("Select Dataset(1,2,3) : "))
    if DatasetIndex == 1:
        data = pd.read_csv('malicious_phish.csv', on_bad_lines='skip', encoding='utf-8')
    if DatasetIndex == 2:
        data = pd.read_csv('dataset_phishing.csv', on_bad_lines='skip', encoding='utf-8')
    if DatasetIndex == 3:
        data = pd.read_csv('good-bad.csv', on_bad_lines='skip', encoding='utf-8')



data.columns = data.columns.str.replace(';', '')
if(DatasetIndex == 3):
    print("For Dataset 3, Minimum 50k entries are required.")
num_entries_to_process = int(input("Enter the number of entries to process: "))
data = data.head(num_entries_to_process)
data['type'] = data['type'].str.replace(';', '')


replacement_dict = {'legitimate': 'good', 'benign': 'good', 'phishing': 'bad', 'malware': 'bad', 'defacement': 'bad'}
data['type'] = data['type'].replace(replacement_dict)


print("data")
print(data.head(10))


print("\ndata info")
data.info()


print("\ndata isnull sum")
data.isnull().sum()


print(data.columns)


allowed_values = ['good', 'bad']
print("\ndata type value counts")
countX = data['type'].value_counts()
print("count X: ", countX)


data = data[data['type'].isin(allowed_values)]


print("\ncount index")
x = countX.index
print(x)

print("\nchange url to blank")
data['url'] = data['url'].replace('www.', '', regex=True)
print(data)


# for good-bad dataset ("good-bad.csv")
print("\nchange classes to numbers")
rem = {"Category": {"good": 0, "bad": 1}}
data['Category'] = data['type']
data = data.replace(rem)

# Replace or drop non-finite values in 'Category' column
data['Category'] = data['Category'].replace([np.inf, -np.inf, np.nan],
                                            0)  # Replace with 0, but you can choose a different value
data['Category'] = data['Category'].astype(int)


print(data.head(20))

print("\nadd url length")
data['url_len'] = data['url'].apply(lambda x: len(str(x)))
print(data.head())



def process_tld(url):
    try:
        #         Extract the top level domain (TLD) from the URL given
        res = get_tld(url, as_object=True, fail_silently=False, fix_protocol=True)
        pri_domain = res.parsed_url.netloc
    except:
        pri_domain = None
    return pri_domain


data['domain'] = data['url'].apply(lambda i: process_tld(i))
print(data.head())

feature = ['@', '?', '-', '=', '.', '#', '%', '+', '$', '!', '*', ',', '//']
for a in feature:
    data[a] = data['url'].apply(lambda i: i.count(a))


def httpSecure(url):
    htp = urlparse(url).scheme  # It supports the following URL schemes: file , ftp , gopher , hdl ,
    # http , https ... from urllib.parse
    match = str(htp)
    if match == 'https':
        # print match.group()
        return 1
    else:
        # print 'No matching pattern found'
        return 0


data['https'] = data['url'].apply(lambda i: httpSecure(i))
print(data.head())


def digit_count(url):
    digits = 0
    for i in url:
        if i.isnumeric():
            digits = digits + 1
    return digits


data['digits'] = data['url'].apply(lambda i: digit_count(i))
print(data.head())


def letter_count(url):
    letters = 0
    for i in url:
        if i.isalpha():
            letters = letters + 1
    return letters


data['letters'] = data['url'].apply(lambda i: letter_count(i))



def Shortining_Service(url):
    match = re.search('bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs|'
                      'yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|snipurl\.com|'
                      'short\.to|BudURL\.com|ping\.fm|post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|'
                      'doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|t\.co|lnkd\.in|'
                      'db\.tt|qr\.ae|adf\.ly|goo\.gl|bitly\.com|cur\.lv|tinyurl\.com|ow\.ly|bit\.ly|ity\.im|'
                      'q\.gs|is\.gd|po\.st|bc\.vc|twitthis\.com|u\.to|j\.mp|buzurl\.com|cutt\.us|u\.bb|yourls\.org|'
                      'x\.co|prettylinkpro\.com|scrnch\.me|filoops\.info|vzturl\.com|qr\.net|1url\.com|tweez\.me|v\.gd|'
                      'tr\.im|link\.zip\.net',
                      url)
    if match:
        return 1
    else:
        return 0


data['Shortining_Service'] = data['url'].apply(lambda x: Shortining_Service(x))
print(data.head())


def having_ip_address(url):
    match = re.search(
        '(([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.'
        '([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\/)|'  # IPv4
        '(([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.'
        '([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\/)|'  # IPv4 with port
        '((0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\/)'  # IPv4 in hexadecimal
        '(?:[a-fA-F0-9]{1,4}:){7}[a-fA-F0-9]{1,4}|'
        '([0-9]+(?:\.[0-9]+){3}:[0-9]+)|'
        '((?:(?:\d|[01]?\d\d|2[0-4]\d|25[0-5])\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d|\d)(?:\/\d{1,2})?)', url)  # Ipv6
    if match:
        return 1
    else:
        return 0


data['having_ip_address'] = data['url'].apply(lambda i: having_ip_address(i))
print(data.head())

data['having_ip_address'].value_counts()


X = data.drop(['url', 'type', 'Category', 'domain'], axis=1)
y = data['Category']

print(x)
print()
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
print(X_train)
print(X_test)
print(y_train)
print(y_test)

X_train_np = X_train.values
y_train_np = y_train.values
X_test_np = X_test.values
y_test_np = y_test.values

# X_test and y_test are also validation sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# Models to train
models = [RandomForestClassifier, AdaBoostClassifier,
          KNeighborsClassifier, SGDClassifier, ExtraTreesClassifier, GaussianNB]

accuracy_test = []

recall_scores = []
def train_and_evaluate(model_class, X_train, X_test, y_train, y_test):
    model = model_class()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    # Calculate accuracy
    acc = accuracy_score(pred, y_test)
    accuracy_test.append(acc)  # Append accuracy to the list
    print('\033[01m{} Performance\033[0m'.format(model.__class__.__name__))
    print('Test Accuracy: \033[32m\033[01m{:.2f}%\033[30m\033[0m'.format(acc * 100))

    # Calculate recall
    rec = recall_score(y_test, pred)
    recall_scores.append(rec)  # Append recall to the list
    print('Test Recall: \033[32m\033[01m{:.2f}%\033[30m\033[0m'.format(rec * 100))

    # Print classification report
    print('\033[01mClassification Report\033[0m')
    print(model_class.__name__)
    print(classification_report(y_test, pred))

    # Print confusion matrix
    print('\033[01mConfusion Matrix\033[0m')
    cf_matrix = confusion_matrix(y_test, pred)
    print(cf_matrix)
    print('\033[31m###################- End -###################\033[0m')

    return cf_matrix


# Parallelize the training process using Joblib
with parallel_backend('threading', n_jobs=-1):
    confusion_matrices = [train_and_evaluate(model_class, X_train, X_test, y_train, y_test) for model_class in models]

plt.figure(figsize=(15, 8))
for i, cf_matrix in enumerate(confusion_matrices):
    plt.subplot(2, 3, i + 1)
    sns.heatmap(cf_matrix / np.sum(cf_matrix), annot=True, fmt='0.2%', cmap='Blues',
                xticklabels=['legitimate', 'phishing'],
                yticklabels=['legitimate', 'phishing'])
    plt.title(models[i].__name__)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

plt.tight_layout(pad=2)  # Increase or decrease the pad value as needed
plt.show()



# Convert DataFrame to numpy arrays
X_train_np = X_train.values
y_train_np = y_train.values
X_test_np = X_test.values
y_test_np = y_test.values

# Build a simple Sequential model
model = Sequential()
model.add(Dense(64, input_dim = X_train_np.shape[1], activation='relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

# Compile the model
def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision_val = true_positives / (possible_positives + K.epsilon())
    return precision_val

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', precision])

def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall_val = true_positives / (possible_positives + K.epsilon())
    return recall_val

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', precision, recall])

# Train the model
seqTrain = model.fit(X_train_np, y_train_np, epochs=10, batch_size=32, validation_data=(X_test_np, y_test_np))

# Evaluate the model
results = model.evaluate(X_test_np, y_test_np)

# Print loss, accuracy, precision, and recall
lossSeq = results[0]
accuracySeq = results[1]
precisionSeq = results[2]
recallSeq = results[3]
print(f'Test Loss: {lossSeq}')
print(f'Test Accuracy: {accuracySeq}')
print(f'Test Precision: {precisionSeq}')
print(f'Test Recall: {recallSeq}')
accuracy_test.append(accuracySeq)
recall_scores.append(recallSeq)

# Assuming you have predictions (y_pred) from your Keras model
y_pred = model.predict(X_test_np)
y_pred_classes = (y_pred > 0.5).astype(int)  # Convert probabilities to classes

# Compute confusion matrix
conf_matrix = confusion_matrix(y_test_np, y_pred_classes)

# Plot confusion matrix
plt.figure(figsize = (6, 6))
sns.heatmap(conf_matrix, annot = True, fmt = 'd', cmap = 'Blues', cbar = False,
            xticklabels = ['legitimate', 'phishing'],
            yticklabels = ['legitimate', 'phishing'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


def calculate_precision(conf_matrix):
    true_positives = conf_matrix[1, 1]  # Assuming class 1 corresponds to phishing (check index if different)
    false_positives = conf_matrix[0, 1]  # Assuming class 0 corresponds to legitimate (check index if different)
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    return precision

# Calculate precision for each model
precision_scores = []
for i, cf_matrix in enumerate(confusion_matrices):
    precision = calculate_precision(cf_matrix)
    precision_scores.append(precision)

precision_scores.append(precisionSeq)


# Convert scores to percentages
accuracy_test_percentage = [acc * 100 for acc in accuracy_test]
precision_scores_percentage = [precision * 100 for precision in precision_scores]
recall_scores_percentage = [recallVal * 100 for recallVal in recall_scores]

# Create the DataFrame with scores in percentage form
output = pd.DataFrame({
    "Model": [
        'Random Forest Classifier',
        'AdaBoost Classifier',
        'KNeighbors Classifier',
        'SGD Classifier',
        'Extra Trees Classifier',
        'Gaussian NB',
        'Sequential'
    ],
    "Accuracy (%)": accuracy_test_percentage,
    "Precision (%)": precision_scores_percentage,
    "Recall (%)": recall_scores_percentage
})

# Print the DataFrame
print()
print(output)

print()

# Sequential
# Obtain predictions for the training set
train_predictions = model.predict(X_train_np)
train_predictions_classes = (train_predictions > 0.5).astype(int)

# Evaluate metrics for the training set
train_conf_matrix = confusion_matrix(y_train_np, train_predictions_classes)
train_accuracy = accuracy_score(y_train_np, train_predictions_classes)
train_precision = calculate_precision(train_conf_matrix)
train_recall = recall_score(y_train_np, train_predictions_classes)

print("Sequential Model")
print("Training Set Metrics:")
print(f"Accuracy: {train_accuracy * 100:.2f}%")
print(f"Precision: {train_precision * 100:.2f}%")
print(f"Recall: {train_recall * 100:.2f}%")
print()

# Obtain predictions for the validation set (X_val, y_val)
val_predictions = model.predict(X_test)
val_predictions_classes = (val_predictions > 0.5).astype(int)

# Calculate metrics for the validation set
val_conf_matrix = confusion_matrix(y_test, val_predictions_classes)
val_accuracy = accuracy_score(y_test, val_predictions_classes)
val_precision = calculate_precision(val_conf_matrix)
val_recall = recall_score(y_test, val_predictions_classes)

# Print validation set metrics
print("Validation Set Metrics:")
print(f"Accuracy: {val_accuracy * 100:.2f}%")
print(f"Precision: {val_precision * 100:.2f}%")
print(f"Recall: {val_recall * 100:.2f}%")

print()


# Print Train and Validation sets' metrics
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

# List of models
models = [
    RandomForestClassifier(),
    AdaBoostClassifier(),
    KNeighborsClassifier(),
    SGDClassifier(),
    ExtraTreesClassifier(),
    GaussianNB()
]

# Assuming X_train_np, y_train_np are your training data
# Loop through each model and calculate metrics for the training set
for model in models:
    model.fit(X_train_np, y_train_np)  # Train the model

    # Obtain predictions on the training set
    train_predictions = model.predict(X_train_np)
    train_conf_matrix = confusion_matrix(y_train_np, train_predictions)
    train_accuracy = accuracy_score(y_train_np, train_predictions)
    train_precision = precision_score(y_train_np, train_predictions)
    train_recall = recall_score(y_train_np, train_predictions)

    # Print metrics for the training set
    print(f"Model: {type(model).__name__}")
    print(f"Training Set Accuracy: {train_accuracy * 100:.2f}%")
    print(f"Training Set Precision: {train_precision * 100:.2f}%")
    print(f"Training Set Recall: {train_recall * 100:.2f}%")
    print()


for model in models:
    model.fit(X_test_np, y_test_np)  # Train the model

    # Obtain predictions on the training set
    val_predictions = model.predict(X_test_np)
    val_conf_matrix = confusion_matrix(y_test_np, val_predictions)
    val_accuracy = accuracy_score(y_test_np, val_predictions)
    val_precision = precision_score(y_test_np, val_predictions)
    val_recall = recall_score(y_test_np, val_predictions)

    # Print metrics for the training set
    print(f"Model: {type(model).__name__}")
    print(f"Validation Set Accuracy: {val_accuracy * 100:.2f}%")
    print(f"Validation Set Precision: {val_precision * 100:.2f}%")
    print(f"Validation Set Recall: {val_recall * 100:.2f}%")
    print()