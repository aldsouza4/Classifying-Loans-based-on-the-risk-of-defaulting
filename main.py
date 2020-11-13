import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
sns.set_style('white')

# Start

dataFrame = pd.read_csv('lending_club_loan_two.csv')

print(dataFrame.to_string())

dataFrame['term'] = dataFrame['term'].apply(lambda term: int(term[:3]))

dataFrame["loan_status"]=dataFrame['loan_status'].map({"Fully Paid":1, "Charged Off":0})

dataFrame=dataFrame.drop('title', axis=1)

total_acc_avg = dataFrame.groupby('total_acc').mean()['mort_acc']

def fill_mort_acc(total_acc, mort_acc):
    if np.isnan(mort_acc):
        return total_acc_avg[total_acc]
    else:
        return mort_acc

dataFrame['mort_acc']=dataFrame.apply(lambda x : fill_mort_acc(x['total_acc'], x['mort_acc']), axis=1)

dataFrame=dataFrame.dropna()

dataFrame= dataFrame.drop('grade', axis=1)

dummies= pd.get_dummies(dataFrame['sub_grade'], drop_first=True)

dataFrame = pd.concat([dataFrame.drop('sub_grade', axis=1), dummies], axis=1)

dummies= pd.get_dummies(dataFrame[['verification_status', 'application_type',
                            'purpose','initial_list_status']], drop_first=True)

dataFrame = pd.concat([dataFrame.drop(['verification_status', 'application_type', 'purpose',
                         'initial_list_status'], axis=1), dummies], axis=1)

dataFrame['home_ownership'] = dataFrame['home_ownership'].replace(['NONE', 'ANY'], 'OTHER')

dummies= pd.get_dummies(dataFrame['home_ownership'], drop_first=True)

dataFrame = pd.concat([dataFrame.drop('home_ownership', axis=1), dummies], axis=1)

dataFrame['zip_code']=dataFrame['address'].amapply(lambda x : x[-5:])

dummies = pd.get_dummies(dataFrame['zip_code'], drop_first=True)

dataFrame = pd.concat([dataFrame.drop('address', axis=1), dummies], axis=1)

dataFrame = dataFrame.drop('issue_d', axis=1)

dataFrame['earliest_cr_line'] = dataFrame['earliest_cr_line'].apply(lambda x: int(x[-4:]))

dataFrame = dataFrame.drop('emp_title', axis=1)

dataFrame = dataFrame.drop('emp_length', axis=1)

X = dataFrame.drop('loan_status', axis=1).values
y = dataFrame['loan_status'].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model= Sequential()

model.add(Dense(78, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(35, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(25, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam')

model.fit(x=X_train, y=y_train, epochs=25, batch_size=256, validation_data=(X_test,y_test))

model.save('LoanPredict.h5')


