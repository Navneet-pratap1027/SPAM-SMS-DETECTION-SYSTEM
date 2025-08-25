import pandas as pd
import numpy as np 
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score
from sklearn.model_selection import train_test_split 

# importing csv file
df = pd.read_csv('spam.csv', encoding='ISO-8859-1', usecols=[0, 1], names=['label', 'message'])


df.drop(0,inplace=True)

## creating dependent and independent variables
X=df.message
## spam=0,ham=1
y = np.where(df['label'].str.contains('spam'), 0, 1)

## spliting the data
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.3,random_state=42)

# encoding data 
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
X_train = tfidf.fit_transform(X_train)
X_test = tfidf.transform(X_test)


##model training

models={
    "Logistic-Regression" : LogisticRegression(),
    "Naive-Bayes" : MultinomialNB(),
    "Support-Vector-Machine" : SVC(kernel="linear")
}


for i in range(len(list(models))):
    model=list(models.values())[i]
    model.fit(X_train,y_train)
    
    
    
    ## Make Prediction
    y_predict_train=model.predict(X_train)
    y_predict_test=model.predict(X_test)
    
    
    ## Train set performance
    y_predict_train_accuracy=accuracy_score(y_train,y_predict_train)
    y_predict_train_f1_score=f1_score(y_train,y_predict_train,average='weighted',zero_division=0)
    y_predict_train_precision=precision_score(y_train,y_predict_train,average='weighted',zero_division=0)
    y_predict_train_recall=recall_score(y_train,y_predict_train,average='weighted',zero_division=0)
    
    ## Test set performance
    y_predict_test_accuracy=accuracy_score(y_test,y_predict_test)
    y_predict_test_f1_score=f1_score(y_test,y_predict_test,average='weighted')
    y_predict_test_precision=precision_score(y_test,y_predict_test,average='weighted')
    y_predict_test_recall=recall_score(y_test,y_predict_test,average='weighted')
    
    print(list(models.keys())[i])
    
    print('Model performance for Training set')
    print("- Accuracy: {:.4f}".format(y_predict_train_accuracy))
    print('- F1 score: {:.4f}'.format(y_predict_train_f1_score))
    
    print('- Precision: {:.4f}'.format(y_predict_train_precision))
    print('- Recall: {:.4f}'.format(y_predict_train_recall))

    
    
    print('----------------------------------')
    
    print('Model performance for Test set')
    print('- Accuracy: {:.4f}'.format(y_predict_test_accuracy))
    print('- F1 score: {:.4f}'.format(y_predict_test_f1_score))
    print('- Precision: {:.4f}'.format(y_predict_test_precision))
    print('- Recall: {:.4f}'.format(y_predict_test_recall))

    
    print('='*35)
    print('\n')
    
    
