from selenium import webdriver

import matplotlib.pyplot as plt
import re
import numpy as np

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
nltk.download('punkt')


# Importing 
import pandas as pd
#The One-vs-Rest strategy splits a multi-class classification into one binary classification problem per class.
from sklearn.multiclass import OneVsRestClassifier

# classifers being used
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB

#importing word cloud for visualisation 
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# for train test-test classifier
from sklearn.model_selection import train_test_split
# for accuracy
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import make_scorer, roc_curve, roc_auc_score
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics.pairwise import cosine_similarity


#CSV TO DF

news={1:"Business",2:"Entertainment",3:"Politics",4:"Science",5:"Sports"}

for i in range(1,6):
    locals()["df"+news[i]] = pd.read_csv("C:\\Users\\likhi\\OneDrive\\Desktop\\nmims studies\\DMA\\Inshorts_"+news[i]+".csv")
    if "index" in locals()["df"+news[i]]:
        locals()["df"+news[i]]=locals()["df"+news[i]].drop(["index"],axis=1)
    if "Unnamed: 0" in locals()["df"+news[i]]:
        locals()["df"+news[i]]=locals()["df"+news[i]].drop(["Unnamed: 0"],axis=1)

#Combining all the dataframes in one dataframe with equal proportion
dfInshorts = pd.concat([locals()["df"+news[1]].head(2000), locals()["df"+news[2]].head(2000), locals()["df"+news[3]].head(2000), locals()["df"+news[4]].head(2000), locals()["df"+news[5]].head(2000)], axis=0,ignore_index=True)

dfInshorts


#Visualize the words in each classification to get a better idea of the data set

from wordcloud import WordCloud

stop = set(stopwords.words('english'))

science = dfInshorts[dfInshorts['Category'] == 'science']
science = science['Content']

sports = dfInshorts[dfInshorts['Category'] == 'sports']
sports = sports['Content']

politics = dfInshorts[dfInshorts['Category'] == 'politics']
politics = politics['Content']

business = dfInshorts[dfInshorts['Category'] == 'business']
business = business['Content']

entertainment = dfInshorts[dfInshorts['Category'] == 'entertainment']
entertainment = entertainment['Content']

def wordcloud_draw(dataset, color = 'white'):
    words = ' '.join(dataset)
    cleaned_word = ' '.join([word for word in words.split() if (word != 'news' and word != 'text')])
    wordcloud = WordCloud(stopwords = stop, background_color = color, width = 3000, height = 2500).generate(cleaned_word)
    plt.figure(1, figsize = (15,7))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()

print("business words:")
wordcloud_draw(business, 'white')

print("science words:")
wordcloud_draw(science, 'white')

print("politics words:")
wordcloud_draw(politics, 'white')

print("sports words:")
wordcloud_draw(sports, 'white')

print("entertainment words:")
wordcloud_draw(entertainment, 'white')

## Cleaning and preparing the dataset for classification 

#used regex

def remove_tags(text):
    remove = re.compile(r'')
    return re.sub(remove, '', text)
dfInshorts['Content'] = dfInshorts['Content'].apply(remove_tags)


dfInshorts['Content'].head(10)

def convert_lower_remove_special_char(text):
    reviews = ''
    if type(text) is not int:
        for x in text:
            if x.isalnum():
                
                  reviews = reviews + x
            else:
                  reviews = reviews + ' '
                    
        reviews=reviews.lower()
        
                    
    return reviews
dfInshorts['Content'] = dfInshorts['Content'].apply(convert_lower_remove_special_char)

#Joining again
for i in range(0,len(dfInshorts['Content'])):
    dfInshorts['Content'][i]=" ".join(dfInshorts['Content'][i].split())


dfInshorts['Content'].head(20)

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    return [x for x in words if x not in stop_words]
dfInshorts['Content'] = dfInshorts['Content'].apply(remove_stopwords)
dfInshorts['Content'][1]

dfInshorts['Content']

def lemmatize_word(text):
    wordnet = WordNetLemmatizer()
    return " ".join([wordnet.lemmatize(word) for word in text])
dfInshorts['Content'] = dfInshorts['Content'].apply(lemmatize_word)
dfInshorts['Content'][1]

dfInshorts['Content']

#Creating a fit bag of words
from sklearn.feature_extraction.text import CountVectorizer
x = np.array(dfInshorts.Content.values)
y = np.array(dfInshorts.Category.values)
cv = CountVectorizer(max_features = 5000) #TOP 5000 words marked with 1 if occured in a (row) sentence
x = cv.fit_transform(dfInshorts.Content).toarray() 
print("X.shape = ",x.shape)
print("y.shape = ",y.shape)

## Splitting data, Train and Test

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0, shuffle = True)
print(len(x_train))
print(len(x_test))

## Now we are ready to train the model 
'''
Used a one vs all approach for all the models, as the computation time and the complexity is quiet lower compared to one vs one.

Random Forest,
Multinomial Naive Bayes,
Support Vector Classifer,
Decision Tree Classifier,
K Nearest Neighbour,
Gaussian Naive Bayes
'''
perform_list = [ ]


def model(MdName):
    model=''
    if MdName == 'Logistic Regression':
        model = LogisticRegression()
    elif MdName == 'Support Vector Classifer':
        model = SVC()
    elif MdName == 'Decision Tree Classifier':
        model = DecisionTreeClassifier()
    elif MdName == 'Random Forest':
        model = RandomForestClassifier(n_estimators=100 ,criterion='entropy' , random_state=0)
    elif MdName == 'Multinomial Naive Bayes':
        model = MultinomialNB(alpha=1.0,fit_prior=True)
    elif MdName == 'K Nearest Neighbour':
        model = KNeighborsClassifier(n_neighbors=10 , metric= 'minkowski' , p = 4)
    elif MdName == 'Gaussian Naive Bayes':
        model = GaussianNB()
        
    oneVsRest = OneVsRestClassifier(model)
    oneVsRest.fit(x_train, y_train)
    y_pred = oneVsRest.predict(x_test)

    acc = round(accuracy_score(y_test, y_pred) * 100, 2)

    prec, reca, f1, supp = score(y_test, y_pred, average='micro')

    print(f'Test Accuracy Score of Basic {MdName}: % {acc}')

    print(f'Precision : {prec}')

    print(f'Recall : {reca}')

    print(f'F1-score : {f1}')


    perform_list.append(dict([

    ('Model', MdName),

    ('Test Accuracy', round(acc, 2)),

    ('Precision', round(prec, 2)),

    ('Recall', round(reca, 2)),

    ('F1', round(f1, 2))

    ]))

model('Logistic Regression')
model('Random Forest')
model('Multinomial Naive Bayes')
model('Support Vector Classifer')
model('Decision Tree Classifier')
model('K Nearest Neighbour')
model('Gaussian Naive Bayes')


#Check the performance of all the models
performance = pd.DataFrame(data=perform_list)
performance = performance[['Model', 'Test Accuracy', 'Precision', 'Recall', 'F1']]

model = performance["Model"]

# most accurate model for the prediction 
maxAccuracy = performance["Test Accuracy"].max()
print("Lets see the highest test accuracy which is at", maxAccuracy," -----Multinomial Naive Bayes")

#Choosing MNB as our best classifier. 
MNB_Model = MultinomialNB(alpha=1.0,fit_prior=True)
MNB_Classifier=MNB_Model.fit(x_train, y_train)
y_pred = MNB_Classifier.predict(x_test)

# Answer expected science news 
yp = cv.transform(["A tree that is up to 5,484 years old has been found in a forest in southern Chile and it is believed to be the world's oldest tree, as per a new study. The age of the ancient alerce tree known as 'great grandfather' beats the current record-holder, which is a 4,853-year-old bristlecone pine tree in California"])
result = MNB_Classifier.predict(yp)
print(result)



#Use this piece of code to dump the model and the vectorizer
'''
import pickle
pickle.dump(MNB_Classifier, open('MNB_Classifier.pkl','wb'))
pickle.dump(cv, open("Vectorizer.pkl", "wb"))

v = pickle.load(open('Vectorizer.pkl','rb'))
MNB_Classifier = pickle.load(open('MNB_Classifier.pkl','rb'))
z=MNB_Classifier.predict(v.transform(["A tree that is up to 5,484 years old has been found in a forest in southern Chile and it is believed to be the world's oldest tree, as per a new study. The age of the ancient alerce tree known as 'great grandfather' beats the current record-holder, which is a 4,853-year-old bristlecone pine tree in California"]))

'''


'''
# Conclusion 
The scraped data had a huge amount of words. Hence cleaning and preparing the data was cruicial for training the model.

    Special characters were removed
    
    Stop words were removed
    
    Lemmatized the content
    
    Created a df of top 5000 frequent words in the dataset 
    
    Splitted the data into train and test
    
    Multi-nomiall Naives Bayes performs the best compared to other classification models
    
    

Multi-nomail Naives Bayes classification performs the best. The reason is that we have 5 classes. So it works better for multi class. 
'''