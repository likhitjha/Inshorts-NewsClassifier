# Inshorts-NewsClassifier
#### End Product: https://leekith-newsclassifier.herokuapp.com/
## This project has 4 main parts :
 
### 1) Data Collection
- Out of the several categories in Inhorts, we select these 5 categories,
- We will be scraping the news card content from the Inshorts website: https://www.inshorts.com 
- Categories:

  - Science

  - Sports

  - Politics

  - Entertainment

  - Business

- Scrapped data from Inshort's website is used in the next process

### 2) Data Cleaning 
- Cleaning and reorganizing the data and finally transforming it into number type, so that we can feed it into the model.
- Making a vectorizer of the vocab.

### 3) Classifying
- Used a one vs all approach for all the models, as the computation time and complexity are quite lower than one vs one.
- Models:

  - Random Forest

  - Multinomial Naive Bayes

  - Support Vector Classifier

  - Decision Tree Classifier

  - K Nearest Neighbour

  - Gaussian Naive Bayes

- Best model: MNB-Multinomial Naive Bayes

### 4) Deploying
- Saving the MNB model and the Vectorizer using PICKLE.
- Created a local website to deploy the model
- Used Heroku for showcasing the model: 
https://leekith-newsclassifier.herokuapp.com/
