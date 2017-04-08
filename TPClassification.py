import sklearn
import sklearn.datasets
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV

# Based on http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html
# All refereces used are here (web project) : https://linkyou.srvz-webapp.he-arc.ch/collection/ai-classification-scikit-ressources-10


# Set to True for a detailed step by step output
DETAILED_OUTPUT = False

############################################################################
#################### STEP 1 | Loading datasets #############################
############################################################################
'''
We load the text files with folder names used as supervised signal label names.
In this particular case, there are simply two categories :
   "neg" for negative reviews
   "pos" for positive reviews
http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_files.html
'''

categories = ['pos', 'neg'] # Possible classification categories

reviews_train = sklearn.datasets.load_files(
    "./classification/processed/reviews-train",
    description = "Training data composed of processed positive and negative movie review files.",
    categories=categories,  # Use the defined categories
    load_content=True,      # Load files in memory
    shuffle=True,           # Shuffle data (might be important for models that make the assumption that the samples are independent and identically distributed)
    encoding='latin-1',     # Encoding
    decode_error='strict',  # Decode error mode
    random_state=42,        # The seed used by the random number generator
    )

reviews_test = sklearn.datasets.load_files(
    "./classification/processed/reviews-test",
    description = "Testing data composed of processed positive and negative review files.",
    categories=categories,
    load_content=True,
    shuffle=True,
    encoding='latin-1',
    decode_error='strict',
    random_state=42,
    )


############################################################################
###################### STEP 2 | Vectorisation ##############################
############################################################################
'''
In order too perform machine learning on text documents,
we first need to turn the text content into numerical feature vectors.
http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html#sklearn.feature_extraction.text.CountVectorizer.fit_transform
'''

# Convert the collection of text documents to a matrix of token counts
count_vect = CountVectorizer()

# Learn the vocabulary dictionary and return term-document matrix.
X_train_counts = count_vect.fit_transform(reviews_train.data)


############################################################################
###################### STEP 3 | Indexation Ef/idf ##########################
############################################################################
'''
Counting occurences is a good thing but longer documents will have higher
average count values than shorter documents (even with same topics).

To avoid this we want to calculate a "Term Frequency" value
    -> tf for "Term Frequency"

And because the most descriptive words are often the less used,
we calculate another value on top of the tf, the tf-idf
    -> tf-idf for "Term Frequency times Inverse Document Frequency" -> tf-idf

http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html

'''

tfidf_transformer = TfidfTransformer() # Transform a count matrix to a normalized tf or tf-idf representation
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)


############################################################################
######################## STEP 4 | Classification ###########################
############################################################################
'''
Now that we have the features we can train the classifier to predict the character of a review.

We firstly use the "naïve Bayes" classifier (suitable for classification with discrete features).
http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html

'''

# Naive Bayes classifier for multinomial models
clf = MultinomialNB().fit(X_train_tfidf, reviews_train.target)

'''
# Working example of prediction
docs_new = ['Nul pas bien', 'Beau incroyable superbe']
X_new_counts = count_vect.transform(docs_new)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)
predicted = clf.predict(X_new_tfidf)
for doc, category in zip(docs_new, predicted):
    print('%r => %s' % (doc, reviews_train.target_names[category]))
'''

############################################################################
######################## STEP 5 | Evaluation ###########################
############################################################################
'''
We can finally evaluate the predictive accuracy of the moel.

http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html
'''

docs_test = reviews_test.data # Test documents

# A pipeline with MultinomialNB (naïve Bayes)
text_clf_NB = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB()),])

# A pipeline with support vector machine (SVM)
text_clf_SGDC = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42)),])

text_clf_NB = text_clf_NB.fit(reviews_train.data, reviews_train.target)
_ = text_clf_SGDC.fit(reviews_train.data, reviews_train.target)

# Predictions with both models
predicted_NB = text_clf_NB.predict(docs_test)
predicted_SGDC = text_clf_SGDC.predict(docs_test)

print("\nPrediction accuracies :")
print("\tNaïve Bayes prediction \t : {0}".format(np.mean(predicted_NB == reviews_test.target)))
print("\tSVM prediction \t\t : {0}".format(np.mean(predicted_SGDC == reviews_test.target)))


############################################################################
################### STEP + | Tuning / Optimization #########################
############################################################################

'''
Classifiers tend to have many parameters. It is possible to run exhaustive search
of the best parameters on a grid of possible alues.
'''

# Possible parameters
parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
              'tfidf__use_idf': (True, False),
              'clf__alpha': (1e-2, 1e-3),
}

gs_clf = GridSearchCV(text_clf_SGDC, parameters, n_jobs=-1) # Use all cores

# Try fit on a subset of data
gs_clf = gs_clf.fit(reviews_train.data[:400], reviews_train.target[:400])

demoString = 'Superbe génial aimer beau'

print("\nThe demo prediction for \"{0}\" is : {1}".format(demoString, reviews_train.target_names[gs_clf.predict([demoString])[0]]))

print("\nThe best score with SVM is {0}".format(gs_clf.best_score_))
print("\nFound with the following parameters :\n")
for param_name in sorted(parameters.keys()):
    print("\t%s: %r" % (param_name, gs_clf.best_params_[param_name]))


############################################################################
####################### EVENTUAL OUTPUTS #############################
############################################################################

if DETAILED_OUTPUT:

    print("\n\nSTEP 1 | Datasets loading outputs :")
    print(reviews_train.target_names)
    print(len(reviews_train.data))
    print(len(reviews_train.filenames))
    print("\n".join(reviews_train.data[0].split("\n")[:3]))

    print("\n\nSTEP 2 | Vectorisation outputs :")
    print(X_train_counts.shape)
    print(count_vect.vocabulary_.get(u'algorithm'))


    print("\n\nSTEP 3 | Indexation outputs :")
    print(X_train_tfidf.shape)

    print("\n\n+ | Metrics outputs :")
    print("\n\nClassification report for Naïve Bayes")
    print(metrics.classification_report(reviews_test.target, predicted_NB,target_names=reviews_test.target_names))
    print("Confusion matrix for Naïve Bayes")
    print(metrics.confusion_matrix(reviews_test.target, predicted_NB))

    print("\n\nClassification report for SVM")
    print(metrics.classification_report(reviews_test.target, predicted_SGDC,target_names=reviews_test.target_names))
    print("Confusion matrix for SVM")
    print(metrics.confusion_matrix(reviews_test.target, predicted_SGDC))
