import sklearn
import sklearn.datasets
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier

categories = ['pos', 'neg']

# 1. Etape de chargement
reviews_train = sklearn.datasets.load_files(
    "./classification/processed/reviews-train",
    description = None,
    categories=categories,
    load_content=True,
    shuffle=True,
    encoding='latin-1',
    decode_error='strict',
    random_state=42)

reviews_test = sklearn.datasets.load_files(
    "./classification/processed/reviews-test",
    description = None,
    categories=categories,
    load_content=True,
    shuffle=True,
    encoding='latin-1',
    decode_error='strict',
    random_state=42)

# Debug prints
print(reviews_train)
print(reviews_train.target_names)
print(len(reviews_train.data))
print(len(reviews_train.filenames))
print("\n".join(reviews_train.data[0].split("\n")[:3]))


# Etape de vectorisation
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(reviews_train.data)

print(X_train_counts.shape)
print(count_vect.vocabulary_.get(u'algorithm'))

# La matrice est remplie avec soit un 0 soit un nombre d'occurence sur la ligne

# 3. Etape d'indexation Ef/idf

#Méthode lente
#tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
#X_train_tf = tf_transformer.transform(X_train_counts)
#print(X_train_tf.shape)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
print(X_train_tfidf.shape)


# 4. Etape de classification

# On applique le modèle, par exemple naive bayes (probabiliste) ou SVM (géométrique)
# Support Vector Machine (modèle géométrique)
# Features ou caractéristiques (colonnes) expriement l'espace multidimensionnel (défini si facile ou non de séparer)
clf = MultinomialNB().fit(X_train_tfidf, reviews_train.target)

docs_new = ['God is love', 'OpenGL on the GPU is fast']
X_new_counts = count_vect.transform(docs_new)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)

predicted = clf.predict(X_new_tfidf)

for doc, category in zip(docs_new, predicted):
    print('%r => %s' % (doc, reviews_train.target_names[category]))

# 5. Etape d'évaluation

text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB()),
])

text_clf = text_clf.fit(reviews_train.data, reviews_train.target)

docs_test = reviews_test.data
predicted = text_clf.predict(docs_test)
print(np.mean(predicted == reviews_test.target))



text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                           alpha=1e-3, n_iter=5, random_state=42)),
])
_ = text_clf.fit(reviews_train.data, reviews_train.target)
predicted = text_clf.predict(docs_test)
print(np.mean(predicted == reviews_test.target))



print(metrics.classification_report(reviews_test.target, predicted,
    target_names=reviews_test.target_names))

print(metrics.confusion_matrix(reviews_test.target, predicted))

# Examen : comprendre naive bayes (pas détails calcul), créer le tableau et exemple classification
# Expliquer les cinq étapes
