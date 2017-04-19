# TP IA-Classification

Pour atteindre les objectifs du TP, deux scripts ont été développés. Ce sont les suivants:

* corpus_creation.py
* TPClassification.py

## Création des corpus

Ce premier script ```corpus_creation.py``` permet de traiter et destribuer des fichiers positifs et négatifs afin de pouvoir les utiliser pour la classification. Cela se fait en fonction de certains pourcentage et en ne gardant que les types de mots qui nous intéressent.

## Classification

La classification est basée sur le [ce tutoriel scikit-learn(http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html) expliquant comment travailler avec des données textuelles. Des explications pour chaque étape sont disponibles en commentaires directement dans le code.

La classification est effectuée dans le fichier ```TPClassification.py```

Toutes les références utilisées peuvent être trouvées à l'adresse suivante :
[Linkyou - AI - Classification Scikit Ressources](https://linkyou.srvz-webapp.he-arc.ch/collection/ai-classification-scikit-ressources-10) 


## Utilisation

### 1. Préparation des fichiers et dossier

### 2. Création des corpus

On lance simplement le script de création des corpus.

```
python .\corpus_creation.py
```

### 3. Classification

Lancez le script de classification.

```
python .\TPClassification.py
```
