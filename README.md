# TP IA-Classification

Pour atteindre les objectifs du TP, deux scripts ont été développés. Ce sont les suivants:

* ```corpus_creation.py```
* ```TPClassification.py```

## Création des corpus

Ce premier script ```corpus_creation.py``` permet de traiter et destribuer des fichiers positifs et négatifs afin de pouvoir les utiliser pour la classification. Cela se fait en fonction de certains pourcentages (selon la donnée) et en ne gardant que les types de mots qui nous intéressent.

### Rappel de la distribution

Il y a 1000 fichiers de reviews positives et 1000 fichiers de reviews négatives soit un total de 2000 fichiers.

* 80% de ces fichiers sont utilisés pour l'entraînement (soit 400 positifs et 400 négatifs)
* 20% de ces fichiers sont utilisés pour les tests (soit 200 positits et 200 négatifs)

## Classification

La classification est basée sur le [ce tutoriel scikit-learn(http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html) expliquant comment travailler avec des données textuelles. Des explications pour chaque étape sont disponibles en commentaires directement dans le code.

La classification est effectuée dans le fichier ```TPClassification.py```

Toutes les références utilisées peuvent être trouvées à l'adresse suivante :
[Linkyou - AI - Classification Scikit Ressources](https://linkyou.srvz-webapp.he-arc.ch/collection/ai-classification-scikit-ressources-10) 


## Utilisation

Pour la démonstration, les processed files ne sont pas généré par défaut, il faut donc d'abord exécuter le script de création des corpus avant de pouvoir exécuter le script de classification.

### 1. Préparation des fichiers et dossier

Tous les fichiers nécessaires sont disponibles. Les dossiers se trouvant dans classification/processed sont vides par défaut et sont remplis par le script de création des corpus.

Les fichiers dans corpus_example contiennt un exemple de corpus créé précédemment (au cas où).

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

## Commentaires additionnels

La partie optimisation ne fonctionne pas sur Windows (probablement à cause de l'utilisation de fork()).
