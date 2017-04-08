import os
import random

# Possible categories of classification
categories = ["pos", "neg"]

# Folders containing the (already) tagged negative and positive reviews
source_folders = ["classification/tagged/neg", "classification/tagged/pos"]

# Folders in which we will put the processed files to train and test the classification
target_folders =  ["classification/processed/reviews-train/neg",
                    "classification/processed/reviews-train/pos",
                    "classification/processed/reviews-test/neg",
                    "classification/processed/reviews-test/pos",
                    ]

# The type of text informations judged interesting (see any tagged file for example)
interest_words = ["NOM", "ADJ", "VER"]

# Ratios of positives / negatives for training
train_main_percentage = 0.8
train_pos_percentage = 0.5
train_neg_percentage = 1 - train_pos_percentage

# Ratios of positives / negatives for testing
test_main_percentage = 1 - train_main_percentage
test_pos_percentage = 0.5
test_neg_percentage = 1 - test_pos_percentage

# TODO: Tout ceci est une horreur
def process_review_files(source_folder, target_folder, files):
    '''Process each files in the given folders'''
    for processed_file in files:
        with open("{0}/{1}".format(source_folder, processed_file), 'r') as input_f:
            strWords = ""
            for line in input_f:
                for word in interest_words:
                    if word in line:
                        wordOfInterest = line.split("\t")[2]
                        if "|" in wordOfInterest:
                            wordOfInterest = wordOfInterest.split("|")[0]
                        strWords += wordOfInterest
        with open("{0}/processed_{1}".format(target_folder, processed_file), 'w') as output_f:
            output_f.write(strWords)

if __name__ == '__main__':

    print("Starting program")

    # All negative and positive files in source folders
    negFiles = [filename for filename in os.listdir(source_folders[0])]
    posFiles = [filename for filename in os.listdir(source_folders[1])]

    # Counting files for fun
    nbTotNegFiles = len([filename for filename in os.listdir(source_folders[0])])
    nbTotPosFiles = len([filename for filename in os.listdir(source_folders[1])])

    # Calculate the number of indices needed for each case in function of global variables
    posTrainIndices = int(round(((nbTotNegFiles + nbTotPosFiles) * train_main_percentage) * train_pos_percentage))
    negTrainIndices = int(round(((nbTotNegFiles + nbTotPosFiles) * train_main_percentage) * train_neg_percentage))
    posTestIndices = int(round(((nbTotNegFiles + nbTotPosFiles) * test_main_percentage) * test_pos_percentage))
    negTestIndices = int(round(((nbTotNegFiles + nbTotPosFiles) * test_main_percentage) * test_neg_percentage))

    ############################################################################
    ############## RANDOM INDICES GENERATION FOR EACH CASE #####################
    ############################################################################

    # Generate random list of indices for each training and testing cases
    randomPosTrainIndices = random.sample(range(nbTotPosFiles), posTrainIndices) # Positive training indices
    randomNegTrainIndices = random.sample(range(nbTotNegFiles), negTrainIndices) # Negative training indices

    randomPosTestIndices = random.sample(range(nbTotPosFiles), posTestIndices) # Positive testing indices
    randomNegTestIndices = random.sample(range(nbTotNegFiles), negTestIndices) # Negative testing indices

    ############################################################################
    ############## LIST OF FILES IN FUNCTION OF RANDOM INDICES #################
    ############################################################################

    workingFilesPosTrain = [posFiles[i] for i in randomPosTrainIndices] # Working files
    workingFilesNegTrain = [negFiles[i] for i in randomNegTrainIndices]

    workingFilesPosTest = [posFiles[i] for i in randomPosTestIndices]
    workingFilesNegTest = [negFiles[i] for i in randomNegTestIndices]

    process_review_files(source_folders[0], target_folders[0], workingFilesNegTrain)
    process_review_files(source_folders[0], target_folders[2], workingFilesNegTest)

    process_review_files(source_folders[1], target_folders[1], workingFilesPosTrain)
    process_review_files(source_folders[1], target_folders[3], workingFilesPosTest)

    ############################################################################
    #################### INFORMATIVE OUTPUTS FOR TESTING #######################
    ############################################################################
    print("{0} files in {1}".format(nbTotNegFiles, source_folders[1]))
    print("{0} files in {1}".format(nbTotPosFiles, source_folders[0]))

    print("There will be {0} training positive files from folder {1}.".format(posTrainIndices, source_folders[1]))
    print("There will be {0} training negative files from folder {1}.".format(negTrainIndices, source_folders[0]))
    print("There will be {0} testing positive files from folder {1}.".format(posTestIndices, source_folders[1]))
    print("There will be {0} testing negative files from folder {1}.".format(negTestIndices, source_folders[0]))

    print("{0} random indices generated for training positive.".format(len(randomPosTrainIndices)))
    print("{0} random indices generated for training negative.".format(len(randomNegTrainIndices)))
    print("{0} random indices generated for testing positive.".format(len(randomPosTestIndices)))
    print("{0} random indices generated for testing negative.".format(len(randomNegTestIndices)))
