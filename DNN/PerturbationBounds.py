'''
Author: Bineet Ghosh
Email: bineet@cs.unc.edu
Date: November 22, 2019
'''

'''
This is based on the paper, "Maximum Resilience of Artificial Neural Networks"
Chih-Hong Cheng, Georg N¨uhrenberg, and Harald Ruess
'''

'''
This computes the maximum amount of perturbation a Deep
Neural Network can tolerate without a
mis-classification
'''

from PerturbationBoundsAPI import *

testData=[[0.73,0.36,0.48,0.5,0.53,0.91,0.92],[0.84,0.44,0.48,0.5,0.48,0.71,0.74],[0.48,0.45,0.48,0.5,0.6,0.78,0.8],[0.54,0.49,0.48,0.5,0.4,0.87,0.88],[0.48,0.41,0.48,0.5,0.51,0.9,0.88],[0.5,0.66,0.48,0.5,0.31,0.92,0.92]]
while (True):
    model1=DNNTrainer(INP_SIZE1,OUT_SIZE1,NO_LAYERS1,NEURONS1,DATASET_FILE1,DELIMETER1,EPOCH1,BATCH1)
    pb=PerturbationBounds(model1)
    pertMax=pb.maxPerturbationFinder(1,testData[0])

    if pertMax!=False:
        print("\n\n================= DNN Summary =================")
        print("DNN 1 => Accuracy: ",model1.getAccuracy()*100)

        print("\n")

        print(">>>>>> TEST DATA <<<<<<<<")
        print("Data: ",end="")
        print("-----Prediction------")
        model1.getPredictions("../Data/ecoliDataLabelTestNew.csv",",",7,8)
        print("=-=-=-=-=-=-")

        print("\n")

        print("\n\n================= DNN Perturbation =================")
        PerturbationBounds.printPerturbation(pertMax)

        print("\n")

        print("\n\n================= Perturbation Tester =================")
        model1.perturbationTesterOne(testData[0]+[1],INP_SIZE1)

        break
