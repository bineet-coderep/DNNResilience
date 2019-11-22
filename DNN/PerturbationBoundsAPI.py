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
This provides a list of APIs to compute
the maximum amount of perturbation a Deep
Neural Network can tolerate without a
mis-classification
'''

from keras.models import Sequential
from keras.layers import Dense
import numpy
from gurobipy import *

'''
Parameters for fixing the architecture
of the three DNN
'''

#DNN 1
INP_SIZE1=7
OUT_SIZE1=8
NO_LAYERS1=4
NEURONS1=[8,8,8,8]
DATASET_FILE1="../Data/ecoliData.csv"
DELIMETER1=","
EPOCH1=200
BATCH1=20

#DNN 2
INP_SIZE2=7
OUT_SIZE2=8
NO_LAYERS2=5
NEURONS2=[16,8,16,8,16]
DATASET_FILE2="../Data/ecoliData.csv"
DELIMETER2=","
EPOCH2=150
BATCH2=10

#DNN 3
INP_SIZE3=7
OUT_SIZE3=8
NO_LAYERS3=7
NEURONS3=[8,12,8,17,8,7,11]
DATASET_FILE3="../Data/ecoliData.csv"
DELIMETER3=","
EPOCH3=150
BATCH3=10


EPSILON=1e-5


'''Following Parameters are for
computing the Maximum  Perturbation Tolerance
'''

MAX_THRESHOLD=0.5
M=20 #Value of M, to be used for applying big-M method

class DNNTrainer:
    '''This class trains a Deep Neural Network
    according to the given architecture
    '''

    def __init__(self,inp_s,out_s,n_layers,neurons,dataFile,d,epoch,batch):
        self.inp_size=inp_s
        self.out_size=out_s
        self.n_layers=n_layers
        self.neurons=neurons
        self.epoch=epoch
        self.batch_size=batch
        self.dataset = numpy.loadtxt(dataFile, delimiter=d)
        self.X = self.dataset[:,:inp_s]
        self.Y = self.dataset[:,inp_s:(inp_s+out_s)]


        '''Trains a Neural Network according to the
        architecture dictated by the above Parameters
        '''

        print("========= Training Started =========")
        self.model = Sequential()
        self.model.add(Dense(neurons[0], input_dim=inp_s, init='uniform', activation='relu'))
        for n in neurons[1:]:
            self.model.add(Dense(n, init='uniform', activation='relu'))
        self.model.add(Dense(out_s, init='uniform', activation='relu'))
        self.model.add(Dense(out_s, init='uniform', activation='softmax'))
        self.model.compile(loss='binary_crossentropy' , optimizer='adam', metrics=['accuracy'])
        self.model.fit(self.X, self.Y, nb_epoch=epoch, batch_size=batch)
        print("Accuracy: ",self.getAccuracy())
        print("========= Training Completed =========")

    def getAccuracy(self):
        '''Returns the accuracy of the trained model
        '''
        scores = self.model.evaluate(self.X, self.Y)
        return (scores[1])

    def getWeights(self,layer):
        '''Returns the weight matrix of the trained
        Neural Network
        '''
        return self.model.layers[layer].get_weights()

    def getPredictions(self,fname,d,inp_s,out_s):
        '''Predict the class labels of the dataset
        as given by the file fname
        '''
        datasetInp = numpy.loadtxt(fname, delimiter=d)
        I = datasetInp[:,:inp_s]
        O = datasetInp[:,inp_s:(inp_s+out_s)]
        predictions = self.model.predict(I)

        print("\n\n============ Prediction Results ============\n")

        for i in range(len(O)):
            indx=-7
            max=-9999

            # Find the label with max confidence
            for j in range(len(predictions[i])):
                if predictions[i][j]>max:
                    max=predictions[i][j]
                    indx=j

            print("Data Set - Predicted Label ",i,": ",(indx+1)) #Prints the predicted label
            for j in range(len(O[i])):
                if (O[i][j]==1):
                    print("Data Set - Actual Label ",i,": ",(j+1)) # Prints the actual label
                    print("")

    def perturbationTester(self,fname,d,inp_s,out_s):

        '''This function is to test empirically how perturbation to the
        input affects the predicted class label.

        The module lets you add perturbation to the input and see the
        new prediction outcome
        '''

        '''
        Input:
        (1) Row Number from the dataset in fname which you want
        to perturb.
        (2) Then add the amount of perturbation.

        Ouput:
        (1) New Predicted Label of the perturbed data
        '''

        datasetInp = numpy.loadtxt(fname, delimiter=d)
        I = datasetInp[:,:inp_s]
        O = datasetInp[:,inp_s:(inp_s+out_s)]


        while(True):
            dt=int(input("Enter the row number: "))
            for j in range(len(O[dt])):
                if (O[dt][j]==1):
                    print("True Label : ",(j+1))
            predictions = self.model.predict(I[dt:dt+1])
            indx=-7
            max=-9999
            for j in range(len(predictions[0])):
                if predictions[0][j]>max:
                    max=predictions[0][j]
                    indx=j
            print("Predicted Label : ",(indx+1))
            print("True Data: ", I[dt])
            print("Enter Perturbations: ")
            for i in range(inp_s):
                q=float(input())
                I[dt][i]=I[dt][i]+q
            print("Perturbed Data: ", I[dt])
            predictions = self.model.predict(I[dt:dt+1])
            print("Predicted Label After Perturbations: ",(indx+1))
            print("--------------------------------")
            input("Press Any Key!")

    def perturbationTesterOne(self,data,inp_s):

        '''This function is same as the previous one
        except it only works for one single input vector.

        Also, the input to this function is an array
        and not a vector
        '''

        datasetInp = data
        I = numpy.asarray([datasetInp[:inp_s]])
        O = datasetInp[inp_s:(inp_s+1)][0]

        while(True):

            print("True Label : ",O+1)
            predictions = self.model.predict(I)
            indx=-7
            max=-9999
            for j in range(len(predictions[0])):
                if predictions[0][j]>max:
                    max=predictions[0][j]
                    indx=j
            print("Predicted Label : ",(indx+1))
            print("True Data: ", I[0])
            print("Enter Perturbations: ")
            for i in range(inp_s):
                q=float(input())
                I[0][i]=I[0][i]+q
            print("Perturbed Data: ", I[0])
            predictions = self.model.predict(I)
            indx=-7
            max=-9999
            for j in range(len(predictions[0])):
                if predictions[0][j]>max:
                    max=predictions[0][j]
                    indx=j
            print("Predicted Label After Perturbations: ",(indx+1))
            print("--------------------------------")
            input("Press Any Key!")

class PerturbationBounds:
    '''This class is to compute the amount of perturbation an
    input can tolerate without getting misclassified
    '''

    def __init__(self,dnn):
        self.dnnTrained=dnn #Type: DNNTrainer

    @staticmethod
    def printPerturbation(pert):
        '''Prints the amount of perturbation the
        input can tolerate without getting misclassified
        '''

        if (pert!=False):
            for i in range(len(pert)):
                print("delta.",i,'%.4f'%pert[i])

    def layerMatrices(self):
        '''Returns the weight and bias matrices of all
        the layers
        '''

        matList=[]
        for i in range(NO_LAYERS1+1):
            l=[]
            w=self.dnnTrained.getWeights(i)[0] #Keras Matrix
            l=[numpy.transpose(w),numpy.reshape(self.dnnTrained.getWeights(i)[1],(self.dnnTrained.getWeights(i)[1].size,1))]
            matList.append(l)

        return matList

    def maxPerturbationFinder(self,lb,ip):
        ''' Returns the amout of pertutbation input
        ip can tolerate without getting misclassified from
        class label lb'''

        model = Model("qp");

        matList=self.layerMatrices()

        neurons=NEURONS1+[OUT_SIZE1]

        neuron_op=[]
        pert=[]

        #Encode the value of x (I/P)
        inputVars=[]
        for i in range(INP_SIZE1):
            name="x."+str(i)
            inputVars.append(model.addVar(-GRB.INFINITY,GRB.INFINITY,name=name,vtype='C'))

        for i in range(INP_SIZE1):
            model.addConstr(inputVars[i]==ip[i])

        #-------------------------------

        #Encode Perturbation Variables
        perturbVars=[]
        for i in range(INP_SIZE1):
            name="delta."+str(i)
            perturbVars.append(model.addVar(-GRB.INFINITY,GRB.INFINITY,name=name,vtype='C'))

        for i in range(INP_SIZE1):
            name="Fault-C"+str(i)
            model.addConstr(perturbVars[i]>=-1,name+".1")
            model.addConstr(perturbVars[i]<=1,name+".2")
            #model.addConstr(perturbVars[i]==0)
        #-------------------------------

        #Encode Perturbation Absoulte Variables
        perturbVarsAbs=[]
        for i in range(INP_SIZE1):
            name="deltaAbs."+str(i)
            perturbVarsAbs.append(model.addVar(-GRB.INFINITY,GRB.INFINITY,name=name,vtype='C'))

        for i in range(INP_SIZE1):
            model.addConstr(perturbVarsAbs[i]>=perturbVars[i])
            model.addConstr(perturbVarsAbs[i]>=-perturbVars[i])

        #-------------------------------

        #Encode O/P of each layers
        op_vars=[]
        for i in range(NO_LAYERS1+1):
            tmp=[]
            for j in range(neurons[i]):
                name="op_var("+str(i)+","+str(j)+")"
                tmp.append(model.addVar(-GRB.INFINITY,GRB.INFINITY,name=name,vtype='C'))
            op_vars.append(tmp)
        #-------------------------

        #Additional b_i Variables required for encoding
        b=[]
        for i in range(NO_LAYERS1+1):
            t=[]
            for j in range(neurons[i]):
                name="b("+str(i)+","+str(j)+")"
                t.append(model.addVar(-GRB.INFINITY,GRB.INFINITY,name=name,vtype='I'))
            b.append(t)

        for i in range(NO_LAYERS1+1):
            for j in range(neurons[i]):
                model.addConstr(b[i][j]>=0)
                model.addConstr(b[i][j]<=1)
        #------------------------------------------

        #Additional M_i Variables required for encoding
        M_Vars=[]
        for i in range(NO_LAYERS1+1):
            t=[]
            for j in range(neurons[i]):
                name="M("+str(i)+","+str(j)+")"
                t.append(model.addVar(-GRB.INFINITY,GRB.INFINITY,name=name,vtype='C'))
            M_Vars.append(t)

        for i in range(NO_LAYERS1+1):
            for j in range(neurons[i]):
                model.addConstr(M_Vars[i][j]==M)
        #------------------------------------------

        #Neuron wise ReLU encoding (big-M method)
        for i in range(NO_LAYERS1+1):
            for j in range(neurons[i]):
                e=0

                if i==0:
                    e=0
                    for k in range(INP_SIZE1):
                        e=e+(matList[i][0][j][k]*(inputVars[k]+perturbVars[k]))
                    e=(e)+matList[i][1][j][0]
                else:
                    e=0
                    for k in range(neurons[i-1]):
                        e=e+(matList[i][0][j][k]*op_vars[i-1][k])
                    e=(e)+matList[i][1][j][0]



                model.addConstr(op_vars[i][j]>=0) #(2a)
                model.addConstr(op_vars[i][j]>=e) #(2b)
                model.addConstr(e-(b[i][j]*M_Vars[i][j])<=0) #(3a)
                model.addConstr(e+((1-b[i][j])*M_Vars[i][j])>=0) #(3b)
                model.addConstr(op_vars[i][j]<=e+((1-b[i][j])*M_Vars[i][j])) #(4a)
                model.addConstr(op_vars[i][j]<=b[i][j]*M_Vars[i][j]) #(4b)
        #----------------------------------------

        #Perturbation Tolerance of class_label
        class_label=lb;
        L=NO_LAYERS1+1

        cl=class_label
        for i in range(OUT_SIZE1):
            if i!=cl:
                model.addConstr(op_vars[L-1][cl]>=op_vars[L-1][i])
        #----------------------------------------

        #Set the Objective function
        obj=0
        for i in range(INP_SIZE1):
            obj=obj+(perturbVarsAbs[i])

        model.addConstr(obj>=MAX_THRESHOLD)
        #---------------------------------

        #Start the Gurobi Engine
        model.setObjective(obj)
        model.optimize()

        status = model.Status
        if status==GRB.Status.UNBOUNDED:
            print("UNBOUNDED ")
        else:
            if status == GRB.Status.INF_OR_UNBD or \
               status == GRB.Status.INFEASIBLE  or \
               status == GRB.Status.UNBOUNDED:
                print('**The model cannot be solved because it is infeasible or unbounded**')
            else:
                print("\n\nPerturbation\n\n")
                print()
                for v in model.getVars():
                    print('%s %g' % (v.varName, v.x))

                for opR in op_vars:
                    t=[]
                    for op in opR:
                        t.append(op.x)
                    neuron_op.append(t)

                for i in perturbVars:
                    pert.append(i.x)

                return pert


            print("------------------------------")


        return False
        #----------------------------------


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
        #model1.perturbationTester("../Data/ecoliDataLabelTestNew.csv",",",7,8)

        break
