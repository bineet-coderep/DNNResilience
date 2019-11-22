'''
This code is to test (ad-hoc) how perturbation
affects classification results of a DNN.
Given a set of architecture and datasets
three DNNs are generated and then the effects of perturbation
is tested.

Please note that this code is a platform to perform brute-force testing.
No mathematical gurantee is provided. This is a 'scape-goat'
to see how perturbation can affect results.
'''

'''
Parameters for fixing the architecture
of the three DNN
'''

from keras.models import Sequential
from keras.layers import Dense
import numpy
from gurobipy import *

#DNN 1
INP_SIZE1=7
OUT_SIZE1=8
NO_LAYERS1=4
NEURONS1=[8,8,8,8]
DATASET_FILE1="../Data/ecoliData.csv"
DELIMETER1=","
EPOCH1=2000
BATCH1=15

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

class DNNTrainer:

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
        scores = self.model.evaluate(self.X, self.Y)
        return (scores[1])

    def getWeights(self,layer):
        return self.model.layers[layer].get_weights()

    def getPredictions(self,fname,d,inp_s,out_s):
        datasetInp = numpy.loadtxt(fname, delimiter=d)
        I = datasetInp[:,:inp_s]
        O = datasetInp[:,inp_s:(inp_s+out_s)]
        predictions = self.model.predict(I)

        print("\n\n============ Prediction Results ============\n")

        for i in range(len(O)):
            indx=-7
            max=-9999
            for j in range(len(predictions[i])):
                if predictions[i][j]>max:
                    max=predictions[i][j]
                    indx=j
            print("Data Set - Predicted Label ",i,": ",(indx+1))
            for j in range(len(O[i])):
                if (O[i][j]==1):
                    print("Data Set - Actual Label ",i,": ",(j+1))
                    print("")

    def perturbationTester(self,fname,d,inp_s,out_s):
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


class GurobiTest:

    '''To be removed later
    '''

    def __init__(self):
        print("Test")

    def findBound(self):
        model = Model("qp")

        varX=model.addVar(-GRB.INFINITY,GRB.INFINITY,name="X",vtype='C')
        varY=model.addVar(-GRB.INFINITY,GRB.INFINITY,name="Y",vtype='C')
        varZ=model.addVar(-GRB.INFINITY,GRB.INFINITY,name="Z",vtype='C')
        model.addConstr(varX>=0,"Min Constraint")
        model.addConstr(varX<=1,"Max Constraint")
        model.addConstr(varZ==-4*varY+5,"Z Constraint")
        model.addConstr(varY==-7*varX+3,"Y Constraint")

        f=6*varZ+7

        model.setObjective(f,GRB.MAXIMIZE)
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
                print("\n\nValues\n\n")
                for v in model.getVars():
                    print('%s %g' % (v.varName, v.x))
                print('Obj: %g' % f.getValue())

        ## Extend to max(0,f(x))


class PerturbationBounds:

    def __init__(self,dnn):
        self.dnnTrained=dnn #Type: DNNTrainer

    def evalFunc(W,X,B):
        X=filterZero(X)
        Y=add(matmul(W,X),B)
        return Y

    def filterZero(X):
        for i in range(len(X)):
            if X[i]<0:
                X[i]=0
        return X

    def findBound2(self):

        #Find the lower bound and the upper bound for label i (given)
        #Find the maximum \delta tolerance within the same bound
        print("Under Construction!!")

        print(self.dnnTrained.getWeights(1))

    def createSingleMatrix(A,B):
        ''' Creates a single matrix based on
        by joining the weight and the bias
        matrix'''

        r=A.shape[0]
        c=A.shape[1]
        C=numpy.zeros((r+1,c+1),dtype=numpy.float128)

        for i in range(r):
            for j in range(c):
                C[i][j]=A[i][j]
        for i in range(r):
            C[i][c]=B[i][0]
        C[r][c]=1

        return C

    def layerMatrices(self,maxStatusList):
        '''W0=numpy.array([
        [3,0,0,0,0,2,4],
        [1,2,3,0,0,2.9,2.9],
        [8,1,2,0,0,2.9,2.9],
        [7,0,0,8,2,3.9,3.9],
        [8,0,0,3,7,3.9,3.9],
        [0,0,0,0,0,6,3],
        [0,0,0,0,0,2,1],
        ])
        B0=numpy.array([
        [3],
        [0],
        [0],
        [9],
        [11],
        [0],
        [0],
        ])
        W1=numpy.array([
        [6,3,0,2,1,2,5],
        [1,4,3,1,2,2.7,5.2],
        [7,3,2,0,2,3.9,2.2],
        [6,2,0,2,2,4.2,2.1],
        [5,2,0,6,7,2,2],
        [2,3,1,6,0,3,2],
        [4,0,2,0,1,2,4],
        ])
        B1=numpy.array([
        [2],
        [5],
        [2],
        [1],
        [10],
        [4],
        [2],
        ])
        W2=numpy.array([
        [2,6,3,2,4,2,1],
        [2,3,2,6,5,2.2,1.4],
        [8,2,4,2,5,3.2,4.1],
        [4,1,1,4,3,2.5,6.1],
        [7,3,1,3,8,2,3],
        [4,1,0,0,0,1,2],
        [6,2,4,2,8,1,3],
        ])
        B2=numpy.array([
        [3],
        [1],
        [5],
        [2],
        [1],
        [5],
        [3],
        ])

        matList=[[W2,B2],[W1,B1],[W0,B0]]

        '''

        matList=[]
        for i in range(NO_LAYERS1,-1,-1):
            l=[]
            w=self.dnnTrained.getWeights(i)[0] #Keras Matrix

            '''for j in range(len(maxStatusList[i])):
                if maxStatusList[i][j]==0:
                    # jth Row becomes 0
                    for k in range(w.shape[1]):
                        w[j][k]=0'''

            l=[numpy.transpose(w),numpy.reshape(self.dnnTrained.getWeights(i)[1],(self.dnnTrained.getWeights(i)[1].size,1))]
            matList.append(l)

        return matList

    def formulateNetFunc(self,matList):

        fun=numpy.identity(OUT_SIZE1+1)
        for i in matList:
            t0=PerturbationBounds.createSingleMatrix(i[0],i[1])
            fun=numpy.matmul(fun,t0)

        return fun

    def solveBoundOpti(self,fun):

        model = Model("qp")

        #Create Variables
        r=fun.shape[0]
        c=fun.shape[1]

        inputVars=[]
        for i in range(c-1):
            name="x."+str(i)
            inputVars.append(model.addVar(-GRB.INFINITY,GRB.INFINITY,name=name,vtype='C'))
        inputVars.append(model.addVar(-GRB.INFINITY,GRB.INFINITY,name="x.dummy",vtype='C'))

        perturbVars=[]
        for i in range(c-1):
            name="delta."+str(i)
            perturbVars.append(model.addVar(-GRB.INFINITY,GRB.INFINITY,name=name,vtype='C'))
        perturbVars.append(model.addVar(-GRB.INFINITY,GRB.INFINITY,name="delta.dummy",vtype='C'))


        #Add Input Constraint (Under Research)
        '''for i in range(c-1):
            model.optimize()
            name="Input-C"+str(i)
            model.addConstr(inputVars[i]>=-5,name+".1")
            model.addConstr(inputVars[i]<=5,name+".2")'''

        model.addConstr(inputVars[0]==0.1)
        model.addConstr(inputVars[1]==0.49)
        model.addConstr(inputVars[2]==0.48)
        model.addConstr(inputVars[3]==0.5)
        model.addConstr(inputVars[4]==0.41)
        model.addConstr(inputVars[5]==0.67)
        model.addConstr(inputVars[6]==0.21)

        '''model.addConstr(inputVars[0]<=0.1+EPSILON)
        model.addConstr(inputVars[0]>=0.1-EPSILON)
        model.addConstr(inputVars[1]<=0.49+EPSILON)
        model.addConstr(inputVars[1]>=0.49-EPSILON)
        model.addConstr(inputVars[2]<=0.48+EPSILON)
        model.addConstr(inputVars[2]>=0.48-EPSILON)
        model.addConstr(inputVars[3]<=0.5+EPSILON)
        model.addConstr(inputVars[3]>=0.5-EPSILON)
        model.addConstr(inputVars[4]<=0.41+EPSILON)
        model.addConstr(inputVars[4]>=0.41-EPSILON)
        model.addConstr(inputVars[5]<=0.67+EPSILON)
        model.addConstr(inputVars[5]>=0.67-EPSILON)
        model.addConstr(inputVars[6]<=0.21+EPSILON)
        model.addConstr(inputVars[6]>=0.21-EPSILON)'''

        for i in range(c-1):
            model.optimize()
            name="Fault-C"+str(i)
            model.addConstr(perturbVars[i]>=-1,name+".1")
            model.addConstr(perturbVars[i]<=1,name+".2")


        name="Input-C"+str(c-1)
        model.addConstr(inputVars[c-1]>=1,name+".1")
        model.addConstr(inputVars[c-1]<=1,name+".2")
        name="Perturb-C"+str(c-1)
        model.addConstr(perturbVars[c-1]>=0,name+".1")
        model.addConstr(perturbVars[c-1]<=0,name+".2")

        #Add function encoding as Constraint
        #Let us assume this for class label 0

        individualFuncs=[]
        for i in range(r):
            f=0
            for j in range(c):
                f=f+(fun[i][j]*(inputVars[j]+perturbVars[j]))
            individualFuncs.append(f)

        setLabel=1
        for i in range(r-1):
            if i != setLabel:
                name="Fun-C"+str(i)
                model.addConstr(individualFuncs[setLabel]+EPSILON>=individualFuncs[i],name)


        #model.addConstr(3*inputVars[0]+perturbVars[0] <= 4000*(inputVars[1]+perturbVars[1]),"Test")


        # Objective Function
        obj=0
        for i in range(c-1):
            obj=obj+(perturbVars[i]*perturbVars[i])

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
                print("\n\nCounter Example\n\n")
                for v in model.getVars():
                    print('%s %g' % (v.varName, v.x))
                #print('Obj: %g' % obj.getValue())
                return True


            print("------------------------------")

        return False

    def findBound(self):
        print("Started!")
        maxStatusList1=[[1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1]]
        maxStatusList2=[[1,1,1,1,1,1,1,1,1],[1,1,1,1,1,0,1,1],[1,1,1,1,1,1,1,1],[1,1,1,0,1,1,1,1],[1,1,1,1,1,1,1,1]]
        maxStatusList3=[[1,1,1,1,1,1,1,1,1],[1,1,1,0,1,1,1,1],[1,1,1,1,1,1,1,1],[1,0,1,1,1,1,1,1],[1,1,1,0,1,1,1,1]]
        maxStatusList4=[[1,1,1,1,1,1,1,1,1],[1,1,1,1,1,0,1,1],[1,1,1,1,1,1,1,1],[1,1,1,0,1,1,1,1],[1,1,1,1,1,1,1,1]]
        #maxStatusList=[[1,1,1,1,1,1,1,1,1],[0,1,0,0,0,0,0,1],[0,1,0,0,0,0,0,1],[0,1,0,0,0,0,0,1],[0,1,0,0,0,0,0,1]]
        maxActivity=[maxStatusList1,maxStatusList2,maxStatusList3,maxStatusList4]
        f=True
        for l in maxActivity:
            layerMats=self.layerMatrices(l)
            netFunction=self.formulateNetFunc(layerMats)
            f=f and (self.solveBoundOpti(netFunction))
        return f











'''model1=DNNTrainer(INP_SIZE1,OUT_SIZE1,NO_LAYERS1,NEURONS1,DATASET_FILE1,DELIMETER1,EPOCH1,BATCH1)
model2=DNNTrainer(INP_SIZE2,OUT_SIZE2,NO_LAYERS2,NEURONS2,DATASET_FILE2,DELIMETER2,EPOCH2,BATCH2)
model3=DNNTrainer(INP_SIZE3,OUT_SIZE3,NO_LAYERS3,NEURONS3,DATASET_FILE3,DELIMETER3,EPOCH3,BATCH3)

print("\n\n================= DNN Summary =================")
print("DNN 1 => Accuracy: ",model1.getAccuracy()*100)
print("DNN 2 => Accuracy: ",model2.getAccuracy()*100)
print("DNN 3 => Accuracy: ",model3.getAccuracy()*100)

model1.perturbationTester(DATASET_FILE1,DELIMETER1,INP_SIZE1,OUT_SIZE1)
'''

'''
model4=DNNTrainer(4,4,2,[2,4],"../Data/tstdata2.csv",DELIMETER1,EPOCH1,BATCH1)
print("\n\n================= DNN Summary =================")
print("DNN 1 => Accuracy: ",model4.getAccuracy()*100)
print(model4.getWeights(0))
print(model4.getWeights(1))
print(model4.getWeights(2))
print(model4.getWeights(3))
'''

'''model1=DNNTrainer(INP_SIZE1,OUT_SIZE1,NO_LAYERS1,NEURONS1,DATASET_FILE1,DELIMETER1,EPOCH1,BATCH1)
#model1.getPredictions("../Data/ecoliDataLabel1.csv",",",7,8)
pb=PerturbationBounds(model1)
pb.findBound()'''


flag=False

while(flag!=True):
    model1=DNNTrainer(INP_SIZE1,OUT_SIZE1,NO_LAYERS1,NEURONS1,DATASET_FILE1,DELIMETER1,EPOCH1,BATCH1)
    #model1.getPredictions("../Data/ecoliDataLabel1.csv",",",7,8)
    pb=PerturbationBounds(model1)
    flag=pb.findBound()
    model1.getPredictions("../Data/ecoliDataLabel1.csv",",",7,8)
    print("Acceptance: ",flag)
