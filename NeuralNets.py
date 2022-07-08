from ImportLibs import *

class Neural_Network:

    def __init__(self,InputFeatures,Output,Input_Predict,Output_Predict,Nepochs,tol,eta,batch,index_env,NNLossFnScale,json_file,model_file,polars):
        self.InputFeatures  = InputFeatures
        self.Output         = Output
        self.Input_Predict  = Input_Predict
        self.Output_Predict = Output_Predict
        self.Nepochs        = Nepochs
        self.tol            = tol
        self.eta            = eta
        self.batch          = batch
        self.index_env      = index_env
        self.NNLossFnScale  = NNLossFnScale
        self.jsonFileName = json_file
        self.WeightsFileName = model_file
        self.polarsflag = polars

    def Keras_firsttry(self):

        ########## Hyper-Parameters #################

        # No. of nuerons
        nn = 100

        # Regularization parameter
        eta = self.eta

        # Tolerance
        tol = self.tol

        # Number of Epochs
        Nepochs = self.Nepochs

        # Batch size
        batch = self.batch

        flag = self.index_env
        flag2 = self.polarsflag

        NNLossFnScale = self.NNLossFnScale

        # Model files
        json_filename = self.jsonFileName
        model_filename = self.WeightsFileName

        ############################################

        ## Some MPI stuff to increase training speed on Mac -> Not fully tested
        config = tf.ConfigProto()
        config.intra_op_parallelism_threads = 8
        config.inter_op_parallelism_threads = 8

        ###### Step 1: Loading Data ################
        
        InputFeatures       = self.InputFeatures
        Output              = self.Output

        Input_Predict       = self.Input_Predict
        Output_Predict      = self.Output_Predict

        if flag2 == 1:
            conv_file_name = "file_conv_Polars.txt"
        elif flag2 == 0:
            conv_file_name = "file_conv_Pressure.txt"
        file_conv = open(conv_file_name,'w')
        file_conv.write('Epoch Train_Acc Test_Acc \n')

        ###### Step 1.5: Splitting, shuffling and scaling the data #######
        xTrain, xTest, yTrain, yTest, LossScaleTrain, LossScaleTest = train_test_split(InputFeatures,Output,NNLossFnScale,test_size=0.25, random_state=0, shuffle =  True)
        [nps, inputDim]    = xTrain.shape
        [nps, outputDim]   = yTrain.shape
        print(inputDim)

        ## Scaling Only Outputs 
        scalerOUT = MinMaxScaler()
        scalerOUT.fit(Output)

        # Inputs Scaled outside in another routine
        # scalerIN = MinMaxScaler()
        # scalerIN.fit(InputFeatures)

        if(flag == 1):
            print("Applying Min-max scaling!")
            yTrain_scaled = scalerOUT.transform(yTrain)
            yTest_scaled = scalerOUT.transform(yTest)
            Output_scaled = scalerOUT.transform(Output)
            # xTrain_scaled = scalerIN.transform(xTrain)
            # xTest_scaled = scalerIN.transform(xTest)
        elif(flag == 0):
            print("Not applying Min-max scaling!")
            yTrain_scaled = yTrain
            yTest_scaled  = yTest

        xTrain_scaled = xTrain
        xTest_scaled  = xTest

        ###### Step 2: Defining Keras Model ########

        model = Sequential()
        model.add((Dense(nn,input_dim = inputDim)))
        model.add(LeakyReLU(alpha=0.3))

        # Uncomment to add more layers of leaky ReLU
        model.add((Dense(nn)))
        model.add(LeakyReLU(alpha=0.3))

        model.add((Dense(nn)))
        model.add(LeakyReLU(alpha=0.3))

        model.add((Dense(nn)))
        model.add(LeakyReLU(alpha=0.3))        

        model.add((Dense(nn)))
        model.add(LeakyReLU(alpha=0.3))

        model.add((Dense(nn)))
        model.add(LeakyReLU(alpha=0.3))

        # model.add((Dense(nn)))
        # model.add(LeakyReLU(alpha=0.3))

        # model.add((Dense(nn)))
        # model.add(LeakyReLU(alpha=0.3))        

        # model.add((Dense(nn)))
        # model.add(LeakyReLU(alpha=0.3))  

        # model.add((Dense(nn)))
        # model.add(LeakyReLU(alpha=0.3)) 

        # model.add((Dense(nn)))
        # model.add(LeakyReLU(alpha=0.3))  

        # model.add((Dense(nn)))
        # model.add(LeakyReLU(alpha=0.3))  

        # model.add((Dense(nn)))
        # model.add(LeakyReLU(alpha=0.3))  

        # model.add((Dense(nn)))
        # model.add(LeakyReLU(alpha=0.3))  

        # model.add((Dense(nn)))
        # model.add(LeakyReLU(alpha=0.3))  

        # model.add((Dense(nn)))
        # model.add(LeakyReLU(alpha=0.3))  

        # model.add((Dense(nn)))
        # model.add(LeakyReLU(alpha=0.3))  


        # model.add((Dense(nn)))
        # model.add(LeakyReLU(alpha=0.3))  

        # model.add((Dense(nn)))
        # model.add(LeakyReLU(alpha=0.3))  

        # model.add((Dense(nn)))
        # model.add(LeakyReLU(alpha=0.3))  

        model.add((Dense(outputDim)))

        # Uncomment to use relu/sigmoids for activation functions
        # model = Sequential()
        # model.add((Dense(nn,input_dim = inputDim, activation='relu')))
        # # model.add((Dense(nn, activation='relu')))
        # # # model.add((Dense(nn, activation='relu')))
        # # # model.add((Dense(nn, activation='sigmoid')))
        # # model.add((Dense(nn, activation='sigmoid')))
        # model.add((Dense(nn, activation='sigmoid')))
        # model.add((Dense(outputDim, activation='sigmoid')))
        # # model.add(Dense(outputDim))
        # # # now add a ReLU layer explicitly:
        # model.add(LeakyReLU(alpha=0.05))

        # ###### Step 3: Compiling Keras Model #######
        # # keras.optimizers.SGD(lr=0.1, nesterov=True)
        # # model.compile(loss=tf.keras.losses.MeanAbsolutePercentageError(), optimizer= "adam", metrics=['mse'])

        model.compile(loss="mse", optimizer= "adam", metrics=['mse'])

        ###### Step 4: Fit/Execute the model #######

        for i in range(0,Nepochs):
            print('iter:',i)
            train_history = model.fit(xTrain_scaled,yTrain_scaled, epochs=1, batch_size=batch, sample_weight=LossScaleTrain)
            print('Step done!')
            loss          = train_history.history['loss']
            print(i,loss[-1])
            if(i%100 == 0):
                _,accuracy_train = model.evaluate(xTrain_scaled,yTrain_scaled)
                _,accuracy_test = model.evaluate(xTest_scaled,yTest_scaled)

                Temp_predict_train =  model.predict(xTrain)
                Temp_predict_train = scalerOUT.inverse_transform(Temp_predict_train)

                Temp_predict_test =  model.predict(xTest)
                Temp_predict_test = scalerOUT.inverse_transform(Temp_predict_test)


                Train_error = np.subtract(yTrain,Temp_predict_train)
                Test_error  = np.subtract(yTest,Temp_predict_test)
                print('Neural Network, Train accuracy: ' + str(accuracy_train) + " Test accuracy " + str(accuracy_test))
                file_conv.write(str(i) + " " + str(accuracy_train) + " " + str(accuracy_test)  + "\n")
                file_conv.flush()
                os.fsync(file_conv.fileno())

            if(i%1000 == 0):
                model_json = model.to_json()
                with open("./NNmodel_Test.json", "w") as json_file:
                    json_file.write(model_json)
                    # serialize weights to HDF5
                model.save_weights("model_Test.h5")
                print("Saved model to disk")
            
            if(loss[-1] <= tol):
                break

        ###### Step 5: Evaluate the network accuracy ######

        _,accuracy = model.evaluate(xTrain_scaled,yTrain_scaled)
        print('Neural Network training accuracy: ',accuracy)

        _,accuracy = model.evaluate(xTest_scaled,yTest_scaled)
        print('Neural Network test accuracy: ',accuracy)

        ######## Prediction for testing/verification ##########
        # print('Predictions! 1st actual and 2nd scaled and 3d inverse transform orig!')
        # predictions = model.predict(InputFeatures)
        # predictions = scalerOUT.inverse_transform(predictions)
        # # predictions_scaleinput = model.predict(InputFeatures_scaled)

        # # predictions_invtrans = predictions
        # # ## Re-Scaling Outputs 
        # # predictions_scaled = model.predict(InputFeatures_scaled)
        # # # predictions_invtrans = predictions_scaled
        # # predictions_invtrans = scalerOUT.inverse_transform(predictions_scaled)

        # print('Original predictions')
        # print(predictions,np.amin(predictions))
       
        # # print('Scaled-back predictions')
        # # print(predictions_invtrans,np.amin(predictions_invtrans))

        # # print('Scaled-input predictions')
        # # print(predictions_scaleinput,np.amin(predictions_scaleinput))

        # print('Output')
        # print(Output,np.amin(Output))
        #######################################################

        ######## Save network as a model #############
        model_json = model.to_json()
        with open(json_filename, "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights(model_filename)
        print("Saved model to disk")

        file_conv.close()


        ######Prediction based on specified input features ########
        print(Input_Predict.shape)
        # Input_Predict_scaled = scalerIN.transform(xTrain)
        # Input_Predict_scaled = scalerIN.transform(Input_Predict);
        predictions = model.predict(Input_Predict)

        if(flag == 1):
            predictions = scalerOUT.inverse_transform(predictions)
        # # predictions_scaleinput = model.predict(InputFeatures_scaled)

        print('Predictions')
        print(predictions,np.amin(predictions))

        print('Output')
        print(Output_Predict,np.amin(Output_Predict))

        # print(np.corrcoef(np.hstack((predictions[:],Output_Predict[:]))))


    ################################################################
        ###### Save neural network #################

        if(print_NNweights == 1):
            weight1 = model.layers[0].get_weights()
            weight2 = model.layers[2].get_weights()
            weight3 = model.layers[4].get_weights()            
            # weight3 = model.layers[2].get_weights()
            # weight4 = model.layers[3].get_weights()
            # weight5 = model.layers[4].get_weights()
            # weight6 = model.layers[5].get_weights()

            isdir = os.path.isdir("input_files")

            if(isdir == 0):
                cmd = "mkdir input_files"
                os.system(cmd)
                cmd = "mkdir input_files/network"
                os.system(cmd)

            f1 = open('input_files/network/w1.dat','w+')
            for i in range(0,inputDim):
                for j in range(0,nn):
                    print("%20.18f"% (weight1[0][i][j]), file=f1)
            f1.close()


            f1 = open('input_files/network/w2.dat','w+')
            for i in range(0,nn):
                for j in range(0,outputDim):
                    print("%20.18f"% (weight2[0][i][j]), file=f1)
            f1.close()
            
            f1 = open('input_files/network/w3.dat','w+')
            for i in range(0,nn):
                for j in range(0,outputDim):
                    print("%20.18f"% (weight3[0][i][j]), file=f1)
            f1.close()

            f1 = open('input_files/network/b1.dat','w+')
            for i in range(0,nn):
                print("%20.18f"% (weight1[1][i]), file=f1)
            f1.close()


            f1 = open('input_files/network/b2.dat','w+')
            for i in range(0,outputDim):
                print("%20.18f"% (weight2[1][i]), file=f1)
            f1.close()           

            f1 = open('input_files/network/b3.dat','w+')
            for i in range(0,outputDim):
                print("%20.18f"% (weight3[1][i]), file=f1)
            f1.close()           

            print('Neural network weights saved in file!')

        return(predictions)

    def PredictNeuralNets(self,jsonFileName,WeightsFileName,Input_Predict,Output_Predict):

        Output = self.Output # need for self-scaling later
        index_env = self.index_env
        json_file = open(jsonFileName, 'r')
        loaded_model_json = json_file.read()
        json_file.close()

        loaded_model = model_from_json(loaded_model_json)
        
        # load weights into new model
        loaded_model.load_weights(WeightsFileName)
        print("Neural Network weights and model are loaded")
 
        # evaluate loaded model on test data
        loaded_model.compile(loss=tf.keras.losses.MeanAbsolutePercentageError(), optimizer= "adam", metrics=['mse'])

        ## Min-max scaling is again outputted. Just to cross-check if correct ones are being used or not. 
        ## Incorrect scaling can give rubbish results

        # print(min(Output[:,0]),max(Output[:,0]))
        # print(min(Output[:,1]),max(Output[:,1]))
        # print(min(Output[:,2]),max(Output[:,2]))
        # print(min(Output[:,3]),max(Output[:,3]))
        # print(min(Output[:,4]),max(Output[:,4]))
        # print(min(Output[:,5]),max(Output[:,5]))

        scalerOUT = MinMaxScaler()
        scalerOUT.fit(Output)

        # scalerIN = MinMaxScaler()
        # scalerIN.fit(Input_scaling)

        print(Input_Predict.shape)
        # Input_Predict_scaled = scalerIN.transform(Input_scaling)
        predictions = loaded_model.predict(Input_Predict)

        if(index_env == 1):
            predictions = scalerOUT.inverse_transform(predictions)

        _,accuracy = loaded_model.evaluate(Input_Predict,Output_Predict)
        print('Neural Net inference accuracy: ', accuracy)          

        return(predictions)
                #         Cp_upper_Train = np.linalg.norm(yTrain[:,4])
                # Cp_upper_Train_error = np.linalg.norm(Train_error[:,4])

                # Cp_lower_Train = np.linalg.norm(yTrain[:,5])
                # Cp_lower_Train_error = np.linalg.norm(Train_error[:,5])

                # Cp_upper_Train_pred = Cp_upper_Train_error/Cp_upper_Train
                # Cp_lower_Train_pred = Cp_lower_Train_error/Cp_lower_Train

                # Cp_upper_Test = np.linalg.norm(yTest[:,4])
                # Cp_upper_Test_error = np.linalg.norm(Test_error[:,4])

                # Cp_lower_Test = np.linalg.norm(yTes
                #     t[:,5])
                # Cp_lower_Test_error = np.linalg.norm(Test_error[:,5])

                # Cp_upper_Test_pred = Cp_upper_Test_error/Cp_upper_Test
                # Cp_lower_Test_pred = Cp_lower_Test_error/Cp_lower_Test
