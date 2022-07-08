from ImportLibs import *
import ReadAirfoilData
from NeuralNets import *
# Approach: 
# 1 - plot the data
# 2 - load all the data
# 3 - decide on the inputs
# 4 - pick a training set
# 5 - pick a validation set
# 6 - pick a test set
# 7 - compile results
# Might want to consider 1 model for lift, drag, moment coefficients and another for coefficient of pressure
# Send to NASA

# take out thick, camb from inputs
# less neurons
# add all NACA airfoils
# evaluate on some airfoil
########## File Name ##########
pathname = 'C:/Users/derrick/Documents/MATLAB/NASA airfoil-learning-dataset/h5/'
input_file_polars = 'NACA_Airfoils_Polars.h5'
input_file_pressure = 'NACA_Airfoils_Pressure3.h5'

TestFile_polars = 'naca0008-il - NACA 0008_Polars2.h5'
TestFile_pressure = 'naca0008-il - NACA 0008_Pressure.h5'

jsonFileName_polars = 'C:/Users/derrick/Documents/Python/NASA Airfoil/NNmodel_Test_Polars8.json'
WeightsFileName_polars = 'C:/Users/derrick/Documents/Python/NASA Airfoil/model_Test_Polars8.h5' 

jsonFileName_pressure = 'C:/Users/derrick/Documents/Python/NASA Airfoil/NNmodel_Test_Pressure10.json'
WeightsFileName_pressure = 'C:/Users/derrick/Documents/Python/NASA Airfoil/model_Test_Pressure10.h5'

########## Parameters that change for each airfoil ##########
AOA = np.linspace(-10.25,10.25,83)
print(AOA)
Re = 100000.
Ncrit = 5.

########## Neural Network Training Parameters ###########

#Regularization parameter
eta = 0.000

# Tolerance
tol = 1.0e-6
 
# Number of Epochs
Nepochs = 501

# Batch size
batch = 8192

# min-max scaling flag
flag = 1

######## Modes of Operation #########
# mode 1 = train NN model
# mode 2 = evaluate NN model some data set

mode = 2


########### Get Inputs/Output ###############
print('Reading Training Data file!')
[InputFeat_polars,OutputFeat_polars] = ReadAirfoilData.ReadInputOutputFeatures_Polars(pathname+input_file_polars)
[InputFeat_pressure,OutputFeat_pressure] = ReadAirfoilData.ReadInputOutputFeatures_Pressure(pathname+input_file_pressure)
[npts_polars,b] = InputFeat_polars.shape
[npts_pressure,b] = InputFeat_pressure.shape
print('Input shape')
print(InputFeat_polars.shape,InputFeat_pressure.shape)
print('Output shape')
print(OutputFeat_polars.shape,OutputFeat_pressure.shape)

print('Reading Testing Data file!')
[Input_test_polars,Output_test_polars] = ReadAirfoilData.ReadInputOutputFeatures_Polars(pathname+TestFile_polars)
[Input_test_pressure,Output_test_pressure] = ReadAirfoilData.ReadInputOutputFeatures_Pressure(pathname+TestFile_pressure)

print('Scaling Inputs')
InputFeat_polars_scaled = ReadAirfoilData.DetermineScaledInputs(InputFeat_polars)
Input_test_polars_scaled = ReadAirfoilData.DetermineScaledInputs(Input_test_polars)

InputFeat_pressure_scaled = ReadAirfoilData.DetermineScaledInputs(InputFeat_pressure)
Input_test_pressure_scaled = ReadAirfoilData.DetermineScaledInputs(Input_test_pressure)

[Input_test_polars,b] = ReadAirfoilData.ReadInputOutputFeatures_Polars(pathname+TestFile_polars)
[Input_test_pressure,c] = ReadAirfoilData.ReadInputOutputFeatures_Pressure(pathname+TestFile_pressure)

############### Neural Networks Training ################
NNLossFnScale_polars = np.ones([npts_polars])

NNLossFnScale_pressure = np.ones([npts_pressure])
if mode == 1:
	Input_Predict_polars = InputFeat_polars_scaled
	Output_Predict_polars = OutputFeat_polars

	Input_Predict_pressure = InputFeat_pressure_scaled
	Output_Predict_pressure = OutputFeat_pressure

	print('Input Prediction shape')
	print(Input_Predict_polars.shape,Input_Predict_pressure.shape)
	print('Output Prediction shape')
	print(Output_Predict_pressure.shape,Output_Predict_pressure.shape)

	# print('Training Neural Network Model for Polars')
	# NNetwork1 = Neural_Network(InputFeat_polars_scaled,OutputFeat_polars,Input_Predict_polars,Output_Predict_polars,Nepochs,tol,eta,batch,flag,NNLossFnScale_polars,jsonFileName_polars,WeightsFileName_polars,1)
	# predictions1 = NNetwork1.Keras_firsttry()

	print('Training Neural Network Model for Pressure')
	NNetwork2 = Neural_Network(InputFeat_pressure_scaled,OutputFeat_pressure,Input_Predict_pressure,Output_Predict_pressure,Nepochs,tol,eta,batch,flag,NNLossFnScale_pressure,jsonFileName_pressure,WeightsFileName_pressure,0)
	predictions1 = NNetwork2.Keras_firsttry()
	print('Done!')

############### NN Evaluation and Post-Processing #################
if mode == 2:
	for j in range(len(AOA)):
		print(j)
		print('Neural Network Inference Mode')
		
		print('Getting Inference Data!')
		# cut data
		# print([Input_test_polars[0,0],Input_test_polars[0,1]])

		[Input_Predict_polars,Output_Predict_polars] = ReadAirfoilData.GetEvaluationData(Input_test_polars,Input_test_polars_scaled,Output_test_polars,AOA[j],Re,Ncrit)
		[Input_Predict_pressure,Output_Predict_pressure] = ReadAirfoilData.GetEvaluationData(Input_test_pressure,Input_test_pressure_scaled,Output_test_pressure,AOA[j],Re,Ncrit)

		# Neural_Network inputs: all inputs from test file, all output from training, desired input and output data 
		Output_Scaling_polars = OutputFeat_polars
		Output_Scaling_pressure = OutputFeat_pressure

		print('Dimensions:')
		# print(Input_test_polars.shape,Output_Scaling_polars.shape,Input_Predict_polars.shape,Output_Predict_polars.shape)
		print(Input_test_pressure.shape,Output_Scaling_pressure.shape,Input_Predict_pressure.shape,Output_Predict_pressure.shape)
		
		# NNetwork1 = Neural_Network(Input_test_polars,Output_Scaling_polars,Input_Predict_polars,Output_Predict_polars,Nepochs,tol,eta,batch,flag,NNLossFnScale_polars,jsonFileName_polars,WeightsFileName_polars,1)	
		# predictions1 = NNetwork1.PredictNeuralNets(jsonFileName_polars,WeightsFileName_polars,Input_Predict_polars,Output_Predict_polars)

		NNetwork2 = Neural_Network(Input_test_pressure,Output_Scaling_pressure,Input_Predict_pressure,Output_Predict_pressure,Nepochs,tol,eta,batch,flag,NNLossFnScale_pressure,jsonFileName_pressure,WeightsFileName_pressure,0)	
		predictions2 = NNetwork2.PredictNeuralNets(jsonFileName_pressure,WeightsFileName_pressure,Input_Predict_pressure,Output_Predict_pressure)
		print('Writing Output Files')
		output_file = 'C:/Users/derrick/Documents/MATLAB/NASA airfoil-learning-dataset/NNpredict/Test_NNpredict_AOA_'+str(AOA[j])+'_Re_'+str(Re)+'_Ncrit_'+str(Ncrit)+'.h5'
		actual_data_file = 'C:/Users/derrick/Documents/MATLAB/NASA airfoil-learning-dataset/ActualData/Test_Actual_AOA_'+str(AOA[j])+'_Re_'+str(Re)+'_Ncrit_'+str(Ncrit)+'.h5'
	
		# Cd_output = predictions1[:,0]
		# Cdp_output = predictions1[:,1]
		# Cl_output = predictions1[:,2]
		# Cm_output = predictions1[:,3]
		Cp_upper_output = predictions2[:,0]
		Cp_lower_output = predictions2[:,1]

		# Neural Network Prediction
		# ReadAirfoilData.WriteOutputFile(output_file,Cd_output,Cdp_output,Cl_output,Cm_output,Cp_upper_output,Cp_lower_output)
		# ReadAirfoilData.WriteOutputFile(output_file,Cd_output,Cdp_output,Cl_output,Cm_output,1,2)
		ReadAirfoilData.WriteOutputFile(output_file,1,2,3,4,Cp_upper_output,Cp_lower_output)
		# ReadAirfoilData.WriteOutputFile(output_file,1,2,3,4,Cp_upper_output,5)

		# ReadAirfoilData.WriteOutputFile(output_file,Cd_output,1,2,3,Cp_upper_output,Cp_lower_output)
		# ReadAirfoilData.WriteOutputFile(output_file,Cd_output,1,2,3,4,5)

		# Actual Data
		# ReadAirfoilData.WriteOutputFile(actual_data_file,Output_Predict_polars[:,0],Output_Predict_polars[:,1],Output_Predict_polars[:,2],Output_Predict_polars[:,3],Output_Predict_pressure[:,0],Output_Predict_pressure[:,1])
		# ReadAirfoilData.WriteOutputFile(actual_data_file,Output_Predict_polars[:,0],Output_Predict_polars[:,1],Output_Predict_polars[:,2],Output_Predict_polars[:,3],1,2)
		# ReadAirfoilData.WriteOutputFile(actual_data_file,Output_Predict_polars[:,0],1,2,3,Output_Predict_pressure[:,0],Output_Predict_pressure[:,1])
		# ReadAirfoilData.WriteOutputFile(actual_data_file,Output_Predict_polars[:,0],1,2,3,4,5)
		ReadAirfoilData.WriteOutputFile(actual_data_file,1,2,3,4,Output_Predict_pressure[:,0],Output_Predict_pressure[:,1])
		# ReadAirfoilData.WriteOutputFile(actual_data_file,1,2,3,4,Output_Predict_pressure[:,0],1)

		# print(Cd_output.shape, Output_Predict_polars.shape)
		# print('skin drag coefficent relative error')
		# print(np.fabs((Cd_output[:] - Output_Predict_polars[:,0])/OutputFeat_polars[:,0]))
		# print('pressure correlation ')
		# print(np.corrcoef([Output_Predict_pressure[:,0],Cp_lower_output[:]]))
		print('Done!')
		# exit()
# Make a flow chart and example NN architecture for better organization of stuff
# use same output features from training in the evaluation for neural network call