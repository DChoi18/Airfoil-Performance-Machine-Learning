from ImportLibs import *
import ReadAirfoilData

def main():
	path = 'C:/Users/derrick/Documents/MATLAB/NASA airfoil-learning-dataset/h5/'
	file_name = 'TestSet_Pressure.h5'

	f = path+file_name

	[Input,Output] = ReadAirfoilData.ReadInputOutputFeatures_Pressure(f)

	print('Input shape')
	print(Input.shape)
	print('Output shape')
	print(Output.shape)

	print('First AOA = ',Input[0,8])
	print('Re = ',Input[0,0])
	print('Ncrit = ',Input[0,1])
	AOA = -10.25
	Re = 100000.
	Ncrit = 5.
	[Input_Predict,Output_Predict] = ReadAirfoilData.GetEvaluationData(Input,Output,AOA,Re,Ncrit)

	print(Input_Predict.shape,Output_Predict.shape)
	print(Input_Predict[:,8])
	print(Input_Predict[:,0])
	print(Input_Predict[:,1])

	# for i in range(len(Input[:,8])):
	# 	if i%100 == 0:
	# 		print(Input[i,8])

if __name__ == '__main__':
	main()