from ImportLibs import *

# things to do: stack these into usable arrays, test this function, create another function to get Evaluation data set


def ReadInputOutputFeatures_Polars(filename):
	print('Reading File')
	f = h5py.File(filename,'r')
	# Inputs
	Re = f['Re']
	Ncrit = f['Ncrit']
	# thick = f['thickness']
	# camb = f['camber']
	max_thick = f['max_thick']
	max_camb = f['max_camb']
	pos_max_camb = f['pos_max_camb']
	pos_max_thick = f['pos_max_t']
	alpha = f['alpha']
	# Outputs
	Cd = f['Cd']
	Cdp = f['Cdp']
	Cl = f['Cl']
	Cm = f['Cm']


	# convert to numpy arrays
	Re = np.array(Re[:],ndmin = 2)
	Ncrit = np.array(Ncrit[:],ndmin = 2)
	# thick = np.array(thick[:],ndmin = 2)
	# camb = np.array(camb[:],ndmin = 2)
	max_thick = np.array(max_thick[:],ndmin = 2)
	max_camb = np.array(max_camb[:],ndmin = 2)
	pos_max_camb = np.array(pos_max_camb[:],ndmin = 2)
	pos_max_thick = np.array(pos_max_thick[:],ndmin = 2)
	alpha = np.array(alpha[:],ndmin = 2)
	Cd = np.array(Cd[:],ndmin = 2)
	Cdp = np.array(Cdp[:],ndmin = 2)
	Cl = np.array(Cl[:],ndmin = 2)
	Cm = np.array(Cm[:],ndmin = 2)

	InpFeat = np.vstack((Re,Ncrit,max_thick,max_camb,pos_max_thick,pos_max_camb,alpha))
	Output = np.vstack((Cd,Cdp,Cl,Cm))
	# Output = np.vstack((Cd,Cdp))
	# Output = Cd
	InpFeat = np.transpose(InpFeat)
	Output = np.transpose(Output)

	return InpFeat,Output

def ReadInputOutputFeatures_Pressure(filename):
	print('Reading File')
	f = h5py.File(filename,'r')
	# Inputs
	Re = f['Re']
	Ncrit = f['Ncrit']
	thick = f['thickness']
	camb = f['camber']
	max_thick = f['max_thick']
	max_camb = f['max_camb']
	pos_max_camb = f['pos_max_camb']
	pos_max_thick = f['pos_max_t']
	xu = f['xu_coord']
	xl = f['xl_coord']
	yu = f['yu_coord']
	yl = f['yl_coord']
	alpha = f['alpha']
	# Outputs
	Cp_upper = f['Cp_ps']
	Cp_lower = f['Cp_ss']


	# convert to numpy arrays
	Re = np.array(Re[:],ndmin = 2)
	Ncrit = np.array(Ncrit[:],ndmin = 2)
	thick = np.array(thick[:],ndmin = 2)
	camb = np.array(camb[:],ndmin = 2)
	max_thick = np.array(max_thick[:],ndmin = 2)
	max_camb = np.array(max_camb[:],ndmin = 2)
	pos_max_camb = np.array(pos_max_camb[:],ndmin = 2)
	pos_max_thick = np.array(pos_max_thick[:],ndmin = 2)
	xu = np.array(xu[:],ndmin = 2)
	xl = np.array(xl[:],ndmin = 2)
	yu = np.array(yu[:],ndmin = 2)
	yu = np.array(yl[:],ndmin = 2)
	alpha = np.array(alpha[:],ndmin = 2)
	Cp_upper = np.array(Cp_upper[:],ndmin = 2)
	Cp_lower = np.array(Cp_lower[:],ndmin = 2)
	
	InpFeat = np.vstack((Re,Ncrit,thick,camb,xu,yu,yl,max_thick,max_camb,pos_max_camb,pos_max_thick,alpha))
	# InpFeat = np.vstack((Re,Ncrit,xu,xl,yu,yl,thick,camb,alpha))
	Output = np.vstack((Cp_upper,Cp_lower))
	# Output = Cp_upper

	InpFeat = np.transpose(InpFeat)
	Output = np.transpose(Output)

	return InpFeat,Output

def GetEvaluationData(InputFeatures,Inputs_Scaled,Output,AOA,Re,Ncrit):

	[npts,nfeat] = InputFeatures.shape
	print(InputFeatures.shape)
	tol = 1e-6

	Re_index = 0
	Ncrit_index = 1

	alpha_index = nfeat-1
	
	print('angle of attack = ',AOA)
	InputFeatures_Predict = []
	Output_Predict = []
	flag = 0
	for i  in range(npts):
		if (np.fabs(AOA - InputFeatures[i,alpha_index])) <= tol\
			and np.fabs(Re- InputFeatures[i,Re_index])<= tol and np.fabs(Ncrit - InputFeatures[i,Ncrit_index]) <= tol:

			InputFeatures_Predict.append(Inputs_Scaled[i,:])
			Output_Predict.append(Output[i,:])
			
			flag = 1
		else:
			if(flag == 1 and np.fabs(AOA-InputFeatures[i,alpha_index])>tol)\
				and np.fabs(Re- InputFeatures[i,Re_index]) > tol and np.fabs(Ncrit - InputFeatures[i,Ncrit_index]) > tol:
				break
	return(np.asarray(InputFeatures_Predict),np.asarray(Output_Predict))

def DetermineScaledInputs(InputFeatures):
	[a,nfeat] = InputFeatures.shape
	Inputs_Scaled = InputFeatures
	Inputs_Scaled = np.asarray(Inputs_Scaled)

	for i in range(nfeat):
		xmin = np.min(Inputs_Scaled[:,i])
		xmax = np.max(Inputs_Scaled[:,i])
		if np.abs(xmax-xmin) < 1e-8:
			xscaled = Inputs_Scaled[:,i]
		else:	
			xscaled = np.divide(Inputs_Scaled[:,i]-xmin,xmax-xmin)
		Inputs_Scaled[:,i] = xscaled

	return Inputs_Scaled

def WriteOutputFile(h5filename,Cd,Cdp,Cl,Cm,Cp_upper,Cp_lower):
	f = h5py.File(h5filename,'w')

	# f.create_dataset('Cd',data = Cd)
	# f.create_dataset('Cdp',data = Cdp)
	# f.create_dataset('Cl',data = Cl)
	# f.create_dataset('Cm',data = Cm)
	f.create_dataset('Cp_upper',data = Cp_upper)
	f.create_dataset('Cp_lower',data = Cp_lower)


	# Cp_upper = np.array(Cp_upper[:],ndmin = 2)
	# Cp_lower = np.array(Cp_lower[:],ndmin = 2)