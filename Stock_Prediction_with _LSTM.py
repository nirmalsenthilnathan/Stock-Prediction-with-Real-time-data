from backtester.trading_system import TradingSystem
from backtester.version import updateCheck
from backtester.features.feature import Feature
import numpy as np
import pandas as pd
from asia_ds1_toolbox.problem1.problem1_trading_params import MyTradingParams
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from keras import optimizers
from tqdm._tqdm_notebook import tqdm_notebook
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
import os
# from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import train_test_split

##################################################################################
##################################################################################
## Template file for problem 1.                                                 ##
##################################################################################
## Make your changes to the functions below.
## SPECIFY features you want to use in getInstrumentFeatureConfigDicts()
## Fill predictions in getPrediction()
## The toolbox does the rest for you
## from downloading and loading data to running backtest
##################################################################################

def build_timeseries(mat, y_col_index, TIME_STEPS):
    # y_col_index is the index of column that would act as output column
    # total number of time-series samples would be len(mat) - TIME_STEPS
    dim_0 = mat.shape[0] - TIME_STEPS
    dim_1 = mat.shape[1]
    x = np.zeros((dim_0, TIME_STEPS, dim_1))
    y = np.zeros((dim_0,))
    
    for i in tqdm_notebook(range(dim_0)):
        x[i] = mat[i:TIME_STEPS+i]
        y[i] = mat[TIME_STEPS+i, y_col_index]
    print("length of time-series i/o",x.shape,y.shape)
    return x, y

def trim_dataset(mat, batch_size):
    """
    trims dataset to a size that's divisible by BATCH_SIZE
    """
    no_of_rows_drop = mat.shape[0]%batch_size
    if(no_of_rows_drop > 0):
        return mat[:-no_of_rows_drop]
    else:
        return mat

def prediction_model(BATCH_SIZE, TIME_STEPS, x_t):
    lstm_model = Sequential()
    # (batch_size, timesteps, data_dim)
    lstm_model.add(LSTM(40, batch_input_shape=(BATCH_SIZE, TIME_STEPS, x_t.shape[2]),
                        dropout=0.0, recurrent_dropout=0.0, stateful=True, return_sequences=True,
                        kernel_initializer='random_uniform'))
    lstm_model.add(Dropout(0.4))
    lstm_model.add(LSTM(20, dropout=0.0))
    lstm_model.add(Dropout(0.4))
    lstm_model.add(Dense(10,activation='relu'))
    lstm_model.add(Dense(1,activation='sigmoid'))
    optimizer = optimizers.RMSprop(lr=0.000001)
    # optimizer = optimizers.SGD(lr=0.000001, decay=1e-6, momentum=0.9, nesterov=True)
    lstm_model.compile(loss='mean_squared_error', optimizer=optimizer)
    
    return lstm_model

def model(self, time, updateNum, instrumentList, instrumentFeatures, MarketFeatures, instrumentFeatureList, targetVariable, predictions, s, i, temp):
    if os.path.exists("Checkpoint") == False:
        os.makedirs("Checkpoint")
    if os.path.exists("Model") == False:
        os.makedirs("Model")
    
    
    BATCH_SIZE = 20
    TIME_STEPS = 15
    IDs = MyTradingFunctions()
    companies = IDs.getInstrumentIds()
    for company in companies:
        if os.path.exists("Checkpoint\\"+str(company)) == False:
            os.makedirs("Checkpoint\\"+str(company))
        if os.path.exists("Model\\"+str(company)) == False:
            os.makedirs("Model\\"+str(company))
        
        OUTPUT = os.path.join("Checkpoint\\"+str(company))
        OUTPUT_MODEL = os.path.join("Model\\"+str(company))
        
    #    OUTPUT = "Checkpoint"
    #    OUTPUT_MODEL = "Model"
        
        if os.path.exists("Checkpoint\\"+str(company)+'model.h5') == False:
            df_ge = pd.read_csv(os.path.join("historicalData\\qq16p1Data\\" + s +".csv"), engine='python')
            train_cols=instrumentFeatureList
            df_train, df_test = train_test_split(df_ge, train_size=0.8, test_size=0.2, shuffle=False)
            print("Train and Test size", len(df_train), len(df_test))
            
            x = df_train.loc[:,train_cols].values
            x_train = x
            x_test = df_test.loc[:,train_cols]
            x_t, y_t = build_timeseries(x_train, 3, TIME_STEPS)
            x_t = trim_dataset(x_t, BATCH_SIZE)
            y_t = trim_dataset(y_t, BATCH_SIZE)
            x_temp, y_temp = build_timeseries(x_test, 3)
            x_val, x_test_t = np.split(trim_dataset(x_temp, BATCH_SIZE),2)
            y_val, y_test_t = np.split(trim_dataset(y_temp, BATCH_SIZE),2)
            
            lstm_model = prediction_model(BATCH_SIZE, TIME_STEPS, x_t)
            
            es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,
                               patience=40, min_delta=0.0001)
            
            model = ModelCheckpoint(os.path.join(OUTPUT_MODEL,"\\model.h5"), monitor='val_loss', verbose=1,
                              save_best_only=True, save_weights_only=False, mode='min', period=1)
            
            csv_logger = CSVLogger(os.path.join(OUTPUT, "company" + '.log'), append=True)
            
            history = lstm_model.fit(x_t, y_t, epochs=40, verbose=2, batch_size=BATCH_SIZE,
                                shuffle=False, validation_data=(trim_dataset(x_val, BATCH_SIZE),
                                trim_dataset(y_val, BATCH_SIZE)), callbacks=[es, model, csv_logger])
            
            save_model(OUTPUT_MODEL)
          
        if os.path.exists("Checkpoint\\"+str(company)+'model1.h5') == False:
            df_ge = pd.read_csv(os.path.join("historicalData\\qq16p1Data\\" + s +".csv"), engine='python')
            if temp == 1:
                df_ge=df_ge.iloc[:, :-1]
            train_cols=instrumentFeatureList
            df_train, df_test = train_test_split(df_ge, train_size=0.8, test_size=0.2, shuffle=False)
            print("Train and Test size", len(df_train), len(df_test))
            
            x = df_train.loc[:,train_cols].values
            x_train = x
            x_test = df_test.loc[:,train_cols]
            x_t, y_t = build_timeseries(x_train, 3, TIME_STEPS)
            x_t = trim_dataset(x_t, BATCH_SIZE)
            y_t = trim_dataset(y_t, BATCH_SIZE)
            x_temp, y_temp = build_timeseries(x_test, 3)
            x_val, x_test_t = np.split(trim_dataset(x_temp, BATCH_SIZE),2)
            y_val, y_test_t = np.split(trim_dataset(y_temp, BATCH_SIZE),2)
            
            lstm_model = prediction_model(BATCH_SIZE, TIME_STEPS, x_t)
            
            es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,
                               patience=40, min_delta=0.0001)
            
            model = ModelCheckpoint(os.path.join(OUTPUT_MODEL,"\\model1.h5"), monitor='val_loss', verbose=1,
                              save_best_only=True, save_weights_only=False, mode='min', period=1)
            
            csv_logger = CSVLogger(os.path.join(OUTPUT, "company" + '.log'), append=True)
            
            history = lstm_model.fit(x_t, y_t, epochs=40, verbose=2, batch_size=BATCH_SIZE,
                                shuffle=False, validation_data=(trim_dataset(x_val, BATCH_SIZE),
                                trim_dataset(y_val, BATCH_SIZE)), callbacks=[es, model, csv_logger])
            
            save_model(OUTPUT_MODEL)    

        if temp==0:
            model = loaded_model(OUTPUT_MODEL, temp)
        else:
            model = loaded_model(OUTPUT_MODEL, temp)
    
    return  model    

def save_model(OUTPUT_MODEL):
    model.save(os.path.join(OUTPUT_MODEL,'\\model.h5'))
    print("Saved model")
    return loaded_model

def loaded_model(OUTPUT_MODEL, temp):
    if temp == 0:
        loaded_model = model.load(os.path.join(OUTPUT_MODEL,'\\model.h5'))
    if temp == 1:
        loaded_model = model.load(os.path.join(OUTPUT_MODEL,'\\model.h5'))
    print("Loaded model")
    return loaded_model
        

class MyTradingFunctions():

	def __init__(self):  #Put any global variables here
		self.params = {}


	###########################################
	## ONLY FILL THE FUNCTION BELOW    ##
	###########################################


	
	def getInstrumentIds(self):
		# return ['ADBE', 'AMZN', 'ABT','BIIB',
		# 		'CL', 'DHR', 'GD', 'INTC', 'MCD']
		a=[line.rstrip('\n') for line in open('historicalData\qq16p1Data\stock_list.txt')]
		print(a)
		return a

	def getInstrumentFeatureConfigDicts(self):

		# ADD RELEVANT FEATURES HERE
		expma10dic = {'featureKey': 'expma10',
				 'featureId': 'exponential_moving_average',
				 'params': {'period': 4,
							  'featureName': 'Share Price'}}
		mom10dic = {'featureKey': 'mom10',
				 'featureId': 'difference',
				 'params': {'period': 4,
							  'featureName': 'Share Price'}}
		return [expma10dic,mom10dic]


	'''
	A function that returns your predicted value based on your heuristics.
	'''

	def getRevenuePrediction(self, time, updateNum, instrumentList, instrumentFeatures, MarketFeatures, instrumentFeatureList, targetVariable, predictions):
        # dataframe for a historical instrument feature (mom10 in this case). The index is the timestamps
		# of upto lookback data points. The columns of this dataframe are the stock symbols/instrumentIds.
		# Get the last row of the dataframe, the most recent datapoint
        
        
        
#		mom10 = instrumentFeatures.getFeatureDf('mom10').iloc[-1]
#		
#		expma10 = instrumentFeatures.getFeatureDf('expma10').iloc[-1] 
#		price = instrumentFeatures.getFeatureDf('Share Price').iloc[-1]
##		for f in instrumentFeatureList:
##		 	print(f)
#		
#		
		for s in instrumentList:
			for i in range(len(instrumentFeatureList)):
				stock_model = model(self, time, updateNum, instrumentList, instrumentFeatures, MarketFeatures, instrumentFeatureList, targetVariable, predictions, s, i, 0)
				predictions = stock_model.predict(instrumentFeatures.getFeatureDf(instrumentFeatureList[i]).iloc[-1])

#				predictions[s] += coeff[i] * (instrumentFeatures.getFeatureDf(features[i]).iloc[-1])[s]

		
		predictions.fillna(0,inplace=True)

		return predictions


	'''
	A function that returns your predicted value based on your heuristics.
	'''

	def getIncomePrediction(self, time, updateNum, instrumentList, instrumentFeatures, MarketFeatures, instrumentFeatureList, targetVariable, predictions):

		# dataframe for a historical instrument feature (mom10 in this case). The index is the timestamps
		# of upto lookback data points. The columns of this dataframe are the stock symbols/instrumentIds.
		# Get the last row of the dataframe, the most recent datapoint
		# import pdb; pdb.set_trace()
#		mom10 = instrumentFeatures.getFeatureDf('mom10').iloc[-1]
#		
#		expma10 = instrumentFeatures.getFeatureDf('expma10').iloc[-1] 
#		price = instrumentFeatures.getFeatureDf('Share Price').iloc[-1]
#		# for f in instrumentFeatureList:
#		# 	print(f)
#		
#		## Linear Regression Implementation
#
#		coeff = [ 0.03249183, 0.49675487]
#		for s in instrumentList:
#			predictions[s] = coeff[0] * mom10[s] + coeff[1] * expma10[s]

		for s in instrumentList:
			for i in range(len(instrumentFeatureList)):
				stock_model = model(self, time, updateNum, instrumentList, instrumentFeatures, MarketFeatures, instrumentFeatureList, targetVariable, predictions, s, i, 1)
				predictions = stock_model.predict(instrumentFeatures.getFeatureDf(instrumentFeatureList[i]).iloc[-1])

		
		predictions.fillna(0,inplace=True)

		return predictions

	##############################################
	##  CHANGE ONLY IF YOU HAVE CUSTOM FEATURES  ##
	###############################################

	def getCustomFeatures(self):
		return {'my_custom_feature_identifier': MyCustomFeatureClassName}

####################################################
##   YOU CAN DEFINE ANY CUSTOM FEATURES HERE      ##
##  If YOU DO, MENTION THEM IN THE FUNCTION ABOVE ##
####################################################
class MyCustomFeatureClassName(Feature):
	''''
	Custom Feature to implement for instrument. This function would return the value of the feature you want to implement.
	1. create a new class MyCustomFeatureClassName for the feature and implement your logic in the function computeForInstrument() -
	2. modify function getCustomFeatures() to return a dictionary with Id for this class
		(follow formats like {'my_custom_feature_identifier': MyCustomFeatureClassName}.
		Make sure 'my_custom_feature_identifier' doesnt conflict with any of the pre defined feature Ids
		def getCustomFeatures(self):
			return {'my_custom_feature_identifier': MyCustomFeatureClassName}
	3. create a dict for this feature in getInstrumentFeatureConfigDicts() above. Dict format is:
			customFeatureDict = {'featureKey': 'my_custom_feature_key',
								'featureId': 'my_custom_feature_identifier',
								'params': {'param1': 'value1'}}
	You can now use this feature by calling it's featureKey, 'my_custom_feature_key' in getPrediction()
	'''
	@classmethod
	def computeForInstrument(cls, updateNum, time, featureParams, featureKey, instrumentManager):
		# Custom parameter which can be used as input to computation of this feature
		param1Value = featureParams['param1']

		# A holder for the all the instrument features
		lookbackInstrumentFeatures = instrumentManager.getLookbackInstrumentFeatures()

		# dataframe for a historical instrument feature (basis in this case). The index is the timestamps
		# atmost upto lookback data points. The columns of this dataframe are the symbols/instrumentIds.
		lookbackInstrumentValue = lookbackInstrumentFeatures.getFeatureDf('symbolVWAP')

		# The last row of the previous dataframe gives the last calculated value for that feature (basis in this case)
		# This returns a series with symbols/instrumentIds as the index.
		currentValue = lookbackInstrumentValue.iloc[-1]

		if param1Value == 'value1':
			return currentValue * 0.1
		else:
			return currentValue * 0.5


if __name__ == "__main__":
	if updateCheck():
		print('Your version of the auquan toolbox package is old. Please update by running the following command:')
		print('pip install -U auquan_toolbox')
	else:
		print('Loading your config dicts and prediction function')
		tf = MyTradingFunctions()
		print('Loaded config dicts and prediction function, Loading Problem Params')
		tsParams1 = MyTradingParams(tf, 'Revenue(Y)')
		tradingSystem = TradingSystem(tsParams1)
#		s=MyTradingFunctions()
#		s.getInstrumentIds()
		results1 = tradingSystem.startTrading(onlyAnalyze=False, shouldPlot=False, makeInstrumentCsvs=False)
#		tsParams2 = MyTradingParams(tf, 'Income(Y)')
#		tradingSystem = TradingSystem(tsParams2)
#		results2 = tradingSystem.startTrading(onlyAnalyze=False, shouldPlot=False, makeInstrumentCsvs=False)
	print('Score: %0.3f'%((results1['score'])))#+results2['score'])/2))