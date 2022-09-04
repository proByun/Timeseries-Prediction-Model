import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

# from sklearnex import patch_sklearn
# patch_sklearn()


# Ridge Regression
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV

# Randomforest Regression
from sklearn.ensemble import RandomForestRegressor

# Support Vector Regression
from sklearn.svm import SVR

# hyper parameter tunning
from sklearn.model_selection import GridSearchCV

# ARIMA
from statsmodels.tsa.arima.model import ARIMA
from pmdarima.arima import auto_arima

# LSTM
# from sklearn.preprocessing import MinMaxScaler
# from keras.models import Sequential
# from keras.layers import Dense, Dropout
# from keras.layers import LSTM

from sklearn.model_selection import train_test_split

"""
Created on Sun Jul 31 20:18:12 2022

@author: Junhyun
"""

class Recursive_Bayesian_Ensemble_Model(TimeSeriesRegression):

    def __init__(self):
        super().__init__()

    def EnsembleWeight(self, y_test, ensembleModel):
        # test loss of each prediction model
        ridge_mse = mean_squared_error(y_test, ensembleModel['ridgeTestPrediction'])
        rf_mse = mean_squared_error(y_test, ensembleModel['rfTestPrediction'])
        svr_mse = mean_squared_error(y_test, ensembleModel['svrTestPrediction'])
        arima_mse = mean_squared_error(y_test, ensembleModel['arimaTestPrediction'])

        # Fitness weight of each model
        fitnessSum = (1 / ridge_mse + 1 / rf_mse + 1 / svr_mse + 1 / arima_mse)  # MSE -> fitness Weight
        ridge_fitWeight = (1 / ridge_mse) / fitnessSum
        rf_fitWeight = (1 / rf_mse) / fitnessSum
        svr_fitWeight = (1 / svr_mse) / fitnessSum
        arima_fitWeight = (1 / arima_mse) / fitnessSum

        fitWeight = {'Ridge_FitWeight': ridge_fitWeight, 'RF_FitWeight': rf_fitWeight, 'SVR_FitWeight': svr_fitWeight,
                     'ARIMA_FitWeight': arima_fitWeight}

        return (fitWeight)

    def EnsemblePrediction(self, X_train, X_test, y_train, y_test, fitWeight):
        ridge = super().Ridge_regression(X_train, X_test, y_train, y_test)
        rf = super().Randomforest_regression(X_train, X_test, y_train, y_test)
        svr = super().Supportvector_regression(X_train, X_test, y_train, y_test)
        arima = super().ARIMA_model(y_train, y_test.shape[0])

        RBEM_train_pred = (ridge['trainPrediction'] * fitWeight['Ridge_FitWeight']) + (
                    rf['trainPrediction'] * fitWeight['RF_FitWeight']) + (
                                      svr['trainPrediction'] * fitWeight['SVR_FitWeight'])
        RBEM_test_pred = (ridge['testPrediction'] * fitWeight['Ridge_FitWeight']) + (
                    rf['testPrediction'] * fitWeight['RF_FitWeight']) + (
                                     svr['testPrediction'] * fitWeight['SVR_FitWeight']) + arima['testPrediction'] * \
                         fitWeight['ARIMA_FitWeight']

        return (
            {
                'trainPrediction': RBEM_train_pred, 'testPrediction': RBEM_test_pred,
                'ridgeTrainPrediction': ridge['trainPrediction'], 'ridgeTestPrediction': ridge['testPrediction'],
                'rfTrainPrediction': rf['trainPrediction'], 'rfTestPrediction': rf['testPrediction'],
                'svrTrainPrediction': svr['trainPrediction'], 'svrTestPrediction': svr['testPrediction'],
                'arimaTestPrediction': arima['testPrediction']
            }
        )

    # Recursive Bayesian Update
    def BayesianUpdate(self, Prior, Likelihood):
        posterior_Sum = Prior['Ridge_FitWeight'] * Likelihood['Ridge_FitWeight'] + Prior['RF_FitWeight'] * Likelihood[
            'RF_FitWeight'] + Prior['SVR_FitWeight'] * Likelihood['SVR_FitWeight'] + Prior['ARIMA_FitWeight'] * \
                        Likelihood['ARIMA_FitWeight']
        Ridge_Updated_Weight = Prior['Ridge_FitWeight'] * Likelihood['Ridge_FitWeight'] / posterior_Sum
        RF_Updated_Weight = Prior['RF_FitWeight'] * Likelihood['RF_FitWeight'] / posterior_Sum
        SVR_Updated_Weight = Prior['SVR_FitWeight'] * Likelihood['SVR_FitWeight'] / posterior_Sum
        ARIMA_Updated_Weight = Prior['ARIMA_FitWeight'] * Likelihood['ARIMA_FitWeight'] / posterior_Sum

        updatedFitWeight = {'Ridge_FitWeight': Ridge_Updated_Weight, 'RF_FitWeight': RF_Updated_Weight,
                            'SVR_FitWeight': SVR_Updated_Weight, 'ARIMA_FitWeight': ARIMA_Updated_Weight}

        return (updatedFitWeight)

    # Recursive Bayesian Ensemble Model
    def RBEM_model(self, X, y, trainCycle=10, predictionCycle=5, Cycle=5):
        """

        Parameters
        ----------
        X : Array
            Input data, shape=(nrow, lag, ncol)
        y : Array
            Output data, shape=(nrow,)
        trainCycle : int
            며칠 주기로 학습할 것인지
        predictionCycle : int
            며칠 주기로 예측할 것인지
        Cycle : int
            위 과정을 몇번 반복할 것인지

        Returns
        -------
        recursive_test_pred : Array
            test 데이터를 recursive하게 예측한 결과


        예시)
        1row = 1일일때,
        trainCycle = 5 -> 5일 주기로 학습
        predictionCycle = 2 -> 2일 주기로 예측

        1월 1일 데이터가 있다고 가정

        - 1월 1일 ~ 1월 5일 (5일) 학습 후, 1월 6일~1월 7일 (2일) 예측 (1cycle)
        - 1월 3일 ~ 1월 7일 (5일) 학습 후, 1월 8일~1월 9일 (2일) 예측 (2cycle) (1월 6일은 실제 데이터임 (예측한 데이터 X)) (현재시점까지 왔다고 가정)
        - 1월 5일 ~ 1월 9일 (5일) 학습 후, 1월 10일~1월 11일 (2일) (1일) 예측 (3cycle) (1월 7일은 실제 데이터임 (예측한 데이터 X)) (현재시점까지 왔다고 가정)


        """

        # Recursive Prediction
        RBEM_test_pred = np.array([])
        Ridge_test_pred = np.array([])
        RF_test_pred = np.array([])
        SVR_test_pred = np.array([])
        ARIMA_test_pred = np.array([])
        y_test = np.array([])

        # Prior of Ensemble Weight
        prior_fitWeight = {'Ridge_FitWeight': 1 / 3, 'RF_FitWeight': 1 / 3, 'SVR_FitWeight': 1 / 3,
                           'ARIMA_FitWeight': 1 / 3}

        for i in range(Cycle):
            # Recursive prediction
            recursive_X_train = X[(predictionCycle * i):(trainCycle + predictionCycle * i)]
            recursive_X_test = X[(trainCycle + predictionCycle * i):(trainCycle + predictionCycle * (i + 1))]

            recursive_y_train = y[(predictionCycle * i):(trainCycle + predictionCycle * i), ]
            recursive_y_test = y[(trainCycle + predictionCycle * i):(trainCycle + predictionCycle * (i + 1)), ]

            y_test = np.append(y_test, recursive_y_test.values)

            ensemble_pred = self.EnsemblePrediction(recursive_X_train, recursive_X_test, recursive_y_train,
                                                    recursive_y_test, prior_fitWeight)

            # Bayesian Ensemble Prediction
            RBEM_test_pred = np.append(RBEM_test_pred, ensemble_pred['testPrediction'])

            # Comprised Model of Ensemble Model
            Ridge_test_pred = np.append(Ridge_test_pred, ensemble_pred['ridgeTestPrediction'])
            RF_test_pred = np.append(RF_test_pred, ensemble_pred['rfTestPrediction'])
            SVR_test_pred = np.append(SVR_test_pred, ensemble_pred['svrTestPrediction'])
            ARIMA_test_pred = np.append(ARIMA_test_pred, ensemble_pred['arimaTestPrediction'])

            # likelihood of Ensemble Weight
            likelihood_fitWeight = self.EnsembleWeight(recursive_y_test.values, ensemble_pred)

            # Bayesian Update Ensemble Weights
            prior_fitWeight = self.BayesianUpdate(prior_fitWeight, likelihood_fitWeight)

        return (
            {
                'RBEM_test_pred': RBEM_test_pred,
                'Ridge_test_pred': Ridge_test_pred,
                'RF_test_pred': RF_test_pred,
                'SVR_test_pred': SVR_test_pred,
                'ARIMA_test_pred': ARIMA_test_pred,
                'y_test': y_test
            }

        )

    def maxCycleNum(self, data, trainCycle=10, predictionCycle=5):
        """

        Parameters
        ----------
        data : DataFrame
            data
        trainCycle : int
            며칠 주기로 학습할 것인지
        predictionCycle : int
            며칠 주기로 예측할 것인지

        Returns
        -------
        maxCycle : int
            Recursive 하게 예측할 수 있는 최대 cycle 수


        """

        maxCycle = int((data.shape[0] - trainCycle) / predictionCycle) - 1

        print('Max Cycle Number : %d' % maxCycle)

        return (maxCycle)
