import pandas as pd
import numpy as np
from prophet import Prophet
class Recursive_Prophet():
     
    # Recursive Prophet Ensemble Model
    def Prophet_model(self, df, trainCycle=10, predictionCycle=5, Cycle=5, freq='H'):
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
        df['ds'] = pd.to_datetime(df['ds'])
        
        # Recursive Prediction
        prophet_test_pred = np.array([])
        y_test = np.array([])
        
        for i in range(Cycle):

            # Recursive prediction
            recursive_X_train = df.iloc[(predictionCycle*i):(trainCycle+predictionCycle*i),:]
            recursive_y_test = df.iloc[(trainCycle+predictionCycle*i):(trainCycle+predictionCycle*(i+1)),1]
            
            model = Prophet()
            model.fit(recursive_X_train)
            future = model.make_future_dataframe(periods=predictionCycle, freq=freq)
            forecast = model.predict(future)
            pred = forecast['yhat'][len(recursive_X_train):].values
            
            y_test = np.append(y_test, recursive_y_test.values)
                                        
            # Bayesian Ensemble Prediction
            prophet_test_pred = np.append(prophet_test_pred, pred)
            

        return(
            { 
                'prophet_test_pred' : prophet_test_pred,
                'y_test' : y_test
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
