# Time_Transformer_BTC_Prediction

### Enviroment install

```
conda create -n btc_precdiction python=3.8
conda activate btc_precdiction
pip install -r requirements_with_jupyter_lab.txt
conda install -c conda-forge ta-lib==0.4.19
pip install seaborn
pip install openpyxl
pip install ccxt
```
### jupyter lab IDE
```
jupyter lab
```

### Enter your binance api key to binance_api_key_example.json

1. Change the file name to binance_api_key.json 

2. Enter your binance api key to json file
```
{"api_key": "enter_your_api_key", "api_secret": "enter_your_secret_api_key"}
```

### Run Auto_prediction.py to start Next Hour price prediciton 

```
python Auto_prediction.py
```

### 3000 hours Testing Performance

![image](https://github.com/Rainman5042/Time_Transformer_BTC_Prediction/blob/main/testing_performance.png)

### 400 hours Testing Performance
![image](https://github.com/Rainman5042/Time_Transformer_BTC_Prediction/blob/main/400_testing_performance.png)
