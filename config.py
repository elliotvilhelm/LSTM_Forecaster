BATCH_SIZE = 1024
BUFFER_SIZE = 10000
EPOCHS = 200
STEP = 1
HISTORY_SIZE = 8
TARGET_DIS = 1
FEATURES = ['Close', 'Open','Volume', 'momentum_rsi', 'volume_adi']
#EST_MODEL = 'checkpoints/2020-07-17_22:37_PM'
TEST_MODEL = 'checkpoints/2020-07-18_16:43_PM'
N_CLASSES = 3

LABEL_UP = [1, 0, 0]
NONE = [0, 1, 0]
LABEL_DOWN = [0, 0, 1]
TEST_TICKERS = ["SPY",
           "QQQ",
           "ROKU",
           "MSFT",
           "BA",
           "AMZN",
           "DDOG",
           "OKTA",
           "BABA",
           "FB",
           "INTC",
           "NVDA", 
           "ADBE",
           "FIT", 
           "IBM", 
           "GRUB", 
           "CAJ", 
           "SIX", 
           "AUY", 
           "F", 
           "GE", 
           "XOM",
           "MRO", 
           "MMM"
           ]
TICKERS = ["SPY",
           "QQQ",
           "ROKU",
           "MSFT",
           "BA",
           "AMZN",
           "DDOG",
           "OKTA",
           "BABA",
           "FB",
           "INTC",
           "NVDA", 
           "ADBE",
           "FIT", 
           "IBM", 
           "GRUB", 
           "CAJ", 
           "SIX", 
           "AUY", 
           "F", 
           "GE", 
           "XOM",
           "MRO", 
           "MMM",
           "GPRO",
           "KHC",
           "WBA", 
           "LEJU", 
           "ACB", 
           "DD"
           ]
