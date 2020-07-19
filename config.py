BATCH_SIZE = 512
BUFFER_SIZE = 10000
EPOCHS = 250
STEP = 1
HISTORY_SIZE = 4
TARGET_DIS = 1

# https://github.com/bukosabino/ta
FEATURES = ['Close',
            'Open',
            'Volume',
            'momentum_rsi',
            'volume_adi',
            'trend_adx_pos',
            'trend_adx_neg',
            'volume_sma_em',
            'volume_vpt',
            'volume_nvi',
            'volume_vwap',
            'momentum_stoch',
            'momentum_mfi',
            'momentum_kama',
            'momentum_ao',
            ]
TEST_MODEL = 'checkpoints/2020-07-18_22:04_PM_002'

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
