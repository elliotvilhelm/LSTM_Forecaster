BATCH_SIZE = 128
BUFFER_SIZE = 10000
EPOCHS = 50
STEP = 1
HISTORY_SIZE = 24
TARGET_DIS = 4

# https://github.com/bukosabino/ta
FEATURES = [
    'Close',
    'Open',
    'Volume',

    'trend_adx_pos',
    'trend_adx_neg',
    'trend_trix',

    'volume_adi',
    'volume_cmf',
    'volume_nvi',
    'volume_sma_em',
    'volume_vpt',
    'volume_vwap',

    'momentum_ao',
    'momentum_kama',
    'momentum_rsi',
    'momentum_stoch',
    'momentum_mfi',

    'volatility_atr',
]

TEST_MODEL = 'checkpoints/2020-07-18_22:29_PM_106'

STD_DENOMINATOR = 50.0

N_CLASSES = 3
BINARY_BULL_LABELS = ["UP", "DOWN"]
CLASS_TO_LABEL = {0: "UP", 1: "NOTHING", 3: "DOWN"}

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
                "ORCL",
                "SIX",
                "AUY",
                "F",
                "GE",
                "XOM",
                "MRO",
                "MMM",
                "WMT"
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
           "ORCL",
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
           "DD",
           "WMT"
           ]
