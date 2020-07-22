BATCH_SIZE = 128
BUFFER_SIZE = 10000
EPOCHS = 500
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

TEST_MODEL = 'checkpoints/2020-07-21_19:10_PM_141'

STD_DENOMINATOR = 30.0

N_CLASSES = 3
BINARY_BULL_LABELS = ["UP", "DOWN"]
CLASS_TO_LABEL = {0: "UP", 1: "NOTHING", 3: "DOWN"}

LABEL_UP = [1, 0, 0]
NONE = [0, 1, 0]
LABEL_DOWN = [0, 0, 1]

TEST_TICKERS = [
    'AAL',
    'AAPL',
    'BA',
    'BABA',
    'CAJ',
    'DDOG',
    'F',
    'FB',
    'MSFT',
    'NFLX',
    'NVDA',
    'OKTA',
    'ROKU',
    'SPY',
    'TSLA',
    'TWTR',
]

TICKERS = [
    'A',
    'AA',
    'AACG',
    'AAL',
    'AAMC',
    'AAME',
    'AAN',
    'AAOI',
    'AAON',
    'AAP',
    'AAPL',
    'AAT',
    'ACB',
    'ADBE',
    'AMZN',
    'AUY',
    'BA',
    'BABA',
    'CAJ',
    'DD',
    'DDOG',
    'EBAY',
    'F',
    'FB',
    'FIT',
    'GE',
    'GPRO',
    'GRUB',
    'IBM',
    'INTC',
    'JPM',
    'KHC',
    'KO',
    'LEJU',
    'M',
    'MMM',
    'MRO',
    'MSFT',
    'NFLX',
    'NVDA',
    'OKTA',
    'OMI',
    'ORCL',
    'QQQ',
    'ROKU',
    'SIX',
    'SNAP',
    'SPY',
    'TWTR',
    'UAL',
    'WBA',
    'WFC',
    'WMT',
    'XOM'
]
