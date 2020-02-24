class Constant:
    DEFAULT_SAVE_PATH = '/tmp/autokeras/'

    # Data

    VALIDATION_SET_SIZE = 0.08333

    # Searcher

    MAX_MODEL_NUM = 1000
    BETA = 2.576
    #BETA = 0
    KERNEL_LAMBDA = 1.0
    T_MIN = 0.0001
    N_NEIGHBOURS = 8
    # T_MIN = 0.8
    # N_NEIGHBOURS = 1

    # Model Defaults

    DENSE_DROPOUT_RATE = 0.5
    CONV_DROPOUT_RATE = 0.25
    CONV_BLOCK_DISTANCE = 2
    DENSE_BLOCK_DISTANCE = 1
    MODEL_LEN = 3
    MODEL_WIDTH = 64

    # ModelTrainer

    DATA_AUGMENTATION = False
    MAX_ITER_NUM = 10
    MIN_LOSS_DEC = 1e-4
    MAX_NO_IMPROVEMENT_NUM = 3
    MAX_BATCH_SIZE = 256
    LIMIT_MEMORY = False
    SEARCH_MAX_ITER = 10