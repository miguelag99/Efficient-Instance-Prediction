from types import SimpleNamespace

baseline_cfg = SimpleNamespace(
    LOG_DIR = 'training_results',
    TAG = 'convnext_tiny',
    WANDB_PROJECT = 'FAEFormer',
    WANDB_ID = '',

    ACCELERATOR = "cuda",
    GPUS = 0,  # which gpus to use
    DEVICES = "auto", # how many gpus to use, auto for all available
    PRECISION = '16-mixed',  # 16-mixed or 32bit
    BATCHSIZE = 2,
    EPOCHS = 30,
    
    OPTIMIZER = SimpleNamespace(
        TYPE = 'AdamW',
        LR = 6e-5,
        WEIGHT_DECAY = 2e-3,
    ),
    
    SCHEDULER = SimpleNamespace(
        TYPE = 'PolynomialLR',
    ),

    GRAD_NORM_CLIP = 0.5,

    N_WORKERS = 2,
    LOGGING_INTERVAL = 5,
    
    
    # Pretrainied or resume training configuration
    # If LOAD_WEIGHTS is True, then the model weights will be loaded from the path 
    # specified in CKPT 
    # If both LOAD_WEIGHTS and RESUME_TRAINING are True, then the model weights will
    # be loaded and 
    # training will resume mantaining the optimizer and shceduler states.
    
    PRETRAINED = SimpleNamespace(
        LOAD_WEIGHTS = False,
        RESUME_TRAINING = False,
        PATH = '/home/perception/workspace/',
        CKPT = 'checkpoints/_ckpt',
    ),
    
    DATASET = SimpleNamespace(
        DATAROOT = '/home/perception/Datasets/nuscenes/',
        VERSION = 'v1.0-trainval',
        NAME = 'nuscenes',
        IGNORE_INDEX = 255,  # Ignore index when creating flow/offset labels
        FILTER_INVISIBLE_VEHICLES = True,  # Filter vehicles not visible from cameras
        N_CAMERAS = 6,  # Number of cameras
    ),
    
    # how many frames of temporal context (1 for single timeframe)
    TIME_RECEPTIVE_FIELD = 3, 
    # how many time steps into the future to predict
    N_FUTURE_FRAMES = 4,  

    IMAGE = SimpleNamespace(
        FINAL_DIM = (224, 480),
        ORIGINAL_DIM = (900, 1600),
        RESIZE_SCALE = 0.3,
        TOP_CROP = 46,
        ORIGINAL_HEIGHT = 900 ,  # Original input RGB camera height
        ORIGINAL_WIDTH = 1600 ,  # Original input RGB camera width
        NAMES = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
    ),


    SEMANTIC_SEG = SimpleNamespace(
        # per class cross entropy weights (bg, dynamic, drivable, lane)
        WEIGHTS = [1.0, 2.0],
        USE_TOP_K = True,  # backprop only top-k hardest pixels
        TOP_K_RATIO = 0.25,
    ),

    INSTANCE_SEG = SimpleNamespace(),

    INSTANCE_FLOW = SimpleNamespace(
        ENABLED = True,
    ),

    PROBABILISTIC = SimpleNamespace(
        ENABLED = False,  # learn a distribution over futures
        WEIGHT = 100.0,
        # number of dimension added (future flow, future centerness, offset, seg)
        FUTURE_DIM = 6,
    ),

    FUTURE_DISCOUNT = 0.95,

    VISUALIZATION = SimpleNamespace(
        OUTPUT_PATH = './visualization_outputs',
        SAMPLE_NUMBER = 1000,
        VIS_GT = True,
    )
)
