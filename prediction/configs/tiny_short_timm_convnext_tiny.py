from types import SimpleNamespace

from prediction.configs.baseline import baseline_cfg

model_cfg = baseline_cfg

model_cfg.LIFT = SimpleNamespace(
        # Short BEV dimensions
        X_BOUND = [-15.0, 15.0, 0.15],  # Forward
        Y_BOUND = [-15.0, 15.0, 0.15],  # Sides
        Z_BOUND = [-10.0, 10.0, 20.0],  # Height
        D_BOUND = [2.0, 50.0, 1.0],
)

model_cfg.MODEL = SimpleNamespace(

        STCONV = SimpleNamespace(
            INPUT_EGOPOSE = True,
        ),    
        
        ENCODER = SimpleNamespace(
            DOWNSAMPLE = 8,
            NAME = 'convnext_tiny.fb_in22k_ft_in1k_384',
            OUT_CHANNELS = 64,
            USE_DEPTH_DISTRIBUTION = True,
        ),
                
        # Tiny
        SEGFORMER = SimpleNamespace(
            N_ENCODER_BLOCKS = 5,
            DEPTHS = [2, 2, 2, 2, 2],
            SEQUENCE_REDUCTION_RATIOS = [8, 8, 4, 2, 1],
            HIDDEN_SIZES = [16, 24, 32, 48, 64], 
            PATCH_SIZES = [7, 3, 3, 3, 3],
            STRIDES = [2, 2, 2, 2, 2],
            NUM_ATTENTION_HEADS = [1, 2, 4, 8, 8],
            MLP_RATIOS = [4, 4, 4, 4, 4],
            HEAD_DIM_MULTIPLIER = 4,
            HEAD_KERNEL = 2,
            HEAD_STRIDE = 2,
        ),

        TEMPORAL_MODEL = SimpleNamespace(
            NAME = 'temporal_block',
            START_OUT_CHANNELS = 64,
            EXTRA_IN_CHANNELS = 0,
            INBETWEEN_LAYERS = 0,
            PYRAMID_POOLING = True,
            INPUT_EGOPOSE = True,
        ),
        DISTRIBUTION = SimpleNamespace(
            LATENT_DIM = 32,
            MIN_LOG_SIGMA = -5.0,
            MAX_LOG_SIGMA = 5.0,
        ),
        FUTURE_PRED = SimpleNamespace(
            N_GRU_BLOCKS = 3,
            N_RES_LAYERS = 3,
        ),

        BN_MOMENTUM = 0.1,
)