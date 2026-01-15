export const CONFIG = {
    DUELING_SHARED_LAYER_LENGTHS: [256, 256, 256],
    GAME_STATE_SIZE: 24,
    ACTION_SIZE: 6,
    LEARNING_RATE: 0.0001,
    SAVE_AFTER_EPISODES: 500,
    BATCH_SIZE: 256,
    DISCOUNT_FACTOR: 0.99,
    REPLAY_BUFFER_SIZE: 500_000,
    TARGET_UPDATE_FREQ: 100,

    // Prioritized Experience Replay parameters
    PER_ALPHA: 0.6,       // Priority exponent (0 = uniform, 1 = full prioritization)
    PER_BETA_START: 0.4,  // Initial importance sampling exponent
    PER_BETA_END: 1.0,    // Final importance sampling exponent
    PER_BETA_FRAMES: 100000, // Number of frames to anneal beta

    EPSILON: 1,
    MIN_EPSILON: 0.05,
    EPSILON_DECAY: 0.000035,

    PLAYER_ONE_ID: 200,
    PLAYER_TWO_ID: 201,
    POSITION_NORMALIZATION: 1 / 500,
    VELOCITY_NORMALIZATION: 1,
};

top.CONFIG = CONFIG;