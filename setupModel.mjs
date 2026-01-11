import { CONFIG } from "./config.mjs";
import { tf } from "./tf.mjs";

export async function setupModel() {
    const model = tf.sequential();

    model.add(
        tf.layers.dense({
            inputShape: [CONFIG.GAME_STATE_SIZE],
            units: CONFIG.HIDDEN_LAYER_LENGTHS[0],
            activation: 'relu',
        })
    );
    
    for (let i = 1; i < CONFIG.HIDDEN_LAYER_LENGTHS.length; i++) {
        model.add(
            tf.layers.dense({
                units: CONFIG.HIDDEN_LAYER_LENGTHS[i],
                activation: 'relu',
            })
        );
    }

    model.add(
        tf.layers.dense({
            units: CONFIG.ACTION_SIZE,
            activation: 'linear',
        })
    );

    model.compile({
        optimizer: tf.train.adam(CONFIG.LEARNING_RATE),
        loss: CONFIG.LOSS,
    });

    return model;
}