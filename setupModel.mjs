import { CONFIG } from "./config.mjs";
import { tf } from "./tf.mjs";

// Actor Critic

export function setupModel() {
    // Actor
    const actor = tf.sequential();
    actor.add(tf.layers.dense({
        inputShape: [CONFIG.GAME_STATE_SIZE],
        units: CONFIG.ACTOR_HIDDEN_LAYER_LENGTHS[0],
        activation: 'relu',
        kernelInitializer: 'heNormal'
    }));
    for (let i = 1; i < CONFIG.ACTOR_HIDDEN_LAYER_LENGTHS.length; i++) {
        actor.add(tf.layers.dense({
            units: CONFIG.ACTOR_HIDDEN_LAYER_LENGTHS[i],
            activation: 'relu',
            kernelInitializer: 'heNormal'
        }));
    }
    actor.add(tf.layers.dense({
        units: CONFIG.ACTION_SIZE,
        activation: 'sigmoid',
        kernelInitializer: 'glorotUniform'
    }));

    // Critic
    const critic = tf.sequential();
    critic.add(tf.layers.dense({
        inputShape: [CONFIG.GAME_STATE_SIZE],
        units: CONFIG.CRITIC_HIDDEN_LAYER_LENGTHS[0],
        activation: 'relu',
        kernelInitializer: 'heNormal'
    }));
    for (let i = 1; i < CONFIG.CRITIC_HIDDEN_LAYER_LENGTHS.length; i++) {
        critic.add(tf.layers.dense({
            units: CONFIG.CRITIC_HIDDEN_LAYER_LENGTHS[i],
            activation: 'relu',
            kernelInitializer: 'heNormal'
        }));
    }
    critic.add(tf.layers.dense({
        units: 1,
        activation: 'linear',
        kernelInitializer: 'glorotUniform'
    }));

    const actorOptimizer = tf.train.adam(CONFIG.ACTOR_LEARNING_RATE);
    const criticOptimizer = tf.train.adam(CONFIG.CRITIC_LEARNING_RATE);

    const model = {
        actor: actor,
        critic: critic,
        optimizer: {
            actor: actorOptimizer,
            critic: criticOptimizer
        }
    };
    return model;
}