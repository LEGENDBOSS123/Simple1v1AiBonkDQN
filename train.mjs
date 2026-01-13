import { CONFIG } from "./config.mjs";
import { log } from "./log.mjs";
import { tf } from "./tf.mjs";

export async function train(model, memory) {
    if (memory.length < CONFIG.BATCH_SIZE) {
        return null;
    }
    const batch = memory.slice(-CONFIG.BATCH_SIZE);
    const actor = model.actor;
    const critic = model.critic;
    const actorOptimizer = model.optimizer.actor;
    const criticOptimizer = model.optimizer.critic;

    let loss = tf.tidy(() => {
        const states = tf.tensor2d(batch.map(m => m.state));
        const nextStates = tf.tensor2d(batch.map(m => m.nextState));
        const actions = tf.tensor2d(batch.map(m => m.action));
        const rewards = tf.tensor1d(batch.map(m => m.reward));
        const dones = tf.tensor1d(batch.map(m => m.done ? 1 : 0));

        const values = critic.predict(states).squeeze();
        const nextValues = critic.predict(nextStates).squeeze();
        const targets = rewards.add(nextValues.mul(CONFIG.DISCOUNT_FACTOR).mul(tf.scalar(1).sub(dones)));
        const advantages = targets.sub(values);

        const criticLoss = criticOptimizer.minimize(() => {
            const predictedValues = critic.predict(states).squeeze();
            return tf.losses.huberLoss(targets, predictedValues);
        }, true);

        const actorLoss = actorOptimizer.minimize(() => {
            const actionProbs = actor.predict(states);

            const epsilon = tf.scalar(1e-8);
            const logProbs = actions.mul(actionProbs.add(epsilon).log()).add(tf.scalar(1).sub(actions).mul(tf.scalar(1).sub(actionProbs).add(epsilon).log()));

            const policyLoss = logProbs.sum(1).mul(advantages).mul(-1).mean();
            const entropy = actionProbs.mul(actionProbs.add(epsilon).log())
                .add(tf.scalar(1).sub(actionProbs).mul(tf.scalar(1).sub(actionProbs).add(epsilon).log()))
                .mul(-1).sum(1).mean()

            return policyLoss.sub(entropy.mul(CONFIG.ENTROPY_COEFFICIENT));
        }, true);

        return {
            actorLoss: actorLoss.dataSync()[0],
            criticLoss: criticLoss.dataSync()[0]
        }

    });
    return loss;
}