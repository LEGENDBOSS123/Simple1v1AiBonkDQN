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

    const { values, nextValues } = tf.tidy(() => {
        const states = tf.tensor2d(batch.map(m => m.state));
        const nextStates = tf.tensor2d(batch.map(m => m.nextState));
        const values = critic.predict(states).squeeze().arraySync();
        const nextValues = critic.predict(nextStates).squeeze().arraySync();
        return { values, nextValues };
    });

    const rewards = batch.map(m => m.reward);
    const dones = batch.map(m => m.done ? 1 : 0);
    const advantagesArr = new Array(batch.length);
    const targetsArr = new Array(batch.length);
    let gae = 0;

    for (let t = batch.length - 1; t >= 0; t--) {
        const delta = rewards[t] + (CONFIG.DISCOUNT_FACTOR * nextValues[t] * (1 - dones[t])) - values[t];

        gae = delta + (CONFIG.DISCOUNT_FACTOR * CONFIG.GAE_LAMBDA * (1 - dones[t]) * gae);

        advantagesArr[t] = gae;
        targetsArr[t] = gae + values[t];
    }
    const advMean = advantagesArr.reduce((a, b) => a + b, 0) / advantagesArr.length;
    const advStd = Math.sqrt(advantagesArr.map(x => Math.pow(x - advMean, 2)).reduce((a, b) => a + b, 0) / advantagesArr.length + 1e-8);
    const normAdvArr = advantagesArr.map(x => (x - advMean) / (advStd + 1e-8));

    let loss = tf.tidy(() => {
        const states = tf.tensor2d(batch.map(m => m.state));
        const actions = tf.tensor2d(batch.map(m => m.action));

        const targets = tf.tensor1d(targetsArr);
        const advantages = tf.tensor1d(normAdvArr);


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