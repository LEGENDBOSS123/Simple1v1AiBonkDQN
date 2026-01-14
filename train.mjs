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
    // let gae = 0;
    // for (let t = batch.length - 1; t >= 0; t--) {
    //     const nextDone = (t === batch.length - 1) ? 0 : dones[t + 1];
    //     if (nextDone) {
    //         gae = 0;
    //     }
    //     const delta = rewards[t] + (CONFIG.DISCOUNT_FACTOR * nextValues[t] * (1 - dones[t])) - values[t];
    //     gae = delta + (CONFIG.DISCOUNT_FACTOR * CONFIG.GAE_LAMBDA * (1 - dones[t]) * gae);

    //     advantagesArr[t] = gae;
    //     targetsArr[t] = gae + values[t];
    // }
    let G = 0;
    for (let t = batch.length - 1; t >= 0; t--) {
        // If this is terminal OR if previous step (t+1) was terminal, reset G
        if (dones[t]) {
            G = rewards[t];
        } else {
            G = rewards[t] + CONFIG.DISCOUNT_FACTOR * G;
        }
        targetsArr[t] = G;
        advantagesArr[t] = G;  // No baseline, pure return as advantage
    }
    // const advMean = advantagesArr.reduce((a, b) => a + b, 0) / advantagesArr.length;
    // const advStd = Math.sqrt(advantagesArr.map(x => Math.pow(x - advMean, 2)).reduce((a, b) => a + b, 0) / advantagesArr.length + 1e-8);
    // const normAdvArr = advantagesArr.map(x => (x - advMean) / (advStd + 1e-8));
    const normAdvArr = advantagesArr;  // No normalization

    console.log("Advantages (raw):", advantagesArr.slice(0, 10));
    console.log("Advantages (norm):", normAdvArr.slice(0, 10));
    console.log("Max/Min raw:", Math.max(...advantagesArr), Math.min(...advantagesArr));

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
                .mul(-1).sum(1).mean();

            return policyLoss.sub(entropy.mul(CONFIG.ENTROPY_COEFFICIENT));
        }, true);

        return {
            actorLoss: actorLoss.dataSync()[0],
            criticLoss: criticLoss.dataSync()[0]
        }

    });
    console.log("Value predictions sample:", values.slice(0, 5));
    console.log("Target values sample:", targetsArr.slice(0, 5));
    console.log("Value range:", Math.max(...values), Math.min(...values));
    console.log("=== DIAGNOSTIC ===");
    console.log("Value range:", Math.min(...values).toFixed(4), "to", Math.max(...values).toFixed(4));
    console.log("Advantage range:", Math.min(...advantagesArr).toFixed(4), "to", Math.max(...advantagesArr).toFixed(4));
    console.log("Reward range:", Math.min(...rewards).toFixed(4), "to", Math.max(...rewards).toFixed(4));

    // Show a winning and losing terminal if they exist
    for (let i = 0; i < batch.length; i++) {
        if (dones[i]) {
            console.log(`Terminal[${i}]: reward=${rewards[i].toFixed(2)}, value=${values[i].toFixed(4)}, adv=${advantagesArr[i].toFixed(4)}`);
        }
    }

    // Check unique actions
    const actionStrings = batch.map(m => m.action.join(''));
    console.log("Unique actions:", [...new Set(actionStrings)].length, "/", batch.length);

    return loss;
}