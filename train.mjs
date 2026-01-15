import { CONFIG } from "./config.mjs";
import { tf } from "./tf.mjs";

/**
 * Computes Q-values from dueling network outputs: Q = V + (A - mean(A))
 */
function computeQ(V, A) {
    const advMean = A.mean(1, true);
    return V.add(A.sub(advMean));
}

/**
 * Selects Q-values for taken actions using one-hot masking
 */
function selectQForActions(Q, actionIndices) {
    const oneHot = tf.oneHot(actionIndices, 2);
    return Q.mul(oneHot).sum(1);
}

/**
 * Train the model using Double Dueling DQN with PER
 */
export async function train(models, batch, importanceWeights = null) {
    if (batch.length < CONFIG.BATCH_SIZE) return null;

    const { model, target: targetModel, optimizer } = models;
    const numActions = CONFIG.ACTION_SIZE;
    const gamma = CONFIG.DISCOUNT_FACTOR;

    return tf.tidy(() => {
        // Prepare batch tensors
        const states = tf.tensor2d(batch.map(b => b.state));
        const nextStates = tf.tensor2d(batch.map(b => b.nextState));
        const actions = tf.tensor2d(batch.map(b => b.action));
        const rewards = tf.tensor1d(batch.map(b => b.reward));
        const dones = tf.tensor1d(batch.map(b => b.done ? 0 : 1));
        const weights = importanceWeights
            ? tf.tensor1d(importanceWeights)
            : tf.ones([batch.length]);

        // Get network outputs for next states (Double DQN)
        const [Vonline, ...Aonline] = model.predict(nextStates);
        const [Vtarget, ...Atarget] = targetModel.predict(nextStates);

        // Compute target Q-values for each action branch
        const targets = [];
        for (let i = 0; i < numActions; i++) {
            const onlineQ = computeQ(Vonline, Aonline[i]);
            const bestAction = onlineQ.argMax(1);

            const targetQ = computeQ(Vtarget, Atarget[i]);
            const selectedQ = selectQForActions(targetQ, bestAction);

            // Bellman target: r + γ * Q_target * (1 - done)
            targets.push(rewards.add(selectedQ.mul(dones).mul(gamma)));
        }

        // Get current Q-values for TD error calculation
        const [Vcurr, ...Acurr] = model.predict(states);
        let tdErrorMax = tf.zeros([batch.length]);

        for (let i = 0; i < numActions; i++) {
            const actionIdx = actions.slice([0, i], [-1, 1]).squeeze([1]).toInt();
            const currentQ = selectQForActions(computeQ(Vcurr, Acurr[i]), actionIdx);
            const branchTdError = targets[i].sub(currentQ).abs();
            tdErrorMax = tf.maximum(tdErrorMax, branchTdError);
        }
        const tdErrors = tdErrorMax;

        const loss = optimizer.minimize(() => {
            const [V, ...As] = model.apply(states, { training: true });
            let totalLoss = tf.zeros([batch.length]);

            for (let i = 0; i < numActions; i++) {
                const actionIdx = actions.slice([0, i], [-1, 1]).squeeze([1]).toInt();
                const predictedQ = selectQForActions(computeQ(V, As[i]), actionIdx);
                const branchLoss = tf.losses.huberLoss(targets[i], predictedQ, undefined, undefined, tf.Reduction.NONE);
                totalLoss = totalLoss.add(branchLoss);
            }
            // Average across branches, then apply importance weights, then take mean
            return totalLoss.div(numActions).mul(weights).mean();
        }, true);

        return {
            loss: loss.dataSync()[0],
            tdErrors: Array.from(tdErrors.dataSync())
        };
    });
}