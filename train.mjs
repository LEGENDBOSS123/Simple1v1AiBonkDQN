import { CONFIG } from "./config.mjs";
import { tf } from "./tf.mjs";

export async function train(model, memory) {
    const batch = sampleBatch(memory, CONFIG.BATCH_SIZE);
    const lossValue = tf.tidy(() => {
        const states = tf.tensor2d(batch.map(m => m.state));
        const nextStates = tf.tensor2d(batch.map(m => m.nextState));
        const rewards = tf.tensor1d(batch.map(m => m.reward));
        const dones = tf.tensor1d(batch.map(m => m.done ? 1 : 0));

        const targetQs = model.predict(states);
        const nextQs = model.predict(nextStates);
        const maxNextQs = nextQs.max(1);

        /*
        if (done == false) {
            reward = reward + discount_factor * max_next_q
        }
        */
        const updatedQs = rewards.add(maxNextQs.mul(CONFIG.DISCOUNT_FACTOR).mul(tf.scalar(1).sub(dones)));

        const targetQsArr = targetQs.arraySync();
        const updatedQsArr = updatedQs.arraySync();
        for (let i = 0; i < batch.length; i++) {
            const memory = batch[i];
            const targetValue = updatedQsArr[i];

            for (let j = 0; j < CONFIG.ACTION_SIZE; j++) {
                if (memory.action[j] === 1) {
                    targetQsArr[i][j] = targetValue;
                }
            }
        }

        const newTargetQs = tf.tensor2d(targetQsArr);
        return model.trainOnBatch(states, newTargetQs);
    });
}

function sampleBatch(array, size) {
    const batch = [];
    for (let i = 0; i < size; i++) {
        const index = Math.floor(Math.random() * array.length);
        batch.push(array[index]);
    }
    return batch;
}