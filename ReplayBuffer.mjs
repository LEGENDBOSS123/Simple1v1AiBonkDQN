export class ReplayBuffer {
    constructor(maxSize) {
        this.buffer = [];
        this.maxSize = maxSize;
        this.priorities = [];
        this.position = 0;
    }

    add(state, action, reward, nextState, done) {
        const memory = {
            state,
            action,
            reward,
            nextState,
            done
        }
        let maxPriority = 0;
        for (let i = 0; i < this.priorities.length; i++) {
            if (this.priorities[i] > maxPriority) {
                maxPriority = this.priorities[i];
            }
        }
        if (maxPriority === 0) {
            maxPriority = 1.0;
        }
        if (this.buffer.length < this.maxSize) {
            this.buffer.push(memory);
            this.priorities.push(maxPriority);
        }
        else {
            this.buffer[this.position] = memory;
            this.priorities[this.position] = maxPriority;
            this.position = (this.position + 1) % this.maxSize;
        }
    }

    sample(batchSize, alpha = 0.6, beta = 0.4) {
        if (this.buffer.length === 0) {
            return { batch: [], indices: [], importanceWeights: [] };
        }

        const probs = this.priorities.map(p => Math.pow(p, alpha));
        const sumProbs = probs.reduce((a, b) => a + b, 0);

        const indices = [];
        const batch = [];
        const rawWeights = [];

        for (let i = 0; i < batchSize; i++) {
            const targetSum = Math.random() * sumProbs;
            let currentSum = 0;
            let index = 0;
            for (let j = 0; j < probs.length; j++) {
                currentSum += probs[j];
                if (currentSum >= targetSum) {
                    index = j;
                    break;
                }
            }
            indices.push(index);
            batch.push(this.buffer[index]);

            // Importance sampling weight: (1 / (N * P(i)))^beta
            const prob = probs[index] / sumProbs;
            const weight = Math.pow(1 / (this.buffer.length * prob), beta);
            rawWeights.push(weight);
        }

        // Normalize weights by max weight for stability
        const maxWeight = Math.max(...rawWeights);
        const importanceWeights = rawWeights.map(w => w / maxWeight);

        return { batch, indices, importanceWeights };
    }

    updatePriorities(indices, tdErrors) {
        for (let i = 0; i < indices.length; i++) {
            const idx = indices[i];
            this.priorities[idx] = Math.abs(tdErrors[i]) + 1e-5;
        }
    }

    size() {
        return this.buffer.length;
    }

    toJSON() {
        return {
            buffer: this.buffer,
            priorities: this.priorities,
            maxSize: this.maxSize,
            position: this.position
        };
    }

    static fromJSON(json) {
        const replayBuffer = new ReplayBuffer(json.maxSize);
        replayBuffer.buffer = json.buffer;
        replayBuffer.priorities = json.priorities;
        replayBuffer.position = json.position ?? 0;
        return replayBuffer;
    }
}