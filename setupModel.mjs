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


export function cloneModel(model) {
    const newModel = setupModel();
    newModel.actor.setWeights(model.actor.getWeights());
    newModel.critic.setWeights(model.critic.getWeights());
    return newModel;
}

export async function downloadModel(model) {
    const artifacts = await model.save(
        tf.io.withSaveHandler(async (artifacts) => {
            return artifacts;
        })
    );
    artifacts.weightData = Array.from(new Float32Array(artifacts.weightData));
    return artifacts;
}

export async function serializeModels(models, currentModel) {
    if (currentModel) {
        models.push(currentModel);
    }
    const arrayOfModels = models;

    const serializedModels = [];
    for (const model of arrayOfModels) {
        const artifactsActor = await downloadModel(model.actor);
        const artifactsCritic = await downloadModel(model.critic);
        const artifacts = {
            actor: artifactsActor,
            critic: artifactsCritic
        };
        serializedModels.push(artifacts);
    }
    return serializedModels;
}

top.saveModels = async function () {
    const serializedModels = await serializeModels(top.models(), top.currentModel());
    await saveBrowserFile(serializedModels, "models.json");
}

export async function loadModelFromArtifacts(artifacts) {
    const model = setupModel();
    model.actor.loadWeights(new Float32Array(artifacts.actor.weightData).buffer, artifacts.actor.weightSpecs);
    model.critic.loadWeights(new Float32Array(artifacts.critic.weightData).buffer, artifacts.critic.weightSpecs);
    return model;
}

export async function deserializeModels(arrayOfArtifacts) {
    const models = [];
    for (const artifacts of arrayOfArtifacts) {
        const model = await loadModelFromArtifacts(artifacts);
        models.push(model);
    }
    const currentModel = models.pop();
    return { models, currentModel };
}

export async function saveBrowserFile(filedata, filename) {
    const jsonString = JSON.stringify(filedata);
    const blob = new Blob([jsonString], { type: "application/json" });
    const url = URL.createObjectURL(blob);

    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);

    URL.revokeObjectURL(url);
}

export function loadBrowserFile() {
    return new Promise((resolve, reject) => {
        const input = document.createElement("input");
        input.type = "file";
        input.accept = "application/json";

        input.onchange = (event) => {
            const file = event.target.files[0];
            const reader = new FileReader();
            reader.onload = (e) => {
                try {
                    const content = e.target.result;
                    const filedata = JSON.parse(content);
                    resolve(filedata);
                } catch (error) {
                    reject(error);
                }
            };
            reader.onerror = (e) => {
                reject(e);
            };
            reader.readAsText(file);
        };

        input.click();
    });
}