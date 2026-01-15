import { CONFIG } from "./config.mjs";
import { tf } from "./tf.mjs";


function setupDuelingModel() {
    const input = tf.input({ shape: [CONFIG.GAME_STATE_SIZE] });

    let x = input;
    let layerIndex = 1;
    for (const units of CONFIG.DUELING_SHARED_LAYER_LENGTHS) {
        x = tf.layers.dense({
            name: `dense_Dense${layerIndex++}`,
            units: units,
            activation: 'relu',
            kernelInitializer: 'heNormal'
        }).apply(x);
    }


    let critic = tf.layers.dense({
        name: `dense_Dense${layerIndex++}`,
        units: 1,
        activation: 'linear',
        kernelInitializer: 'glorotUniform'
    }).apply(x);


    const advantages = [];
    for (let i = 0; i < CONFIG.ACTION_SIZE; i++) {
        advantages.push(
            tf.layers.dense({
                units: 2,
                name: `adv_${i}`,
                activation: 'linear',
                kernelInitializer: 'glorotUniform'
            }).apply(x)
        );
    }

    return tf.model({
        inputs: input,
        outputs: [critic, ...advantages],
    });
}

export function setupModel() {
    const model = setupDuelingModel();
    const target = setupDuelingModel();
    target.setWeights(model.getWeights());
    const optimizer = tf.train.adam(CONFIG.LEARNING_RATE);
    return {
        model: model,
        target: target,
        optimizer: optimizer
    };
}


export function cloneModel(model) {
    const newModel = setupModel();
    newModel.model.setWeights(model.model.getWeights());
    newModel.target.setWeights(model.target.getWeights());
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
        const artifactsModel = await downloadModel(model.model);
        const artifactsTarget = await downloadModel(model.target);
        const artifacts = {
            model: artifactsModel,
            target: artifactsTarget
        };
        serializedModels.push(artifacts);
    }
    return serializedModels;
}

top.saveModels = async function () {
    const serializedModels = await serializeModels(top.models(), top.currentModel());
    const json = {
        models: serializedModels,
        per: top.memory().toJSON()
    }
    await saveBrowserFile(json, "models.json");
}

export async function loadModelFromArtifacts(artifacts) {
    const model = setupModel();
    console.log(artifacts.model.weightData, artifacts.model.weightSpecs);
    console.log(artifacts.target.weightData, artifacts.target.weightSpecs);
    for (let i = 0; i < 4; i++) {

        artifacts.model.weightSpecs[2 * i].name = "dense_Dense" + (i + 1) + "/kernel";
        artifacts.model.weightSpecs[2 * i + 1].name = "dense_Dense" + (i + 1) + "/bias";
    }
    for (let i = 0; i < 4; i++) {

        artifacts.target.weightSpecs[2 * i].name = "dense_Dense" + (i + 1) + "/kernel";
        artifacts.target.weightSpecs[2 * i + 1].name = "dense_Dense" + (i + 1) + "/bias";
    }

    model.model.loadWeights(tf.io.decodeWeights(new Float32Array(artifacts.model.weightData).buffer, artifacts.model.weightSpecs));
    model.target.loadWeights(tf.io.decodeWeights(new Float32Array(artifacts.target.weightData).buffer, artifacts.target.weightSpecs));
    console.log(model);
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