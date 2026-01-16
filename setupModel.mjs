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
        per: top.memory().toJSON(),
        CONFIG: CONFIG
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
    // 1. Convert JSON to string
    const jsonString = JSON.stringify(filedata);

    // 2. Create a stream from the string data
    const stream = new Blob([jsonString]).stream();

    // 3. Pipe through GZIP compression
    const compressedStream = stream.pipeThrough(new CompressionStream("gzip"));

    // 4. Convert stream back to a Blob
    const compressedBlob = await new Response(compressedStream).blob();

    // 5. Download logic (Standard)
    const url = URL.createObjectURL(compressedBlob);
    const a = document.createElement("a");
    a.href = url;
    // Append .gz to indicate compression, or keep original if you handle extension logic elsewhere
    a.download = filename.endsWith('.gz') ? filename : filename + '.gz';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);

    URL.revokeObjectURL(url);
}

/**
 * Loads a GZIP compressed file from the user and parses it back to JSON.
 */
export function loadBrowserFile() {
    return new Promise((resolve, reject) => {
        const input = document.createElement("input");
        input.type = "file";
        // Accept .json or .gz files
        input.accept = "application/json, .gz";

        input.onchange = async (event) => {
            const file = event.target.files[0];
            if (!file) return;

            try {
                // 1. Get the stream from the file
                const stream = file.stream();

                // 2. Pipe through GZIP decompression
                const decompressedStream = stream.pipeThrough(new DecompressionStream("gzip"));

                // 3. Read the decompressed stream as text
                const text = await new Response(decompressedStream).text();

                // 4. Parse JSON
                const filedata = JSON.parse(text);
                resolve(filedata);
            } catch (error) {
                console.error("Decompression failed. Was the file actually compressed?", error);
                // Fallback: Try reading as plain JSON in case the user uploaded an uncompressed file
                try {
                    const text = await file.text();
                    resolve(JSON.parse(text));
                } catch (fallbackError) {
                    reject(error);
                }
            }
        };

        input.click();
    });
}