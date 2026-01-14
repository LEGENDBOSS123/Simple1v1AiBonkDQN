import { CONFIG } from "./config.mjs";
import { log } from "./log.mjs";
import { actionToArray, arrayToAction, getAction, move, predictActionArray, randomAction } from "./move.mjs";
import { Random } from "./Random.mjs";
import { setupLobby } from "./setupLobby.mjs";
import { cloneModel, deserializeModels, loadBrowserFile, setupModel } from "./setupModel.mjs";
import { State } from "./State.mjs";
import { tf } from "./tf.mjs";
import { Time } from "./Time.mjs";
import { train } from "./train.mjs";

let models = [];
let currentModel = null;
let memory = [];

let actor_losses = [];
let critic_losses = [];

top.models = function () { return models; };
top.currentModel = function () { return currentModel; };
top.memory = function () { return memory; };
top.actor_losses = function () { return actor_losses; };
top.critic_losses = function () { return critic_losses; };
top.paused = false;

async function setup() {
    const filePrompt = prompt("Do you want to load a model from file? (y/n)", "n") == "y";
    const filedata = filePrompt ? await loadBrowserFile() : null;
    if (filedata) {
        const deserialized = await deserializeModels(filedata);
        models = deserialized.models;
        currentModel = deserialized.currentModel;
        log(`Loaded ${models.length} models from file.`);
    }
    else {
        currentModel = setupModel();
    }
    log("TensorFlow.js version:", tf.version.tfjs);
    log(`Actor model initialized with ${currentModel.actor.countParams()} parameters.`);
    log(`Critic model initialized with ${currentModel.critic.countParams()} parameters.`);
}


async function main() {

    await setupLobby();

    log("Lobby setup complete.");


    async function gameLoop() {

        while (true) {
            // match start
            top.startGame();
            await Time.sleep(1500);

            // 20 TPS
            let TPS = 1000 / 20;

            let lastState = new State();
            lastState.fetch();
            await Time.sleep(TPS);

            let newState;

            let safeFrames = 0;

            let p2Model = currentModel;
            if (Math.random() < 0.4 && models.length > 1) {
                p2Model = Random.choose(models);
            }

            let lastActionP1 = null;  // Track the action we took

            while (true) {

                newState = new State();
                newState.fetch();

                let rewardCurrentFrame = newState.reward();
                let rewardP1 = rewardCurrentFrame.p1;

                // Store the action we ACTUALLY took last frame (not re-sampled)
                if (lastActionP1 !== null) {
                    memory.push({
                        state: lastState.toArray(),
                        action: lastActionP1,  // The exact sampled action
                        reward: rewardP1,
                        nextState: newState.toArray(),
                        done: newState.done
                    });
                }

                if (memory.length >= CONFIG.BATCH_SIZE) {
                    log("Training...");
                    const loss = await train(currentModel, memory);
                    if (loss) {
                        actor_losses.push(loss.actorLoss);
                        critic_losses.push(loss.criticLoss);
                        log(`Loss - Actor: ${loss.actorLoss.toFixed(4)}`);
                        log(`Loss - Critic: ${loss.criticLoss.toFixed(4)}`);
                    }
                    if (actor_losses.length % CONFIG.SAVE_AFTER_EPISODES === 0) {
                        models.push(cloneModel(currentModel));
                        if (models.length > 5) {
                            models.shift();
                        }
                    }
                    memory.length = 0;
                    newState.done = true;
                }

                if (safeFrames > 300 || newState.done) {
                    break;
                }

                // Sample action and store the EXACT sample
                let probs = predictActionArray(currentModel.actor, newState.toArray());
                let ataP1 = arrayToAction(probs);
                move(CONFIG.PLAYER_ONE_ID, ataP1);
                lastActionP1 = actionToArray(ataP1);  // Store the exact binary array

                // P2
                // let probs2 = predictActionArray(p2Model.actor, newState.flip().toArray());
                // let actionObj2 = binaryArrayToActionObj(sampleActionFromProbs(probs2));
                // move(CONFIG.PLAYER_TWO_ID, actionObj2);

                safeFrames++;
                lastState = newState;

                // 20 FPS
                await Time.sleep(50);
            }

            if (top.paused) {
                log("Paused. Press OK to continue.");
                top.paused = false;
                return;
            }
        }
    }

    gameLoop();
}

function pause() {
    top.paused = true;
}

top.main = main;
setup().then(() => { main() });