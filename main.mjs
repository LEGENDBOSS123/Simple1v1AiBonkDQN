import { CONFIG } from "./config.mjs";
import { log } from "./log.mjs";
import { arrayToAction, getAction, move, predictActionArray } from "./move.mjs";
import { Random } from "./Random.mjs";
import { setupLobby } from "./setupLobby.mjs";
import { setupModel } from "./setupModel.mjs";
import { State } from "./State.mjs";
import { tf } from "./tf.mjs";
import { Time } from "./Time.mjs";
import { train } from "./train.mjs";



async function main() {

    await setupLobby();

    let models = [];
    let memory = [];
    top.memory = memory;
    top.models = models;
    models.push(setupModel());

    log("Lobby set up. TensorFlow.js version:", tf.version.tfjs);
    log(`Actor model initialized with ${models.at(-1).actor.countParams()} parameters.`);
    log(`Critic model initialized with ${models.at(-1).critic.countParams()} parameters.`);

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

            let p2Model = models.at(-1);
            if (Math.random() < 0.2 && models.length > 1) {
                p2Model = Random.choose(models);
            }


            while (true) {

                newState = new State();
                newState.fetch();



                let rewardCurrentFrame = newState.reward();
                let rewardP1 = rewardCurrentFrame.p1;

                let actionP1 = getAction(CONFIG.PLAYER_ONE_ID);

                // player1 memory
                memory.push({
                    state: lastState.toArray(),
                    action: actionP1,
                    reward: rewardP1,
                    nextState: newState.toArray(),
                    done: newState.done
                });

                if (newState.done) {
                    let runningReward = rewardP1;
                    for (let i = memory.length - 2; i >= 0; i--) {
                        if (memory[i].done) {
                            break;
                        }
                        runningReward = memory[i].reward + CONFIG.DISCOUNT_FACTOR * runningReward;
                        memory[i].reward = runningReward;
                    }
                }

                if (memory.length >= CONFIG.BATCH_SIZE) {
                    log("Training...");

                    const loss = await train(models.at(-1), memory);
                    if (loss) {
                        log(`Loss - Actor: ${loss.actorLoss.toFixed(4)}`);
                        log(`Loss - Critic: ${loss.criticLoss.toFixed(4)}`);
                    }
                    memory = [];
                }

                if (safeFrames > 300 || newState.done) {
                    log("Round over.");
                    break;
                }

                let predictedActionsP1 = predictActionArray(models.at(-1).actor, newState.toArray());
                let predictedActionsP2 = predictActionArray(p2Model.actor, newState.flip().toArray());
                move(CONFIG.PLAYER_ONE_ID, arrayToAction(predictedActionsP1));
                // move(CONFIG.PLAYER_TWO_ID, arrayToAction(predictedActionsP2));

                safeFrames++;
                lastState = newState;

                // 20 FPS
                await Time.sleep(50);
            }
        }
    }

    gameLoop();
}
main();