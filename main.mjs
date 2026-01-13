import { CONFIG } from "./config.mjs";
import { log } from "./log.mjs";
import { arrayToAction, getAction, move, predictActionArray } from "./move.mjs";
import { Random } from "./Random.mjs";
import { setupLobby } from "./setupLobby.mjs";
import { cloneModel, setupModel } from "./setupModel.mjs";
import { State } from "./State.mjs";
import { tf } from "./tf.mjs";
import { Time } from "./Time.mjs";
import { train } from "./train.mjs";



async function main() {

    await setupLobby();

    let models = [];
    let memory = [];
    let actor_losses = [];
    let critic_losses = [];
    top.memory = memory;
    top.models = models;
    top.actor_losses = actor_losses;
    top.critic_losses = critic_losses;
    let currentModel = setupModel();

    log("Lobby set up. TensorFlow.js version:", tf.version.tfjs);
    log(`Actor model initialized with ${currentModel.actor.countParams()} parameters.`);
    log(`Critic model initialized with ${currentModel.critic.countParams()} parameters.`);

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


                if (memory.length >= CONFIG.BATCH_SIZE) {
                    log("Training...");

                    const loss = await train(currentModel, memory);
                    if (loss) {
                        actor_losses.push(loss.actorLoss);
                        critic_losses.push(loss.criticLoss);
                        log(`Loss - Actor: ${loss.actorLoss.toFixed(4)}`);
                        log(`Loss - Critic: ${loss.criticLoss.toFixed(4)}`);
                    }
                    if(actor_losses.length % CONFIG.SAVE_AFTER_EPISODES === 0) {
                        models.push(cloneModel(currentModel));
                    }
                    memory.length = 0;
                }

                if (safeFrames > 300 || newState.done) {
                    log("Round over.");
                    break;
                }

                let predictedActionsP1 = predictActionArray(currentModel.actor, newState.toArray());
                if (safeFrames % 60 === 0) {
                    console.log("AI Confidence:", predictedActionsP1.map(p => p.toFixed(2)));
                }
                let predictedActionsP2 = predictActionArray(p2Model.actor, newState.flip().toArray());
                move(CONFIG.PLAYER_ONE_ID, arrayToAction(predictedActionsP1));
                move(CONFIG.PLAYER_TWO_ID, arrayToAction(predictedActionsP2));

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