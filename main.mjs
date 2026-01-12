import { CONFIG } from "./config.mjs";
import { log } from "./log.mjs";
import { getAction, move, predictActionArray, sampleBernoulli } from "./move.mjs";
import { Random } from "./Random.mjs";
import { setupLobby } from "./setupLobby.mjs";
import { setupModel } from "./setupModel.mjs";
import { State } from "./State.mjs";
import { tf } from "./tf.mjs";

await setupLobby();

let models = [];
let memory = [];
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
        while (true) {

            newState = new State();
            newState.fetch();

            let rewardCurrentFrame = newState.reward();
            let rewardP1 = rewardCurrentFrame.p1;
            let rewardP2 = rewardCurrentFrame.p2;

            let actionP1 = getAction(CONFIG.PLAYER_ONE_ID);
            let actionP2 = getAction(CONFIG.PLAYER_TWO_ID);

            // player1 memory
            memory.push({
                state: lastState.toArray(),
                action: actionP1,
                reward: rewardP1,
                nextState: newState.toArray(),
                done: newState.done
            });


            let predictedActionP1 = predictActionArray(models.at(-1).actor, newState.toArray());
            log(predictedActionP1);
            move(CONFIG.PLAYER_ONE_ID, sampleBernoulli(predictedActionP1));

            safeFrames++;
            if (safeFrames > 300) {
                log("Match timed out after 15 seconds.");
                break;
            }
            // 20 FPS
            await Time.sleep(50);
            lastState = newState;
        }
    }
}

gameLoop();