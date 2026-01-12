import { tf } from "./tf.mjs";

export const cValueMap = new Map();
export const keyMap = new Map();

export function getAction(id) {
    if (!keyMap.has(id)) {
        return top.GET_KEYS(0);
    }
    return GET_KEYS(keyMap.get(id));
}

export function actionToArray(action) {
    return [
        action.left ? 1 : 0,
        action.right ? 1 : 0,
        action.up ? 1 : 0,
        action.down ? 1 : 0,
        action.heavy ? 1 : 0,
        action.special ? 1 : 0
    ];
}

export function arrayToAction(arr) {
    return {
        left: arr[0] == 1,
        right: arr[1] == 1,
        up: arr[2] == 1,
        down: arr[3] == 1,
        heavy: arr[4] == 1,
        special: arr[5] == 1
    };
}

export function predictActionArray(model, state) {
    return tf.tidy(() => {
        const stateTensor = tf.tensor2d([state]);
        const actionProbs = model.predict(stateTensor);
        return actionProbs.dataSync();
    });
}

export function sampleBernoulli(p) {
    return Math.random() < p ? 1 : 0;
}

export function move(id, keys) {
    let cvalue = 100;
    if (cValueMap.has(id)) {
        cvalue = cValueMap.get(id);
        cValueMap.set(id, cvalue + 1);
    }
    else {
        cValueMap.set(id, cvalue);
    }
    keyMap.set(id, top.MAKE_KEYS(keys));
    let packet = `42[7,${id},{"i":${top.MAKE_KEYS(keys)},"f":${top.getCurrentFrame()},"c":${cvalue}}]`;
    top.SEND("42" + JSON.stringify([4, { "type": "fakerecieve", "from": top.playerids[myid].userName, "packet": [packet], to: [-1] }]));
    top.RECIEVE(packet);
}