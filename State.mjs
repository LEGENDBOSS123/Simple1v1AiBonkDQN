import { CONFIG } from "./config.mjs";
import { keyMap } from "./move.mjs";

export class State {
    constructor() {
        this.player1 = {
            x: 0,
            y: 0,
            vx: 0,
            vy: 0,
            keysPressed: {
                left: false,
                right: false,
                up: false,
                down: false,
                heavy: false,
                special: false
            },
            heavyAlpha: 0,
            grappleCooldown: 0
        };
        this.player2 = {
            x: 0,
            y: 0,
            vx: 0,
            vy: 0,
            keysPressed: {
                left: false,
                right: false,
                up: false,
                down: false,
                heavy: false,
                special: false
            },
            heavyAlpha: 0,
            grappleCooldown: 0
        };
        this.done = false;
        this.winnerId = null;
    }

    fetch() {
        const p1data = top.playerids[CONFIG.PLAYER_ONE_ID].playerData2;
        const p1rawdata = top.playerids[CONFIG.PLAYER_ONE_ID].playerData;
        this.player1.x = p1data.px / top.scale;
        this.player1.y = p1data.py / top.scale;
        this.player1.vx = p1data.xvel / top.scale;
        this.player1.vy = p1data.yvel / top.scale;
        try {
            this.player1.heavyAlpha = p1rawdata.children[3].alpha;
        } catch (e) {
            this.player1.heavyAlpha = 0;
        }
        this.player1.grappleCooldown = 36 / 184;
        try {
            if (p1rawdata.children[5].batches.length == 1) {
                this.player1.grappleCooldown = p1rawdata.children[5].batches[0].vertexData.length / 184;
            }
        } catch (e) { }
        this.player1.keysPressed = top.GET_KEYS(keyMap.get(CONFIG.PLAYER_ONE_ID) ?? 0);


        const p2data = top.playerids[CONFIG.PLAYER_TWO_ID].playerData2;
        const p2rawdata = top.playerids[CONFIG.PLAYER_TWO_ID].playerData;
        try {
            this.player2.heavyAlpha = p2rawdata.children[3].alpha;
        } catch (e) {
            this.player2.heavyAlpha = 0;
        }
        this.player2.grappleCooldown = 36 / 184;
        try {
            if (p2rawdata.children[5].batches.length == 1) {
                this.player2.grappleCooldown = p2rawdata.children[5].batches[0].vertexData.length / 184;
            }
        } catch (e) { }
        this.player2.x = p2data.px / top.scale;
        this.player2.y = p2data.py / top.scale;
        this.player2.vx = p2data.xvel / top.scale;
        this.player2.vy = p2data.yvel / top.scale;
        this.player2.keysPressed = top.GET_KEYS(keyMap.get(CONFIG.PLAYER_TWO_ID) ?? 0);

        this.done = !p1data.alive || !p2data.alive;
        if (this.done) {
            if (!p1data.alive && !p2data.alive) {
                this.winnerId = null;
            } else if (!p1data.alive) {
                this.winnerId = CONFIG.PLAYER_TWO_ID;
            } else {
                this.winnerId = CONFIG.PLAYER_ONE_ID;
            }
        }
    }

    reward() {
        if (this.winnerId === CONFIG.PLAYER_ONE_ID) {
            return {
                p1: 1,
                p2: -1
            };
        } else if (this.winnerId === CONFIG.PLAYER_TWO_ID) {
            return {
                p1: -1,
                p2: 1
            };
        }

        const dist = Math.hypot(this.player1.x - this.player2.x, this.player1.y - this.player2.y) * CONFIG.POSITION_NORMALIZATION;
        const closenessReward = Math.max(0, (1 - dist) * 0.01);

        const timePenalty = -0.0025;

        const p1Reward = timePenalty + closenessReward;
        const p2Reward = timePenalty + closenessReward;

        return {
            p1: p1Reward,
            p2: p2Reward
        };
    }

    flip() {
        const newState = new State();
        newState.player1 = structuredClone(this.player2);
        newState.player2 = structuredClone(this.player1);
        newState.done = this.done;
        newState.winnerId = this.winnerId;
        return newState;
    }

    toArray() {
        return [
            this.player1.x * CONFIG.POSITION_NORMALIZATION,
            this.player1.y * CONFIG.POSITION_NORMALIZATION,
            this.player1.vx * CONFIG.VELOCITY_NORMALIZATION,
            this.player1.vy * CONFIG.VELOCITY_NORMALIZATION,
            this.player1.heavyAlpha,
            this.player1.grappleCooldown,
            this.player1.keysPressed.left ? 1 : 0,
            this.player1.keysPressed.right ? 1 : 0,
            this.player1.keysPressed.up ? 1 : 0,
            this.player1.keysPressed.down ? 1 : 0,
            this.player1.keysPressed.heavy ? 1 : 0,
            this.player1.keysPressed.special ? 1 : 0,

            this.player2.x * CONFIG.POSITION_NORMALIZATION,
            this.player2.y * CONFIG.POSITION_NORMALIZATION,
            this.player2.vx * CONFIG.VELOCITY_NORMALIZATION,
            this.player2.vy * CONFIG.VELOCITY_NORMALIZATION,
            this.player2.heavyAlpha,
            this.player2.grappleCooldown,
            this.player2.keysPressed.left ? 1 : 0,
            this.player2.keysPressed.right ? 1 : 0,
            this.player2.keysPressed.up ? 1 : 0,
            this.player2.keysPressed.down ? 1 : 0,
            this.player2.keysPressed.heavy ? 1 : 0,
            this.player2.keysPressed.special ? 1 : 0,

            CONFIG.POSITION_NORMALIZATION * (this.player1.x - this.player2.x),
            CONFIG.POSITION_NORMALIZATION * (this.player1.y - this.player2.y),
            CONFIG.VELOCITY_NORMALIZATION * (this.player1.vx - this.player2.vx),
            CONFIG.VELOCITY_NORMALIZATION * (this.player1.vy - this.player2.vy)
        ];
    }
}