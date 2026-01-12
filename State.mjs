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
            }
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
            }
        };
    }

    fetch() {
        const p1data = top.playerids[CONFIG.PLAYER_ONE_ID].playerData2;
        this.player1.x = p1data.px / top.scale;
        this.player1.y = p1data.py / top.scale;
        this.player1.vx = p1data.xvel / top.scale;
        this.player1.vy = p1data.yvel / top.scale;

        this.player1.keysPressed = top.GET_KEYS(keyMap.get(CONFIG.PLAYER_ONE_ID));

        const p2data = top.playerids[CONFIG.PLAYER_TWO_ID].playerData2;
        this.player2.x = p2data.px / top.scale;
        this.player2.y = p2data.py / top.scale;
        this.player2.vx = p2data.xvel / top.scale;
        this.player2.vy = p2data.yvel / top.scale;
        this.player2.keysPressed = top.GET_KEYS(keyMap.get(CONFIG.PLAYER_TWO_ID));
    }

    toArray() {
        return [
            this.player1.x / CONFIG.POSITION_NORMALIZATION,
            this.player1.y / CONFIG.POSITION_NORMALIZATION,
            this.player1.vx / CONFIG.VELOCITY_NORMALIZATION,
            this.player1.vy / CONFIG.VELOCITY_NORMALIZATION,
            this.player1.keysPressed.left ? 1 : 0,
            this.player1.keysPressed.right ? 1 : 0,
            this.player1.keysPressed.up ? 1 : 0,
            this.player1.keysPressed.down ? 1 : 0,
            this.player1.keysPressed.heavy ? 1 : 0,
            this.player1.keysPressed.special ? 1 : 0,

            this.player2.x / CONFIG.POSITION_NORMALIZATION,
            this.player2.y / CONFIG.POSITION_NORMALIZATION,
            this.player2.vx / CONFIG.VELOCITY_NORMALIZATION,
            this.player2.vy / CONFIG.VELOCITY_NORMALIZATION,
            this.player2.keysPressed.left ? 1 : 0,
            this.player2.keysPressed.right ? 1 : 0,
            this.player2.keysPressed.up ? 1 : 0,
            this.player2.keysPressed.down ? 1 : 0,
            this.player2.keysPressed.heavy ? 1 : 0,
            this.player2.keysPressed.special ? 1 : 0
        ];
    }
}