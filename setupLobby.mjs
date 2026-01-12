import { CONFIG } from "./config.mjs";
import { map } from "./getMap.mjs";
import { Time } from "./Time.mjs";

export async function setupLobby() {
    if (top.settedUpLobby) {
        return;
    }
    top.commandhandle("/sandbox");
    top.commandhandle("/addname Bot 1");
    top.commandhandle("/addname Bot 2");

    await Time.sleep(500);
    let ids = Object.keys(playerids).sort((a, b) => { return Number(b) - Number(a) }).slice(0, 2).map(Number);
    CONFIG.PLAYER_ONE_ID = ids[0];
    CONFIG.PLAYER_TWO_ID = ids[1];

    top.loadMap(map);
    top.commandhandle("/moveA s");
    await Time.sleep(500);
    top.SEND(`42[26,{"targetID":${CONFIG.PLAYER_ONE_ID},"targetTeam":1}]`);
    top.SEND(`42[26,{"targetID":${CONFIG.PLAYER_TWO_ID},"targetTeam":1}]`);
    await Time.sleep(500);
    top.settedUpLobby = true;
}