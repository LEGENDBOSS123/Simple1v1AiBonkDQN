export class Memory {
    constructor(state = [], action = [], reward = 0, nextState = [], done = false) {
        this.state = state;
        this.action = action;
        this.reward = reward;
        this.nextState = nextState;
        this.done = done;
    }
}