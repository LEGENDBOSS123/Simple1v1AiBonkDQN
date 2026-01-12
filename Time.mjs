export class Time {
    static sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}