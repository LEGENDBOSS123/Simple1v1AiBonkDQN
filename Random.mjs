export class Random {
    static choose(array) {
        const index = Math.floor(Math.random() * array.length);
        return array[index];
    }
}