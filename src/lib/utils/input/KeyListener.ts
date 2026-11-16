import Key, {KEY_A, KEY_ARROW_DOWN, KEY_ARROW_LEFT, KEY_ARROW_RIGHT, KEY_ARROW_UP, KEY_D, KEY_S, KEY_W} from '../input/Key';
import Vector2 from '../../renderer/math/Vector2';

const keyState: { [k: string]: boolean } = {};
const keyHitCallbacks: { [k: string]: (() => void)[] } = {};

let _active: boolean = true;
//set while the window listeners are bound, destruct() aborts it to remove them
let listenerController: AbortController | null = null;

function bindKeyListenerEvents(win: any = window) {
    if (listenerController) return;
    listenerController = new AbortController();
    const signal = listenerController.signal;

    win.addEventListener(
        'keydown',
        (e: KeyboardEvent) => {
            if (!_active) return;

            const code = e.key.toLowerCase();
            keyState[code] = true;
            keyHitCallbacks[code]?.forEach((callback) => callback());
        },
        {signal},
    );

    win.addEventListener(
        'keyup',
        (e: KeyboardEvent) => {
            if (!_active) return;

            const code = e.key.toLowerCase();
            keyState[code] = false;
        },
        {signal},
    );

    win.addEventListener('blur', () => KeyListener.cancel(), {signal});
}

//removes the window listeners and the key hit callbacks. using the KeyListener again binds the listeners again
function destruct() {
    listenerController?.abort();
    listenerController = null;
    cancel();
    Object.keys(keyHitCallbacks).forEach((key) => delete keyHitCallbacks[key]);
}


function active(value: boolean) {
    if (_active !== value) {
        _active = value;
        if (!value) {
            cancel();
        }
    }
}

function cancel() {
    Object.keys(keyState).forEach((key) => (keyState[key] = false));
}

function getKeyCodeDown(key: Key) {
    bindKeyListenerEvents();
    return !!keyState[key.toLowerCase()];
}

//the arrow keys and WASD as a direction: x is right (+1) or left (-1), y is up / forward (+1) or down / back (-1).
//an arrow key and its letter add up
function getMoveDirection(out: Vector2 = new Vector2()): Vector2 {
    const down = (key: Key) => getKeyCodeDown(key) ? 1 : 0;
    out.x = down(KEY_ARROW_RIGHT) + down(KEY_D) - down(KEY_ARROW_LEFT) - down(KEY_A);
    out.y = down(KEY_ARROW_UP) + down(KEY_W) - down(KEY_ARROW_DOWN) - down(KEY_S);
    return out;
}

function addKeyHitEvent(key: Key, callback: () => void) {
    bindKeyListenerEvents();
    const code = key.toLowerCase();
    keyHitCallbacks[code] = keyHitCallbacks[code] || [];
    keyHitCallbacks[code]?.push(callback);
}

const KeyListener = {
    active,
    cancel,
    getKeyCodeDown,
    getMoveDirection,
    addKeyHitEvent,
    destruct,
}

export default KeyListener;