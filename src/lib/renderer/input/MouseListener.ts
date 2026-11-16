import Vector2 from '../math/Vector2';
import IDestructible from '../core/IDestructible';

const LEFT = 0;
const MIDDLE = 1;
const RIGHT = 2;
//PointerEvent.buttons bit of each button, by button index
const BUTTON_BITS = [1, 4, 2];

/*
mouse, touch and pen through pointer events, so a tap is not handled again as the mouse events the browser emulates
for it. only the primary pointer counts: the first finger, it acts as the left button
 */
export default class MouseListener implements IDestructible {
  private readonly eventElement: HTMLElement;
  private readonly sizeElement: HTMLElement;
  private readonly listenerController = new AbortController();

  private wheelCallbacks: ((dir: number) => void)[] = [];
  private clickCallbacks: (() => void)[] = [];
  private downCallbacks: ((pos: Vector2) => void)[] = [];
  private upCallbacks: (() => void)[] = [];
  private moveCallbacks: ((pos: Vector2) => void)[] = [];
  private dragCallbacks: ((distance: Vector2) => void)[] = [];

  private mousePos: Vector2 = new Vector2();
  private mousePosPrev: Vector2 = new Vector2();
  private mouseVelocity: Vector2 = new Vector2();

  //by button index: left, middle, right
  private buttonDown: boolean[] = [false, false, false];
  //pressed or released since the last update(), also when that happened within one frame (a quick tap)
  private pressedSinceUpdate: boolean[] = [false, false, false];
  private releasedSinceUpdate: boolean[] = [false, false, false];
  //the same, for the frame after update()
  private pressedThisFrame: boolean[] = [false, false, false];
  private releasedThisFrame: boolean[] = [false, false, false];

  private clickCanceled: boolean = false;
  private timeOfMouseDown: number = 0;

  constructor(eventElement: HTMLElement, sizeElement: HTMLElement) {
    this.eventElement = eventElement;
    this.sizeElement = sizeElement;
    if (!eventElement) return;

    //touch drags would otherwise scroll or zoom the page, and the browser then cancels the pointer
    eventElement.style.touchAction = 'none';

    const options = { signal: this.listenerController.signal };
    eventElement.addEventListener('pointerdown', this.onPointerDown, options);
    eventElement.addEventListener('pointermove', this.onPointerMove, options);
    eventElement.addEventListener('pointerup', this.onPointerUp, options);
    eventElement.addEventListener('pointercancel', this.onPointerCancel, options);
    eventElement.addEventListener('click', this.onClick, options);
    eventElement.addEventListener('wheel', this.onWheel, options);
    // Prevent right-click menu
    eventElement.addEventListener('contextmenu', (event) => event.preventDefault(), options);
  }

  private onPointerDown = (event: PointerEvent) => {
    if (!event.isPrimary) return;
    //the up and the moves of a drag still arrive when the pointer leaves the element
    try {
      this.eventElement.setPointerCapture(event.pointerId);
    } catch {
      //the pointer is already gone
    }
    this.setMousePosition(event);
    this.setButtons(event.buttons);
  };

  //also when another button is pressed or released while one is down: then there is no pointerdown or pointerup
  private onPointerMove = (event: PointerEvent) => {
    if (!event.isPrimary) return;
    this.setMousePosition(event);
    this.setButtons(event.buttons);
    for (const callback of this.moveCallbacks) callback(this.mousePos.clone());
  };

  private onPointerUp = (event: PointerEvent) => {
    if (!event.isPrimary) return;
    this.setMousePosition(event);
    this.setButtons(event.buttons);
  };

  private onPointerCancel = (event: PointerEvent) => {
    if (!event.isPrimary) return;
    this.setButtons(0);
  };

  private onClick = () => {
    if (!this.clickCanceled) {
      for (const callback of this.clickCallbacks) callback();
    }
  };

  private onWheel = (event: WheelEvent) => {
    const dir = Math.sign(-event.deltaY);
    for (const callback of this.wheelCallbacks) callback(dir);
  };

  private setMousePosition(event: PointerEvent): void {
    const x = event.clientX || 0;
    const y = event.clientY || 0;
    if (this.sizeElement && this.sizeElement.getBoundingClientRect) {
      const bounds = this.sizeElement.getBoundingClientRect();
      this.mousePos.x = (x - bounds.left) / bounds.width;
      this.mousePos.y = (y - bounds.top) / bounds.height;
    } else {
      this.mousePos.setValues(x, y);
    }
  }

  private setButtons(buttons: number) {
    for (let button = 0; button < 3; button++) {
      const down = (buttons & BUTTON_BITS[button]) !== 0;
      if (down === this.buttonDown[button]) continue;
      this.buttonDown[button] = down;

      if (down) {
        this.pressedSinceUpdate[button] = true;
        this.clickCanceled = false;
        this.timeOfMouseDown = performance.now();
        //no velocity from where the pointer was before
        this.mousePosPrev.copy(this.mousePos);
        if (button === LEFT) {
          for (const callback of this.downCallbacks) callback(this.mousePos.clone());
        }
      } else {
        this.releasedSinceUpdate[button] = true;
        if (button === LEFT) {
          for (const callback of this.upCallbacks) callback();
        }
      }
    }
  }

  // (0,0) x (1, 1)
  public get normalizedPosition(): Vector2 {
    return this.mousePos.clone();
  }

  // (-1,-1) x (1, 1)
  public get screenPosition(): Vector2 {
    return new Vector2(this.mousePos.x * 2 - 1, 1 - this.mousePos.y * 2);
  }

  public get normalizedVelocity(): Vector2 {
    return this.mouseVelocity.clone();
  }

  // Left mouse button
  public get mouseDown(): boolean {
    return this.buttonDown[LEFT];
  }

  public get mouseDownThisFrame(): boolean {
    return this.pressedThisFrame[LEFT];
  }

  public get mouseUpThisFrame(): boolean {
    return this.releasedThisFrame[LEFT];
  }

  public get middleMouseDown(): boolean {
    return this.buttonDown[MIDDLE];
  }

  public get middleMouseDownThisFrame(): boolean {
    return this.pressedThisFrame[MIDDLE];
  }

  public get middleMouseUpThisFrame(): boolean {
    return this.releasedThisFrame[MIDDLE];
  }

  public get rightMouseDown(): boolean {
    return this.buttonDown[RIGHT];
  }

  public get rightMouseDownThisFrame(): boolean {
    return this.pressedThisFrame[RIGHT];
  }

  public get rightMouseUpThisFrame(): boolean {
    return this.releasedThisFrame[RIGHT];
  }

  public addWheelEvent(callback: (dir: number) => void) {
    this.wheelCallbacks.push(callback);
  }

  public addClickEvent(callback: () => void) {
    this.clickCallbacks.push(callback);
  }

  public addDownEvent(callback: (pos: Vector2) => void) {
    this.downCallbacks.push(callback);
  }

  public addMoveEvent(callback: (pos: Vector2) => void) {
    this.moveCallbacks.push(callback);
  }

  public addUpEvent(callback: () => void) {
    this.upCallbacks.push(callback);
  }

  public addDragEvent(callback: (distance: Vector2) => void) {
    this.dragCallbacks.push(callback);
  }

  //removes the dom event listeners and the callbacks
  destruct() {
    this.listenerController.abort();
    for (const callbacks of [this.wheelCallbacks, this.clickCallbacks, this.downCallbacks, this.upCallbacks, this.moveCallbacks, this.dragCallbacks]) {
      callbacks.length = 0;
    }
  }

  //once per frame, before the app reads the state
  public update(): void {
    this.mouseVelocity.copy(this.mousePos).subtract(this.mousePosPrev);
    this.mousePosPrev.copy(this.mousePos);

    if (this.mouseDown && this.mouseVelocity.length() > 0) {
      if (this.mouseVelocity.length() > 0.001) {
        this.clickCanceled = true; // drag cancels click
      }
      for (const callback of this.dragCallbacks) callback(this.mouseVelocity);
    }

    if (performance.now() - this.timeOfMouseDown > 500) {
      this.clickCanceled = true;
    }

    for (let button = 0; button < 3; button++) {
      this.pressedThisFrame[button] = this.pressedSinceUpdate[button];
      this.releasedThisFrame[button] = this.releasedSinceUpdate[button];
      this.pressedSinceUpdate[button] = false;
      this.releasedSinceUpdate[button] = false;
    }
  }
}
