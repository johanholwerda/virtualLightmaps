import Vector2 from '../../renderer/math/Vector2';
import Camera from '../../renderer/camera/Camera';
import Vector3 from '../../renderer/math/Vector3';
import MouseListener from '../../renderer/input/MouseListener';
import Renderer from '../../renderer/render/Renderer';
import {clamp01, dampingFactor} from '../../renderer/utils/MathUtils';
import KeyListener from '../input/KeyListener';
import {KEY_SHIFT} from '../input/Key';
import StreetviewRotationController from '../input/StreetviewRotationController';

export type FlyCameraOptions = {
  fov: number;
  nearPlane: number;
  farPlane: number;
  invertMouse: boolean;
};

export const defaultFlyCameraOptions: FlyCameraOptions = {
  fov: 1,
  nearPlane: 0.01,
  farPlane: 100,
  invertMouse: true,
};

export default class FlyCamera extends Camera {
  protected _mouseListener: MouseListener;
  protected _translateSpeedModifier: number = 10.0;
  protected _dampingModifier: number = 1.0;
  private _renderer: Renderer;
  //per second
  private _translateSpeed: Vector3 = new Vector3();
  private moveDirection: Vector2 = new Vector2();
  private _rotationController: StreetviewRotationController;
  //seconds after the last input in which the camera comes to a full stop, 0: the damping alone, which never quite stops
  private _stopDuration: number = 0;
  private _translateFade: number = 0;
  //false while the left mouse button drags something else, like a Pivot
  public mouseRotationEnabled: boolean = true;

  constructor(renderer: Renderer, opts: Partial<FlyCameraOptions> = {}) {
    const options = {...defaultFlyCameraOptions, ...opts};
    super(options);

    this._renderer = renderer;

    this._mouseListener = renderer.mouseListener;

    this._rotationController = new StreetviewRotationController(renderer, options.invertMouse);
  }

  public set eulerXY(rot: Vector2) {
    this._rotationController.eulerXY = rot;
  }

  public lookAt(pos: Vector3, target: Vector3, up: Vector3 = Vector3.UP): void {
    super.lookAt(pos, target, up);
    this._rotationController.rotation = this.view.rotation;
  }

  //a full stop within seconds after the last input, for renderers that accumulate while the camera stands still
  public set stopDuration(seconds: number) {
    this._stopDuration = seconds;
    this._rotationController.easing = seconds;
  }

  public get stopDuration(): number {
    return this._stopDuration;
  }

  public resetSpeed() {
    this._translateSpeed.setValues(0, 0, 0);
    this._rotationController.resetSpeed();
  }

  public resetTranslateSpeed() {
    this._translateSpeed.setValues(0, 0, 0);
  }

  public rotateBy(rot: Vector2) {
    this._rotationController.rotateBy(rot);
    this.view.rotation = this._rotationController.rotation;
  }

  public update(): void {
    this.projection.aspectRatio = this._renderer.aspectRatio;

    const direction = KeyListener.getMoveDirection(this.moveDirection);
    if (direction.x) this.moveSideWays(-direction.x);
    if (direction.y) this.moveForward(-direction.y);

    //by dt, so it moves and slows down as fast at any frame rate. the same as before at 60 fps
    const dt = this._renderer.time.dt;
    const speed = this._translateSpeed;
    this.view.translateValues(speed.x * dt, speed.y * dt, speed.z * dt);
    speed.multiplyScalar(dampingFactor(Math.pow(0.95, this._dampingModifier), dt));
    //on top of the damping a linear fade to 0 since the last key input, like the easing of StreetviewRotationController
    if (this._stopDuration > 0) {
      if (direction.x || direction.y) {
        this._translateFade = 1;
      } else if (this._translateFade > 0) {
        const fade = clamp01(this._translateFade - dt / this._stopDuration);
        speed.multiplyScalar(fade / this._translateFade);
        this._translateFade = fade;
      }
    }
    if (this.mouseRotationEnabled) {
      this._rotationController.update(this);
    } else {
      this._rotationController.resetSpeed();
    }

    this.view.rotation = this._rotationController.rotation;

    this.checkChanged();
  }

  public getMouseDown(): boolean {
    return this._mouseListener.mouseDown;
  }

  public getMouseSpeed(): Vector2 {
    if (this._mouseListener.mouseDown) {
      return this._mouseListener.normalizedVelocity;
    } else {
      return new Vector2();
    }
  }

  public setTranslateSpeed(speed: number) {
    this._translateSpeedModifier = speed;
  }

  public getTranslateSpeed() {
    return this._translateSpeedModifier;
  }

  public setDamping(damping: number) {
    this._dampingModifier = damping;
  }

  //3x as fast while shift is held
  private get boostedTranslateSpeed(): number {
    return this._translateSpeedModifier * (KeyListener.getKeyCodeDown(KEY_SHIFT) ? 3 : 1);
  }

  //the speed is per second: * 60 keeps the acceleration of when it was per frame at 60 fps
  private moveForward(amount: number) {
    const forward: Vector3 = this.view.forward;
    forward.multiplyScalar(this.boostedTranslateSpeed * amount * this._renderer.time.dt * 60);
    this._translateSpeed.add(forward);
  }

  private moveSideWays(amount: number) {
    const right = this.view.worldRight;
    right.multiplyScalar(-this.boostedTranslateSpeed * amount * this._renderer.time.dt * 60);
    this._translateSpeed.add(right);
  }
}
