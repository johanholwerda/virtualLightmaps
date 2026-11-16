import RotationController from '../input/RotationController';
import MouseListener from '../../renderer/input/MouseListener';
import Renderer from '../../renderer/render/Renderer';
import Vector2 from '../../renderer/math/Vector2';
import Camera from '../../renderer/camera/Camera';
import {clamp01} from '../../renderer/utils/MathUtils';
import Vector4 from "webGPU-framework/renderer/math/Vector4";

export default class StreetviewRotationController extends RotationController {
  public static MOUSE_MOVEMENT_CANCEL_TRIGGER: number = 0.0025;
  private static MAX_ROTATE_SPEED: number = 0.175;
  private mouseListener: MouseListener;
  private renderer: Renderer;
  private mouseDown: boolean = false;

  private _eulerSpeed: Vector2 = new Vector2();

  private _rotating: boolean = false;
  private _rotatingLerp: number = 0;
  private _rotatingLerpPrev: number = 0;
  //-1 indicates no limit
  private _rotationLimitX: Vector2 = new Vector2(-1.4, 1.4);
  private _rotationLimitY: Vector2 = new Vector2(-Math.PI, Math.PI);
  private _clampEuler: boolean;
  public easing = 1.;
  private _rotationSpeedMultiplier: number = 3;

  constructor(renderer: Renderer, mouseInverse: boolean, clampEuler: boolean = true) {
    super(mouseInverse);
    this.renderer = renderer;
    this.mouseListener = renderer.mouseListener;
    this._clampEuler = clampEuler;
  }

  public set rotationSpeedMultiplier(m: number) {
    this._rotationSpeedMultiplier = m;
  }

  public get rotationSpeed(): Vector2 {
    return this._eulerSpeed;
  }

  public set rotationLimit(v: Vector4) {
    this.rotationLimitX = new Vector2(v.x, v.y);
    this.rotationLimitY = new Vector2(v.z, v.w);
  }

  public set rotationLimitX(v: Vector2) {
    this._rotationLimitX.copy(v);
  }

  public set rotationLimitY(v: Vector2) {
    this._rotationLimitY.copy(v);
  }

  public get rotating(): boolean {
    return this._rotating;
  }

  private get mouseSpeed(): Vector2 {
    if (this.mouseListener.mouseDown) {
      return this.mouseListener.normalizedVelocity;
    } else {
      return new Vector2();
    }
  }

  public update(camera: Camera): void {
    const mouseDownPreviousFrame = this.mouseDown;
    this.mouseDown = this.mouseListener.mouseDown;

    const fov = camera.projection.FOV;
    const z = 0.5 / Math.tan(fov * 0.5);
    const fovH = Math.atan2(this.renderer.aspectRatio * 0.5, z) * 2;

    const mm = this.mouseSpeed;

    if (mm.length() > StreetviewRotationController.MOUSE_MOVEMENT_CANCEL_TRIGGER) {
      // this.mouseListener.cancelClick();
    }

    if (this.mouseDown && mouseDownPreviousFrame) {
      const v = this._rotationSpeedMultiplier;
      this._eulerSpeed.setValues(v * mm.y * fov * this._mouseInverseValue, v * mm.x * fovH * this._mouseInverseValue);
      this._rotating = this._eulerSpeed.length() > 0;

      this._rotatingLerp = this._rotating ? 0.5 : 0;
      this._rotatingLerpPrev = this._rotatingLerp;

      this.rotateBy(this._eulerSpeed);
      this._eulerSpeed.setValues(
        Math.min(StreetviewRotationController.MAX_ROTATE_SPEED, Math.abs(this._eulerSpeed.x)) * Math.sign(this._eulerSpeed.x),
        Math.min(StreetviewRotationController.MAX_ROTATE_SPEED, Math.abs(this._eulerSpeed.y)) * Math.sign(this._eulerSpeed.y),
      );
    } else {
      if(this.easing > 0) {
        this._rotatingLerp = clamp01(this._rotatingLerp - this.renderer.time.dt / this.easing);
      }else{
        this._rotatingLerp = 0;
      }
      this._rotating = this._rotatingLerp > 0;
      if (this._rotating) {
        this._eulerSpeed.multiplyScalar(1 - (this._rotatingLerpPrev - this._rotatingLerp) / this._rotatingLerpPrev);
        this.rotateBy(this._eulerSpeed);
      }
      this._rotatingLerpPrev = this._rotatingLerp;
    }
  }

  public resetSpeed() {
    this.mouseDown = false;
    this._eulerSpeed.setValues(0, 0);
  }

  public rotateBy(rot: Vector2) {
    this._eulerXY.add(rot);

    if (this._clampEuler) {
      if(this._rotationLimitX.y - this._rotationLimitX.x < Math.PI) {
        this._eulerXY.x = Math.max(this._rotationLimitX.x, this._eulerXY.x);
        this._eulerXY.x = Math.min(this._rotationLimitX.y, this._eulerXY.x);
      }
      if(this._rotationLimitY.y - this._rotationLimitY.x < Math.PI * 2) {
        this._eulerXY.y = Math.max(this._rotationLimitY.x, this._eulerXY.y);
        this._eulerXY.y = Math.min(this._rotationLimitY.y, this._eulerXY.y);
      }
    }
    this._transform.eulerXY = this._eulerXY;
  }
}
