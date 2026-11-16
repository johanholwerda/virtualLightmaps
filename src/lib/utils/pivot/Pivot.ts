import Renderer from "../../renderer/render/Renderer";
import Camera from "../../renderer/camera/Camera";
import ProjectionOrthographic from "../../renderer/camera/ProjectionOrthographic";
import Vector3 from "../../renderer/math/Vector3";
import Vector2 from "../../renderer/math/Vector2";
import Ray from "../../renderer/math/Ray";
import Quaternion from "../../renderer/math/Quaternion";
import Matrix3x3 from "../../renderer/math/Matrix3x3";
import Transform from "../../renderer/scene/Transform";
import { intersectRaySphereNoScale, intersectRayTorus } from "../../renderer/math/Intersection";
import { clamp } from "../../renderer/utils/MathUtils";
import ParamGroup from "../params/ParamGroup";
import KeyListener from "../input/KeyListener";
import Key, { KEY_ALT, KEY_E, KEY_R, KEY_W } from "../input/Key";
import PivotRenderer, { TORUS_THICKNESS } from "./PivotRenderer";

export enum PivotMode {
  TRANSLATE = 'translate',
  ROTATE = 'rotate',
  SCALE = 'scale'
}

//the handle that is dragged. ALL is the center cube of the scale gizmo
export enum PivotAxis {
  NONE = -1,
  X = 0,
  Y = 1,
  Z = 2,
  ALL = 3
}

const AXES = [PivotAxis.X, PivotAxis.Y, PivotAxis.Z] as const;
//the radius of the hit spheres around the translation cones and the scale cubes, in gizmo units
const HANDLE_HIT_RADIUS = .08;
//the hit tube around the rotation rings is wider than the drawn ring
const TORUS_HIT_THICKNESS = TORUS_THICKNESS * 8;
//the gizmo is this many pixels long on screen
const GIZMO_PIXEL_SIZE = 100;

export type PivotOptions = {
  //the modes that can be selected, the first is the start mode
  modes: PivotMode[];
  //W, E and R select translate, rotate and scale
  keyboardShortcuts: boolean;
};

const defaultPivotOptions: PivotOptions = {
  modes: [PivotMode.TRANSLATE, PivotMode.ROTATE, PivotMode.SCALE],
  keyboardShortcuts: true,
};

const isTextInputFocused = () => {
  const active = document.activeElement;
  return !!(active && (active.tagName === 'INPUT' || active.tagName === 'TEXTAREA'));
};

/*
A translate / rotate / scale gizmo on a Transform, along the axes of its world rotation. It is drawn at a fixed size on
screen. Translation and rotation are applied to the target, then onTranslate / onRotate are called. Scaling depends on
what the target is, so it is left to onScale: the axis that is dragged and the factor to multiply its scale with.
Not active while ALT is held, for cameras that orbit with ALT and the left mouse button
 */
export default class Pivot {
  private renderer: Renderer;
  private pivotRenderer: PivotRenderer;
  private readonly modes: PivotMode[];
  private _mode: PivotMode;
  private dragAxis: PivotAxis = PivotAxis.NONE;
  private lastMousePosition = new Vector2();
  private ray = new Ray();
  //in world space, the direction a ring moves under the mouse where it was grabbed
  private rotationTangent = new Vector3(1, 0, 0);
  private rotationSensitivity = 20;
  private scaleSensitivity = 25;

  public onTranslate: ((target: Transform) => void) | null = null;
  public onRotate: ((target: Transform) => void) | null = null;
  public onScale: ((target: Transform, axis: PivotAxis, multiplier: number) => void) | null = null;
  //the scale handles shown for the target, all by default
  public getScaleAxes: ((target: Transform) => [x: boolean, y: boolean, z: boolean]) | null = null;

  constructor(renderer: Renderer, options: Partial<PivotOptions> = {}) {
    this.renderer = renderer;
    const {modes, keyboardShortcuts} = {...defaultPivotOptions, ...options};
    this.modes = modes;
    this._mode = modes[0];
    this.pivotRenderer = new PivotRenderer(renderer);

    if (keyboardShortcuts) {
      const keys: [Key, PivotMode][] = [[KEY_W, PivotMode.TRANSLATE], [KEY_E, PivotMode.ROTATE], [KEY_R, PivotMode.SCALE]];
      for (const [key, mode] of keys) {
        if (!modes.includes(mode)) continue;
        KeyListener.addKeyHitEvent(key, () => {
          if (!isTextInputFocused()) this.mode = mode;
        });
      }
    }
  }

  //buttons for the modes, when there is more than one
  public init(paramGroup: ParamGroup) {
    if (this.modes.length < 2) return;
    const pg = paramGroup.addGroup('pivot');
    const shortcuts = {[PivotMode.TRANSLATE]: 'W', [PivotMode.ROTATE]: 'E', [PivotMode.SCALE]: 'R'};
    for (const mode of this.modes) pg.addButton(`${mode} (${shortcuts[mode]})`, { callback: () => this.mode = mode });
  }

  public get mode(): PivotMode {
    return this._mode;
  }

  public set mode(mode: PivotMode) {
    if (this._mode === mode || !this.modes.includes(mode)) return;
    this._mode = mode;
    this.dragAxis = PivotAxis.NONE;
  }

  //true while a handle is dragged, the camera should not take the mouse input then
  public get dragging(): boolean {
    return this.dragAxis !== PivotAxis.NONE;
  }

  //the handle of the gizmo of the target under the mouse, NONE if none. For a click handler that selects objects, so
  //a click on a handle doesn't select what is behind it
  public hitTest(camera: Camera, target: Transform | null): PivotAxis {
    if (!target) return PivotAxis.NONE;
    camera.getRay(this.renderer.mouseListener.screenPosition, this.ray);
    return this.mode === PivotMode.ROTATE ? this.hitRings(target, camera) : this.hitHandles(target, camera);
  }

  //before the camera update, so the camera can check dragging on the frame the drag starts
  public update(camera: Camera, target: Transform | null): void {
    const ml = this.renderer.mouseListener;
    if (!target || KeyListener.getKeyCodeDown(KEY_ALT) || !ml.mouseDown) {
      this.dragAxis = PivotAxis.NONE;
      return;
    }

    const mp = ml.screenPosition;
    if (ml.mouseDownThisFrame) {
      this.dragAxis = this.hitTest(camera, target);
      this.lastMousePosition.copy(mp);
      return;
    }
    if (this.dragAxis === PivotAxis.NONE) return;

    const mouseDelta = Vector2.subtract(mp, this.lastMousePosition);
    this.lastMousePosition.copy(mp);
    if (mouseDelta.x === 0 && mouseDelta.y === 0) return;

    switch (this.mode) {
      case PivotMode.TRANSLATE:
        this.translate(target, mouseDelta, camera);
        break;
      case PivotMode.ROTATE:
        this.rotate(target, mouseDelta, camera);
        break;
      case PivotMode.SCALE:
        this.scale(target, mouseDelta, camera);
        break;
    }
  }

  public draw(camera: Camera, target: Transform | null): void {
    if (!target) return;
    const scaleAxes = this.mode === PivotMode.SCALE ? this.getScaleAxes?.(target) : undefined;
    this.pivotRenderer.draw(camera, target, this.getWorldScale(camera, target), this.mode, scaleAxes);
  }

  //the x, y or z axis of the rotation: the column of its matrix
  private getAxis(axis: PivotAxis, rotation: Quaternion): Vector3 {
    const m = Matrix3x3.fromQuat(rotation).m;
    return new Vector3(m[axis * 3], m[axis * 3 + 1], m[axis * 3 + 2]);
  }

  //the world length of the gizmo, so it has the same size on screen at any distance. By the canvas: renderer.height is
  //that of the render target bound at the time, so the draw and the hit test could get different sizes
  private getWorldScale(camera: Camera, target: Transform): number {
    const projection = camera.projection;
    if (projection instanceof ProjectionOrthographic) {
      const orthoHeight = projection.topRight.y - projection.bottomLeft.y;
      return orthoHeight * GIZMO_PIXEL_SIZE / this.renderer.canvas.height;
    }
    const depth = Vector3.dot(Vector3.subtract(target.worldPosition, camera.worldPosition), camera.worldForward);
    return Math.abs(depth) * projection.frustumHeight * 2 * GIZMO_PIXEL_SIZE / this.renderer.canvas.height;
  }

  //the translation cones or the scale cubes
  private hitHandles(target: Transform, camera: Camera): PivotAxis {
    const worldScale = this.getWorldScale(camera, target);
    const hitRadius = HANDLE_HIT_RADIUS * worldScale;
    const position = target.worldPosition;
    const rotation = target.worldRotation;
    const scaleAxes = this.mode === PivotMode.SCALE ? this.getScaleAxes?.(target) : undefined;

    let closest = Infinity;
    let hit = PivotAxis.NONE;
    for (const axis of AXES) {
      if (scaleAxes && !scaleAxes[axis]) continue;
      const handle = this.getAxis(axis, rotation).multiplyScalar(worldScale).add(position);
      const distance = intersectRaySphereNoScale(this.ray, handle, hitRadius);
      if (distance > 0 && distance < closest) {
        closest = distance;
        hit = axis;
      }
    }
    if (this.mode === PivotMode.SCALE) {
      const distance = intersectRaySphereNoScale(this.ray, position, hitRadius);
      if (distance > 0 && distance < closest) hit = PivotAxis.ALL;
    }
    return hit;
  }

  //the closest rotation ring, sets rotationTangent for the drag
  private hitRings(target: Transform, camera: Camera): PivotAxis {
    const worldScale = this.getWorldScale(camera, target);
    const position = target.worldPosition;
    const rotation = target.worldRotation;
    let closest = Infinity;
    let hit = PivotAxis.NONE;
    for (const axis of AXES) {
      const ring = {
        worldPosition: position,
        worldRotation: Quaternion.multiply(rotation, this.pivotRenderer.torusRotations[axis]),
        worldScaleScalar: 1
      };
      const distance = intersectRayTorus(this.ray, worldScale, worldScale * TORUS_HIT_THICKNESS, ring);
      if (distance > 0 && distance < closest) {
        closest = distance;
        hit = axis;
      }
    }
    if (hit !== PivotAxis.NONE) {
      const hitPoint = Vector3.add(this.ray.origin, Vector3.multiplyScalar(this.ray.direction, closest));
      const radial = Vector3.subtract(hitPoint, position).normalize();
      this.rotationTangent = Vector3.cross(this.getAxis(hit, rotation), radial).normalize();
    }
    return hit;
  }

  //the mouse movement on the plane facing the camera at depth 1
  private getWorldMouseDelta(mouseDelta: Vector2, camera: Camera): Vector3 {
    const scale = camera.projection.frustumHeight * .5;
    return Vector3.add(
      Vector3.multiplyScalar(camera.worldRight, mouseDelta.x * scale * camera.projection.aspectRatio),
      Vector3.multiplyScalar(camera.worldUp, mouseDelta.y * scale)
    );
  }

  //along the axis, by the mouse movement at the depth of the target
  private translate(target: Transform, mouseDelta: Vector2, camera: Camera): void {
    const position = target.worldPosition;
    const depth = Vector3.dot(Vector3.subtract(position, camera.worldPosition), camera.worldForward);
    const worldDelta = this.getWorldMouseDelta(mouseDelta, camera).multiplyScalar(depth);
    const axis = this.getAxis(this.dragAxis, target.worldRotation);
    //the camera right and up are negated, see Camera.worldRight
    const distance = -Vector3.dot(worldDelta, axis);
    target.worldPosition = position.add(axis.multiplyScalar(distance));
    this.onTranslate?.(target);
  }

  private rotate(target: Transform, mouseDelta: Vector2, camera: Camera): void {
    const angle = Vector3.dot(this.getWorldMouseDelta(mouseDelta, camera), this.rotationTangent) * this.rotationSensitivity;
    //rotateAxisAngle takes the axis in the space of the parent, where the own axes of the target are those of its rotation
    target.rotateAxisAngle(this.getAxis(this.dragAxis, target.rotation), angle);
    this.onRotate?.(target);
  }

  private scale(target: Transform, mouseDelta: Vector2, camera: Camera): void {
    let delta: number;
    if (this.dragAxis === PivotAxis.ALL) {
      delta = (mouseDelta.x + mouseDelta.y) * .5;
    } else {
      const axis = this.getAxis(this.dragAxis, target.worldRotation);
      delta = Vector3.dot(this.getWorldMouseDelta(mouseDelta, camera), axis);
    }
    this.onScale?.(target, this.dragAxis, clamp(1 + delta * this.scaleSensitivity, .5, 1.5));
  }
}
