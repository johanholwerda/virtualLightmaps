import Renderer from "webGPU-framework/renderer/render/Renderer";
import Camera from "webGPU-framework/renderer/camera/Camera";
import Ray from "webGPU-framework/renderer/math/Ray";
import Vector3 from "webGPU-framework/renderer/math/Vector3";
import ParamBase from "webGPU-framework/utils/params/ParamBase";
import Pivot, { PivotAxis, PivotMode } from "webGPU-framework/utils/pivot/Pivot";
import Scene, { Model } from "./Scene";

/*
A click on a model selects it and shows a pivot on it, a click on the sky deselects. Dragging an axis of the pivot moves the
model: Scene.moveModel updates sceneBuffer and refits the BVH, which is made again after the drag. While a model is dragged the scene is dynamic, like during the day
cycle (see Scene.lightChanging), and groundTruth starts again every frame
 */
export default class SceneEditor {
  private renderer: Renderer;
  private scene: Scene;
  private camera: Camera;
  private pivot: Pivot;
  private selected: Model | null = null;
  private ray = new Ray();
  private lastPosition = new Vector3();

  constructor(renderer: Renderer, scene: Scene, camera: Camera) {
    this.renderer = renderer;
    this.scene = scene;
    this.camera = camera;
    //no keyboard shortcuts: W, E and R would collide with the camera keys
    this.pivot = new Pivot(renderer, {modes: [PivotMode.TRANSLATE], keyboardShortcuts: false});
  }

  public init() {
    this.renderer.mouseListener.addClickEvent(() => this.select());
  }

  //true while an axis of the pivot is dragged: the camera should not rotate, and the lighting changes
  public get dragging(): boolean {
    return this.pivot.dragging;
  }

  private select() {
    //a click on a handle of the pivot is not a click on what is behind it
    const transform = this.selected?.transform ?? null;
    if (this.pivot.hitTest(this.camera, transform) !== PivotAxis.NONE) return;
    this.camera.getRay(this.renderer.mouseListener.screenPosition, this.ray);
    this.selected = this.scene.pick(this.ray);
  }

  //before the camera update, so the camera knows about a drag that starts this frame
  public update() {
    if (!this.selected) {
      this.scene.movingModel = null;
      return;
    }
    const transform = this.selected.transform;
    this.lastPosition.copy(transform.position);
    this.pivot.update(this.camera, transform);
    this.scene.movingModel = this.pivot.dragging ? this.selected : null;
    if (!transform.position.equals(this.lastPosition)) {
      this.scene.moveModel(this.selected);
      //groundTruth converges for a fixed scene only, see GroundTruth.update
      ParamBase.changeCount++;
    }
    //the BVH is only refitted during the drag
    if (!this.pivot.dragging) this.scene.rebuildBVH();
  }

  //on top of the final image
  public draw() {
    if (this.selected) this.pivot.draw(this.camera, this.selected.transform);
  }
}
