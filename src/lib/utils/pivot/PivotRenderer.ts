import Renderer from "../../renderer/render/Renderer";
import Camera from "../../renderer/camera/Camera";
import Transform from "../../renderer/scene/Transform";
import Quaternion from "../../renderer/math/Quaternion";
import Vector3 from "../../renderer/math/Vector3";
import VertexFragmentShader from "../../renderer/material/VertexFragmentShader";
import Mesh from "../../renderer/mesh/Mesh";
import Cone from "../mesh/geometries/Cone";
import Cylinder from "../mesh/geometries/Cylinder";
import Torus from "../mesh/geometries/Torus";
import Cube from "../mesh/geometries/Cube";
import { meshMerge } from "../mesh/MeshMerge";
import { PivotMode } from "./Pivot";
import pivot from "./pivot.wgsl?raw";

//of the rotation rings, in gizmo units
export const TORUS_THICKNESS = .01;
const SHAFT_RADIUS = .01;

const ALL_AXES: [boolean, boolean, boolean] = [true, true, true];

//the transforms of the parts of an axis along x, y and z: rotated from y (along which the cylinder and the cone are made)
//and moved to the offset along the axis
const getAxisTransforms = (offset: number): Transform[] => {
  const x = new Transform();
  x.rotateZ(Math.PI * -.5);
  x.translateX(offset);
  const y = new Transform();
  y.translateY(offset);
  const z = new Transform();
  z.rotateX(Math.PI * .5);
  z.translateZ(offset);
  return [x, y, z];
};

/*
Draws the gizmo of Pivot on top of the scene, in a pass of its own without depth. The meshes are merged with an index
attribute per part (see meshMerge): 0 - 2 are the shafts or the rings, 3 - 5 the cones or cubes, 6 the center cube. pivot.wgsl
colors them by axis and hides the parts of the axes that are masked out
 */
export default class PivotRenderer {
  private renderer: Renderer;
  private shader: VertexFragmentShader;
  private translationMesh: Mesh;
  private rotationMesh: Mesh;
  private scaleMesh: Mesh;
  //of the rings, around their y axis. For the hit test
  public readonly torusRotations: Quaternion[];
  private axisMask = new Vector3();

  constructor(renderer: Renderer) {
    this.renderer = renderer;

    this.shader = new VertexFragmentShader(this.renderer, 'pivot', pivot);
    this.shader.cullMode = 'none';
    this.shader.setAlphaBlending();

    const shaft = new Cylinder(this.renderer, SHAFT_RADIUS, SHAFT_RADIUS, 1, 8);
    const shafts = getAxisTransforms(.5);

    const cone = new Cone(this.renderer, 0.04, 0.08, 16, 1);
    this.translationMesh = meshMerge([shaft, shaft, shaft, cone, cone, cone], [...shafts, ...getAxisTransforms(1)], {addIndexAttribute: true});

    const cube = new Cube(this.renderer, 0.1);
    this.scaleMesh = meshMerge([shaft, shaft, shaft, cube, cube, cube, cube], [...shafts, ...getAxisTransforms(1), new Transform()], {addIndexAttribute: true});

    //the ring of an axis lies in the plane at right angles to it, the torus is made around y
    const torus = new Torus(this.renderer, 8, 64, 1, TORUS_THICKNESS);
    const rings = [new Transform(), new Transform(), new Transform()];
    rings[0].rotateZ(Math.PI * .5);
    rings[2].rotateX(Math.PI * .5);
    this.torusRotations = rings.map(ring => ring.rotation);
    this.rotationMesh = meshMerge([torus, torus, torus], rings, {addIndexAttribute: true});
  }

  //worldScale: the length of the axes in world units. scaleAxes: the axes the scale gizmo shows
  public draw(camera: Camera, target: Transform, worldScale: number, mode: PivotMode, scaleAxes = ALL_AXES): void {
    this.renderer.endPasses();
    this.renderer.startRenderPass(null, {colorClear: false, depthClear: true});

    const mask = mode === PivotMode.SCALE ? scaleAxes : ALL_AXES;
    this.axisMask.setValues(mask[0] ? 1 : 0, mask[1] ? 1 : 0, mask[2] ? 1 : 0);

    this.shader.setMatrix('viewProjection', camera.viewProjection);
    //the rotation only: the gizmo has the same shape for any scale of the target
    this.shader.setMatrix('rotation', target.worldRotationMatrixMat4);
    this.shader.setVector3('position', target.worldPosition);
    this.shader.setFloat('scale', worldScale);
    this.shader.setVector3('axisMask', this.axisMask);

    const mesh = mode === PivotMode.TRANSLATE ? this.translationMesh : mode === PivotMode.ROTATE ? this.rotationMesh : this.scaleMesh;
    this.renderer.draw(mesh, this.shader);
  }
}
