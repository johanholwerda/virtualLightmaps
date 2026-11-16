import Renderer from "monks-webgpu/renderer/render/Renderer";
import VertexFragmentShader from "monks-webgpu/renderer/material/VertexFragmentShader";
import octreeRayMarch from './octreeRayMarch.wgsl?raw';
import ParamGroup from "monks-webgpu/utils/params/ParamGroup";
import Camera from "monks-webgpu/renderer/camera/Camera";
import RenderTexture from "monks-webgpu/renderer/texture/RenderTexture";
import Cube from "monks-webgpu/utils/mesh/geometries/Cube";
import OctreeUpdater from "../octreeUpdater/OctreeUpdater";
import octreeRayMarchShared from '../../includes/octreeRayMarch.wgsl?raw';
import octreeNode from '../../includes/octreeNode.wgsl?raw';
import toneMap from 'monks-webgpu/shaders/toneMap.wgsl?raw';
import normalPacking from 'monks-webgpu/shaders/normalPacking.wgsl?raw';
import gBufferPacking from '../../includes/gBuffer.wgsl?raw';
import Vector3 from "monks-webgpu/renderer/math/Vector3";
import Scene from "../../scene/Scene";

export default class OctreeRayMarch {
  private renderer: Renderer;
  private material: VertexFragmentShader;
  private mesh: Cube;

  public constructor(renderer: Renderer, octreeDepthMax: number, octreeRootNodeSize: number) {
    this.renderer = renderer;

    this.material = new VertexFragmentShader(this.renderer, 'octreeRayMarch', octreeRayMarch, {includes: [octreeRayMarchShared, octreeNode, toneMap, normalPacking, gBufferPacking]});
    this.material.cullMode = 'none';
    this.material.setConstant('OCTREE_DEPTH_MAX', octreeDepthMax);
    this.material.setConstant('OCTREE_ROOT_NODE_SIZE', octreeRootNodeSize);

    this.mesh = new Cube(this.renderer);
  }

  public get exposureMaterial() {
    return this.material;
  }

  public init(paramGroup:ParamGroup, scene:Scene) {
    scene.bindSkyColor(this.material);

    const params = { materials: [this.material] };
    const pg = paramGroup.addGroup('octreeRayMarch');
    //leaves that are too coarse for their distance are stale boxes larger than their geometry, see isTooCoarse in octreeNode.wgsl
    pg.addParamBool('skipCoarseLeaves', true, params);
    //the levels above the leaf the fallback may come from, see getFallbackLighting in octreeNode.wgsl
    pg.addParamInt('fallbackLevels', 2, 0, 12, params);
  }

  public draw(camera:Camera, octree:OctreeUpdater, gBuffer:RenderTexture, destination:RenderTexture| null = null) {
    this.material.setTexture('albedoTexture', gBuffer.views[0]);
    this.material.setTexture('normalDepthTexture', gBuffer.views[1]);
    this.renderer.startRenderPass(destination, { colorClear: false, depthClear: true });
    this.material.setBuffer('octreeBuffer', octree.octreeBuffer);
    this.material.setMatrix('viewProjection', camera.viewProjection);
    this.material.setVector3('cameraPosition', camera.worldPosition);
    this.material.setFloat('scale', octree.scale.value);
    this.renderer.draw(this.mesh, this.material);
  }

}
