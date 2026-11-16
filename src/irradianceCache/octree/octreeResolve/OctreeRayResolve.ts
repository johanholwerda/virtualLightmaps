import Renderer from "webGPU-framework/renderer/render/Renderer";
import VertexFragmentShader from "webGPU-framework/renderer/material/VertexFragmentShader";
import octreeResolve from './octreeResolve.wgsl?raw';
import ParamGroup from "webGPU-framework/utils/params/ParamGroup";
import Camera from "webGPU-framework/renderer/camera/Camera";
import RenderTexture from "webGPU-framework/renderer/texture/RenderTexture";
import OctreeUpdater from "../octreeUpdater/OctreeUpdater";
import octreeNode from '../../includes/octreeNode.wgsl?raw';
import toneMap from 'webGPU-framework/shaders/toneMap.wgsl?raw';
import Vector3 from "webGPU-framework/renderer/math/Vector3";
import normalPacking from 'webGPU-framework/shaders/normalPacking.wgsl?raw';
import gBufferPacking from '../../includes/gBuffer.wgsl?raw';
import cameraRay from '../../includes/cameraRay.wgsl?raw';
import Scene from "../../scene/Scene";

/*
3 * 3 * 3 samples gives a good result but is too slow
how slow would it be to fill a screen space frustum with the octree?
let's say the res is the same as the octree generation res (cell size 16)
@4k: 256 * 128 * 128 = 4M samples

currently in screen space: 8M * 27 = 216M octree samples

or: draw the leaf nodes as point sprites with the size of the nodes
that could cause a lot of overdraw at grazing angles but would have no aliasing
-big plus: variable frequency lighting
splat them orientated around their normal?
basically surfels in an octree
why the octree? as a stable acceleration structure for the lighting pass

volume can be a source of aliasing

can it be linearly interpolated? We don't want surfaces with different normal to blend

-problem: the regular screen-space grid can have a resolution that is too low for the frequency of the objects. Lighting would become muddy
 -tested: it was, and flickered because many nodes map to 1 voxel. The pixels now gather the nearest leaf nodes from the octree itself
  with the weights of octreeSplat (see octreeResolve.wgsl)
 */

export default class OctreeRayResolve {
  private renderer: Renderer;
  private material: VertexFragmentShader;

  public constructor(renderer: Renderer, octreeDepthMax: number, octreeRootNodeSize: number) {
    this.renderer = renderer;

    this.material = new VertexFragmentShader(this.renderer, 'octreeResolve', octreeResolve, {includes: [octreeNode, toneMap, normalPacking, gBufferPacking, cameraRay]});
    this.material.cullMode = 'none';
    this.material.setConstant('OCTREE_DEPTH_MAX', octreeDepthMax);
    this.material.setConstant('OCTREE_ROOT_NODE_SIZE', octreeRootNodeSize);
  }

  public get exposureMaterial() {
    return this.material;
  }

  public init(paramGroup:ParamGroup, scene:Scene) {
    scene.bindSkyColor(this.material);
    const params = { materials: [this.material] };
    const pg = paramGroup.addGroup('OctreeResolve');
    //in node sizes, like the pointSize of octreeSplat (which is the diameter)
    pg.addParamFloat('radius', 1.25, 0, 2, params);
    //the weight below which the lighting of the nearest ancestor node fades in, to fill the holes
    pg.addParamFloat('fallbackWeight', .25, 0, 2, params);
    //the levels above the leaf the fallback may come from. Higher ancestors span other surfaces facing the same way and have
    //stale lighting: bright specks in shadowed corners. See getFallbackLighting in octreeNode.wgsl
    pg.addParamInt('fallbackLevels', 2, 0, 12, params);
    //the pixels no node covers in red, without the fallback
    pg.addParamBool('showHoles', false, params);
  }

  public draw(camera:Camera, gBuffer:RenderTexture, octree:OctreeUpdater, destination:RenderTexture| null = null) {
    this.material.setBuffer('octreeBuffer', octree.octreeBuffer);
    this.material.setVector3('cameraPosition', camera.worldPosition);
    this.material.setMatrix('inverseView', camera.viewMatrix.inverse);
    this.material.setMatrix('inverseProjection', camera.projectionMatrix.inverse);
    this.material.setVector2('resolution', gBuffer.size);
    this.material.setTexture('normalDepthTexture', gBuffer.views[1]);
    this.material.setTexture('albedoTexture', gBuffer.views[0]);
    this.renderer.blit(null, destination, this.material);
  }

}
