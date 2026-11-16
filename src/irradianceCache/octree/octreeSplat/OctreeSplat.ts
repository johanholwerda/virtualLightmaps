import Renderer from "monks-webgpu/renderer/render/Renderer";
import VertexFragmentShader from "monks-webgpu/renderer/material/VertexFragmentShader";
import octreeSplat from './octreeSplat.wgsl?raw';
import ParamGroup from "monks-webgpu/utils/params/ParamGroup";
import Camera from "monks-webgpu/renderer/camera/Camera";
import RenderTexture from "monks-webgpu/renderer/texture/RenderTexture";
import OctreeUpdater from "../octreeUpdater/OctreeUpdater";
import octreeNode from '../../includes/octreeNode.wgsl?raw';
import normalize from './normalize.wgsl?raw';
import toneMap from 'monks-webgpu/shaders/toneMap.wgsl?raw';
import Vector3 from "monks-webgpu/renderer/math/Vector3";
import normalPacking from 'monks-webgpu/shaders/normalPacking.wgsl?raw';
import gBufferPacking from '../../includes/gBuffer.wgsl?raw';
import cameraRay from '../../includes/cameraRay.wgsl?raw';
import Scene from "../../scene/Scene";

export default class OctreeSplat {
  private renderer: Renderer;
  private material: VertexFragmentShader;
  private indexBuffer!: GPUBuffer;
  private indexCount:number;
  public renderTexture:RenderTexture;
  private normalize:VertexFragmentShader;

  public constructor(renderer: Renderer, pointCount: number, octreeDepthMax: number, octreeRootNodeSize:number ) {
    this.renderer = renderer;

    this.material = new VertexFragmentShader(this.renderer, 'octreeSplat', octreeSplat, {includes: [octreeNode, normalPacking, gBufferPacking, cameraRay]});
    this.material.setAdditiveBlending();
    this.material.depthWriteEnabled = false;
    this.material.cullMode = 'none';
    this.material.setConstant('OCTREE_ROOT_NODE_SIZE', octreeRootNodeSize);
    this.material.setConstant('OCTREE_DEPTH_MAX', octreeDepthMax);

    this.indexBuffer?.destroy();

    this.indexCount = pointCount * 6;

    const indexData = new Uint32Array(this.indexCount);
    let ii = 0;
    for(let i = 0; i < pointCount; i++){
      const base = i * 4;
      indexData[ii++] = base;
      indexData[ii++] = base + 1;
      indexData[ii++] = base + 2;
      indexData[ii++] = base + 2;
      indexData[ii++] = base + 1;
      indexData[ii++] = base + 3;
    }

    this.indexBuffer = this.renderer.createBuffer(this.indexCount * 4, GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST);
    this.renderer.writeBuffer(this.indexBuffer, indexData);

    this.renderTexture = new RenderTexture(this.renderer, 1, 1, {scaleToCanvas: true, depth: false, formats: ['rgba16float']});

    this.normalize = new VertexFragmentShader(renderer, 'normalize', normalize, {includes: [toneMap, octreeNode, normalPacking, gBufferPacking, cameraRay]});
    this.normalize.setConstant('OCTREE_DEPTH_MAX', octreeDepthMax);
    this.normalize.setConstant('OCTREE_ROOT_NODE_SIZE', octreeRootNodeSize);
  }

  public get exposureMaterial() {
    return this.normalize;
  }

  public init(paramGroup:ParamGroup, scene:Scene) {
    scene.bindSkyColor(this.normalize);

    const pgParams = { materials: [this.material] };
    const pg = paramGroup.addGroup('octreeSplat');
    pg.addParamFloat('pointSize', 2.5, 0, 5, pgParams);
    pg.addParamBool('solidDisc', false, pgParams);
    //leaves that are too coarse for their distance are stale and their discs spread over other surfaces, see isTooCoarse in octreeNode.wgsl
    pg.addParamBool('skipCoarseLeaves', true, pgParams);
    //the distance from the plane of a disc at which its weight is 0, as a fraction of the disc radius
    pg.addParamFloat('planeDistance', .25, 0, 1, pgParams);
    // pg.addShaderParamFloat('distanceMax', .1);

    const pgNormalizeParams = { materials: [this.normalize] };
    const pgNormalize = paramGroup.addGroup('normalize');
    pgNormalize.addParamBool('showDensity', false, pgNormalizeParams);
    //the splat weight below which the lighting of the nearest ancestor node fades in, to fill the holes
    pgNormalize.addParamFloat('fallbackWeight', .25, 0, 2, pgNormalizeParams);
    //the levels above the leaf the fallback may come from. Higher ancestors span other surfaces facing the same way and have
    //stale lighting: bright specks in shadowed corners. See getFallbackLighting in octreeNode.wgsl
    pgNormalize.addParamInt('fallbackLevels', 2, 0, 12, pgNormalizeParams);
    //the pixels no splat covers in red, without the fallback
    pgNormalize.addParamBool('showHoles', false, pgNormalizeParams);
    //how the pixels that the splats don't light fully got their lighting: magenta a near ancestor, yellow the neighbouring leaves
    //(no near ancestor faces that way), cyan the little splat weight there is (no neighbouring leaf either), red a constant
    pgNormalize.addParamBool('showFill', false, pgNormalizeParams);
  }

  /*
  TODO: draw only visible leaf nodes, draw indirect. Tested: the vertex shader takes about 1 ms with 2^22 nodes
   */
  public draw(camera:Camera, octreeUpdater:OctreeUpdater, gBuffer:RenderTexture, destination:RenderTexture| null = null) {
    this.renderer.startRenderPass(this.renderTexture);
    this.material.setMatrix('inverseView', camera.viewMatrix.inverse);
    this.material.setMatrix('inverseProjection', camera.projectionMatrix.inverse);
    this.material.setTexture('normalDepthTexture', gBuffer.views[1]);
    this.material.setMatrix('viewProjection', camera.viewProjection);
    this.material.setVector3('cameraPosition', camera.worldPosition);
    this.material.setFloat('scale', octreeUpdater.scale.value);
    this.material.setVector2('resolution', this.renderer.size);
    this.material.setFloat('aspectRatio', this.renderer.aspectRatio);
    this.material.setBuffer('octreeBuffer', octreeUpdater.octreeBuffer);
    this.renderer.drawVertices(this.indexCount, this.material, this.indexBuffer);
    this.normalize.setTexture('albedoTexture', gBuffer.views[0]);
    this.normalize.setTexture('normalDepthTexture', gBuffer.views[1]);
    this.normalize.setBuffer('octreeBuffer', octreeUpdater.octreeBuffer);
    this.normalize.setVector3('cameraPosition', camera.worldPosition);
    this.normalize.setMatrix('inverseView', camera.viewMatrix.inverse);
    this.normalize.setMatrix('inverseProjection', camera.projectionMatrix.inverse);
    this.normalize.setVector2('resolution', gBuffer.size);
    this.renderer.blit(this.renderTexture, destination, this.normalize);
  }

}
