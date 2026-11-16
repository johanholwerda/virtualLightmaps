import Renderer from "webGPU-framework/renderer/render/Renderer";
import VertexFragmentShader from "webGPU-framework/renderer/material/VertexFragmentShader";
import ComputeShader from "webGPU-framework/renderer/material/ComputeShader";
import ParamGroup from "webGPU-framework/utils/params/ParamGroup";
import Camera from "webGPU-framework/renderer/camera/Camera";
import RenderTexture from "webGPU-framework/renderer/texture/RenderTexture";
import OctreeUpdater from "../octreeUpdater/OctreeUpdater";
import octreeDebug from './octreeDebug.wgsl?raw';
import octreeDebugInstances from './octreeDebugInstances.wgsl?raw';
import octreeNode from '../../includes/octreeNode.wgsl?raw';
import normalPacking from 'webGPU-framework/shaders/normalPacking.wgsl?raw';
import gBufferPacking from '../../includes/gBuffer.wgsl?raw';

/*
The octree nodes as the 1 pixel edges of their boxes, in the color of the node. Red for the nodes without lighting.
A compute pass lists the nodes to draw (see octreeDebugInstances.wgsl), then one indirect draw of 24 line vertices per node
 */
export default class OctreeDebug {
  private renderer: Renderer;
  private material: VertexFragmentShader;
  private instancesShader: ComputeShader;
  private instancesBuffer: GPUBuffer;
  private drawArgsBuffer: GPUBuffer;
  private readonly workGroupSize = 256;
  private readonly nodeCountMax: number;

  public constructor(renderer: Renderer, nodeCountMax: number, octreeDepthMax: number, octreeRootNodeSize: number) {
    this.renderer = renderer;
    this.nodeCountMax = nodeCountMax;

    //node index, parent index
    this.instancesBuffer = this.renderer.createBuffer(nodeCountMax * 8, GPUBufferUsage.STORAGE);
    this.drawArgsBuffer = this.renderer.createBuffer(4 * 4, GPUBufferUsage.INDIRECT | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);

    this.instancesShader = new ComputeShader(this.renderer, 'octreeDebugInstances', octreeDebugInstances, {includes: [octreeNode]});
    this.instancesShader.setConstant('WORK_GROUP_SIZE', this.workGroupSize);
    this.instancesShader.setConstant('OCTREE_ROOT_NODE_SIZE', octreeRootNodeSize);
    this.instancesShader.setConstant('OCTREE_DEPTH_MAX', octreeDepthMax);
    this.instancesShader.setBuffer('instances', this.instancesBuffer);
    this.instancesShader.setBuffer('drawArgs', this.drawArgsBuffer);

    this.material = new VertexFragmentShader(this.renderer, 'octreeDebug', octreeDebug, {includes: [octreeNode, normalPacking, gBufferPacking]});
    this.material.topology = 'line-list';
    this.material.cullMode = 'none';
    this.material.depthWriteEnabled = false;
    //the depth test is against the g-buffer distance, in the fragment shader
    this.material.depthCompare = 'always';
    this.material.setConstant('OCTREE_ROOT_NODE_SIZE', octreeRootNodeSize);
    this.material.setConstant('OCTREE_DEPTH_MAX', octreeDepthMax);
    this.material.setBuffer('instances', this.instancesBuffer);
  }

  public init(paramGroup: ParamGroup) {
    const params = { materials: [this.material, this.instancesShader] };
    const pg = paramGroup.addGroup('octreeDebug');
    pg.addParamBool('leavesOnly', false, params);
    pg.addParamBool('hideEmpty', true, params);
    pg.addParamFloat('brightness', 1, 0, 10, params);
    pg.addParamBool('depthTest', true, params);
    pg.addParamFloat('depthBias', .01, 0, .1, params);
  }

  public draw(camera: Camera, octreeUpdater: OctreeUpdater, gBuffer: RenderTexture, destination: RenderTexture | null = null) {
    this.renderer.clearBuffer(this.drawArgsBuffer);
    this.instancesShader.setBuffer('octreeBuffer', octreeUpdater.octreeBuffer);
    this.renderer.dispatchCompute(this.instancesShader, Math.ceil(this.nodeCountMax / this.workGroupSize), 1, 1);

    this.renderer.startRenderPass(destination);
    this.material.setMatrix('viewProjection', camera.viewProjection);
    this.material.setVector3('cameraPosition', camera.worldPosition);
    this.material.setTexture('normalDepthTexture', gBuffer.views[1]);
    this.material.setBuffer('octreeBuffer', octreeUpdater.octreeBuffer);
    this.renderer.drawVerticesIndirect(this.material, null, this.drawArgsBuffer);
  }
}
