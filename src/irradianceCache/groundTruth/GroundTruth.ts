import ComputeShader from "monks-webgpu/renderer/material/ComputeShader";
import VertexFragmentShader from "monks-webgpu/renderer/material/VertexFragmentShader";
import Renderer from "monks-webgpu/renderer/render/Renderer";
import ParamGroup from "monks-webgpu/utils/params/ParamGroup";
import ParamBase from "monks-webgpu/utils/params/ParamBase";
import RenderTexture from "monks-webgpu/renderer/texture/RenderTexture";
import Camera from "monks-webgpu/renderer/camera/Camera";
import ParamLog from "monks-webgpu/utils/params/ParamLog";
import ParamInt from "monks-webgpu/utils/params/ParamInt";
import accumulateGroundTruth from './accumulateGroundTruth.wgsl?raw';
import resolveGroundTruth from './resolveGroundTruth.wgsl?raw';
import hash from 'monks-webgpu/shaders/hash.wgsl?raw';
import toneMap from 'monks-webgpu/shaders/toneMap.wgsl?raw';
import rayTraceScene from "../includes/rayTraceScene.wgsl?raw";
import pathTrace from "../includes/pathTrace.wgsl?raw";
import normalPacking from 'monks-webgpu/shaders/normalPacking.wgsl?raw';
import gBufferPacking from '../includes/gBuffer.wgsl?raw';
import cameraRay from '../includes/cameraRay.wgsl?raw';
import Scene from "../scene/Scene";
import r2 from 'monks-webgpu/shaders/r2.wgsl?raw';

/*
the reference the cached lighting of the other draw modes converges to: every pixel of the gBuffer gets sampleCount paths of
bounceCount bounces per frame, without a radiance cache, accumulated in screen space until the camera moves, a param changes
or the canvas resizes. It accumulates the color with the albedo, so with the TAA on the jittered camera anti aliases it:
the TAA is integrated, it has no TAA pass
 */
export default class GroundTruth {
  private renderer: Renderer;
  public accumulateShader: ComputeShader;
  private resolveMaterial: VertexFragmentShader;
  //per pixel: the sum of the lighting (rgb) and the sample count (a)
  private accumulationBuffer: GPUBuffer | null = null;
  private width = 0;
  private height = 0;
  private frameCount = 0;
  private paramChangeCount = -1;
  private readonly workGroupSizeX = 8;
  private readonly workGroupSizeY = 8;
  private frameCountLog!: ParamLog;
  private sampleCount!: ParamInt;

  public constructor(renderer: Renderer) {
    this.renderer = renderer;

    this.accumulateShader = new ComputeShader(this.renderer, 'accumulateGroundTruth', accumulateGroundTruth, {includes: [hash, rayTraceScene, pathTrace, r2, normalPacking, gBufferPacking, cameraRay]});
    this.accumulateShader.setConstant('WORK_GROUP_SIZE_X', this.workGroupSizeX);
    this.accumulateShader.setConstant('WORK_GROUP_SIZE_Y', this.workGroupSizeY);

    this.resolveMaterial = new VertexFragmentShader(this.renderer, 'resolveGroundTruth', resolveGroundTruth, {includes: [toneMap]});
    this.resolveMaterial.cullMode = 'none';
  }

  public get exposureMaterial() {
    return this.resolveMaterial;
  }

  //paths per pixel in the image of the last update
  public get accumulatedSamples(): number {
    return this.frameCount * this.sampleCount.value;
  }

  public init(paramGroup:ParamGroup, scene:Scene) {
    scene.bindSkyColor(this.accumulateShader);

    const params = { materials: [this.accumulateShader] };
    const pg = paramGroup.addGroup('groundTruth');
    //paths per pixel per frame
    this.sampleCount = pg.addParamInt('sampleCount', 1, 1, 64, params);
    //its own bounce limit, the one of the rayTrace group is for the cached paths of the other draw modes
    pg.addParamInt('bounceCount', 4, 1, 16, params);
    this.frameCountLog = pg.addLog('frames: 0');

    scene.bindPathTrace(this.accumulateShader);
  }

  public update(gBuffer:RenderTexture, camera:Camera) {
    const resized = gBuffer.width != this.width || gBuffer.height != this.height;
    if(resized){
      if(this.accumulationBuffer) this.renderer.destroyAfterSubmit(this.accumulationBuffer);
      this.width = gBuffer.width;
      this.height = gBuffer.height;
      this.accumulationBuffer = this.renderer.createBuffer(this.width * this.height * 16, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
      this.accumulateShader.setBuffer('accumulationBuffer', this.accumulationBuffer);
      this.resolveMaterial.setBuffer('accumulationBuffer', this.accumulationBuffer);
    }

    //the lighting params change the result too. The log is no input, see ParamLog
    const paramsChanged = ParamBase.changeCount != this.paramChangeCount;
    if(resized || paramsChanged || camera.changed){
      this.renderer.clearBuffer(this.accumulationBuffer!);
      this.frameCount = 0;
    }
    this.frameCount++;
    this.frameCountLog.setValue(`frames: ${this.frameCount}`);
    this.paramChangeCount = ParamBase.changeCount;

    this.accumulateShader.setVector3('cameraPosition', camera.worldPosition);
    this.accumulateShader.setMatrix('inverseView', camera.viewMatrix.inverse);
    this.accumulateShader.setMatrix('inverseProjection', camera.projectionMatrix.inverse);
    this.accumulateShader.setFloat('frame', this.renderer.time.frameNumber);
    this.accumulateShader.setVector2('resolution', gBuffer.size);
    this.accumulateShader.setTexture('normalDepthTexture', gBuffer.views[1]);
    this.accumulateShader.setTexture('albedoTexture', gBuffer.views[0]);

    const countX = Math.ceil(gBuffer.width / this.workGroupSizeX);
    const countY = Math.ceil(gBuffer.height / this.workGroupSizeY);
    this.renderer.dispatchCompute(this.accumulateShader, countX, countY, 1);
  }

  public draw(gBuffer:RenderTexture, destination:RenderTexture | null = null) {
    this.resolveMaterial.setVector2('resolution', gBuffer.size);
    this.renderer.blit(null, destination, this.resolveMaterial);
  }
}
