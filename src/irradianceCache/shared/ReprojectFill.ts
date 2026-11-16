import RenderTexture from "monks-webgpu/renderer/texture/RenderTexture";
import Renderer from "monks-webgpu/renderer/render/Renderer";
import ParamGroup from "monks-webgpu/utils/params/ParamGroup";
import type { ParamMaterial } from "monks-webgpu/utils/params/ParamBase";
import Matrix4x4 from "monks-webgpu/renderer/math/Matrix4x4";
import Camera from "monks-webgpu/renderer/camera/Camera";
import hash from 'monks-webgpu/shaders/hash.wgsl?raw';
import Texture from "monks-webgpu/renderer/texture/Texture";
import Texture2D from "monks-webgpu/renderer/texture/Texture2D";
import ComputeShader from "monks-webgpu/renderer/material/ComputeShader";
import reproject from './reproject.wgsl?raw';
import fill from './fill.wgsl?raw';
import rayTraceScene from "../includes/rayTraceScene.wgsl?raw";
import pathTrace from "../includes/pathTrace.wgsl?raw";
import Scene from "../scene/Scene";
import Vector3 from "monks-webgpu/renderer/math/Vector3";
import normalPacking from 'monks-webgpu/shaders/normalPacking.wgsl?raw';
import gBufferPacking from '../includes/gBuffer.wgsl?raw';
import cameraRay from '../includes/cameraRay.wgsl?raw';
import r2 from 'monks-webgpu/shaders/r2.wgsl?raw';

//reprojects the lighting of the previous frame (reproject.wgsl) and path traces the new pixels and a refine budget of
//the pixels with too few samples (fill.wgsl). ReprojectAndFill and TemporalAccumulate set the fill params and the
//refine budget, see initFill and getRefineBudget. Resolve shows the lighting
export default abstract class ReprojectFill {
  protected renderer: Renderer;
  private inverseView: Matrix4x4 = new Matrix4x4();
  private inverseProjection: Matrix4x4 = new Matrix4x4();
  protected viewProjectionOld: Matrix4x4 = new Matrix4x4();
  //the camera of the previous frame, the reprojection gets the positions of the history pixels from it and depthOld
  private inverseViewOld: Matrix4x4 = new Matrix4x4();
  private inverseProjectionOld: Matrix4x4 = new Matrix4x4();
  private cameraPositionOld: Vector3 = new Vector3();
  private readonly workGroupSize = 64;
  private readonly workGroupSizeX = 8;
  private readonly workGroupSizeY = 8;
  //the gBuffer depth, written by the reprojection, and the one of the previous frame. Swapped every frame
  private depthTarget!: Texture;
  private depthOld!: Texture;
  //the running average of the light arriving at the pixels, without their albedo (rgb) and its sample count (a)
  private lightingReprojected!: Texture;
  private lightingTarget!: Texture;
  private newPixelsBuffer!: GPUBuffer;
  private refinePixelsBuffer!: GPUBuffer;
  private pixelCountsBuffer: GPUBuffer;
  protected reprojectShader: ComputeShader;
  public fillShader: ComputeShader;
  private refineOffset = 0;
  protected scene!: Scene;

  constructor(renderer: Renderer){
    this.renderer = renderer;

    this.reprojectShader = new ComputeShader(this.renderer, 'reproject', reproject, {includes: [normalPacking, gBufferPacking, cameraRay]});
    this.reprojectShader.setConstant('WORK_GROUP_SIZE_X', this.workGroupSizeX);
    this.reprojectShader.setConstant('WORK_GROUP_SIZE_Y', this.workGroupSizeY);

    this.fillShader = new ComputeShader(this.renderer, 'fill', fill, {includes: [rayTraceScene, pathTrace, r2, hash, normalPacking, gBufferPacking, cameraRay]});
    this.fillShader.setConstant('WORK_GROUP_SIZE', this.workGroupSize);

    //newCount, priorityCount, refineCount
    this.pixelCountsBuffer = this.renderer.createBuffer(12, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    this.reprojectShader.setBuffer('pixelCounts', this.pixelCountsBuffer);
    this.fillShader.setBuffer('pixelCounts', this.pixelCountsBuffer);
  }

  public init(paramGroup:ParamGroup, scene:Scene){
    this.scene = scene;
    const params = { materials: [this.reprojectShader] };
    const pgReproject = paramGroup.addGroup('reproject');
    pgReproject.addParamFloat('depthThreshold', .05, 0, .1, params);
    pgReproject.addParamBool('filterLinear', true, params);
    pgReproject.addParamBool('filterSharp', true, params);

    //with the reproject shader for newPixelStride, prioritySamples and accumulationMax
    const pgFillMaterials = [this.fillShader, this.reprojectShader];
    this.initFill(paramGroup.addGroup('fill'), pgFillMaterials);

    scene.bindPathTrace(this.fillShader);
  }

  //the params of the fill: newSampleCount, sampleCount, accumulationMax and the newPixelStride and prioritySamples of
  //the reprojection
  protected abstract initFill(pgFill:ParamGroup, pgFillMaterials:ParamMaterial[]): void;

  //the refined pixels that get samples this frame
  protected abstract getRefineBudget(pixelCount:number): number;

  //before the reprojection, viewProjectionOld is still the camera of the previous frame
  protected beforeReproject(_camera:Camera){}

  private setSize(){
    const width = this.renderer.width;
    const height = this.renderer.height;
    this.depthTarget?.destruct();
    this.depthOld?.destruct();
    this.lightingReprojected?.destruct();
    this.lightingTarget?.destruct();
    this.depthTarget = new Texture2D(this.renderer, width, height, {format: 'r32float', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING});
    this.depthOld = new Texture2D(this.renderer, width, height, {format: 'r32float', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING});
    this.lightingReprojected = new Texture2D(this.renderer, width, height, {format: 'rgba16float', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC});
    this.lightingTarget = new Texture2D(this.renderer, width, height, {format: 'rgba16float', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_DST});

    this.newPixelsBuffer?.destroy();
    this.refinePixelsBuffer?.destroy();
    this.newPixelsBuffer = this.renderer.createBuffer(width * height * 4, GPUBufferUsage.STORAGE);
    this.refinePixelsBuffer = this.renderer.createBuffer(width * height * 4, GPUBufferUsage.STORAGE);

    this.reprojectShader.setTexture('lighting', this.lightingReprojected);
    this.reprojectShader.setTexture('lightingOld', this.lightingTarget);
    this.reprojectShader.setBuffer('newPixels', this.newPixelsBuffer);
    this.reprojectShader.setBuffer('refinePixels', this.refinePixelsBuffer);
    this.reprojectShader.setVector2('resolution', this.renderer.size);

    this.fillShader.setTexture('lightingReprojected', this.lightingReprojected);
    this.fillShader.setTexture('lighting', this.lightingTarget);
    this.fillShader.setBuffer('newPixels', this.newPixelsBuffer);
    this.fillShader.setBuffer('refinePixels', this.refinePixelsBuffer);
    this.fillShader.setVector2('resolution', this.renderer.size);
  }

  public draw(camera:Camera, gBuffer: RenderTexture){
    if(!this.depthOld || this.depthOld.width !== this.renderer.width || this.depthOld.height !== this.renderer.height) {
      this.setSize();
    }

    Matrix4x4.invert(camera.viewMatrix, this.inverseView);
    Matrix4x4.invert(camera.projectionMatrix, this.inverseProjection);

    this.renderer.clearBuffer(this.pixelCountsBuffer);
    this.beforeReproject(camera);

    this.reprojectShader.setTexture('albedoTexture', gBuffer.views[0]);
    this.reprojectShader.setTexture('normalDepthTexture', gBuffer.views[1]);
    this.reprojectShader.setVector3('cameraPosition', camera.worldPosition);
    this.reprojectShader.setFloat('frame', this.renderer.time.frameNumber);
    //the pixels at accumulationMax stay refine pixels, so their lighting follows the light. See reproject.wgsl
    this.reprojectShader.setFloat('refineAtMax', this.scene.lightChanging ? 1 : 0);
    this.reprojectShader.setMatrix('viewProjectionOld', this.viewProjectionOld);
    this.reprojectShader.setMatrix('inverseView', this.inverseView);
    this.reprojectShader.setMatrix('inverseProjection', this.inverseProjection);
    this.reprojectShader.setVector3('cameraPositionOld', this.cameraPositionOld);
    this.reprojectShader.setMatrix('inverseViewOld', this.inverseViewOld);
    this.reprojectShader.setMatrix('inverseProjectionOld', this.inverseProjectionOld);
    this.reprojectShader.setTexture('depthTarget', this.depthTarget);
    this.reprojectShader.setTexture('depthOld', this.depthOld);
    this.renderer.dispatchCompute(this.reprojectShader, Math.ceil(this.renderer.width / this.workGroupSizeX), Math.ceil(this.renderer.height / this.workGroupSizeY));

    //the pixels the fill doesn't sample keep their reprojected lighting
    this.renderer.copyTextureToTexture(this.lightingReprojected.gpuTexture, this.lightingTarget.gpuTexture);

    const pixelCount = this.renderer.width * this.renderer.height;
    const refineBudget = this.getRefineBudget(pixelCount);
    this.refineOffset = (this.refineOffset + refineBudget) % pixelCount;
    this.fillShader.setTexture('normalDepthTexture', gBuffer.views[1]);
    this.fillShader.setVector3('cameraPosition', camera.worldPosition);
    this.fillShader.setMatrix('inverseProjection', this.inverseProjection);
    this.fillShader.setMatrix('inverseView', this.inverseView);
    this.fillShader.setFloat('frame', this.renderer.time.frameNumber);
    this.fillShader.setFloat('refineOffset', this.refineOffset);
    this.fillShader.setFloat('refineBudget', refineBudget);
    //a pixel is new, refined or neither: the new and refined pixels are at most all pixels. 2d: the group count of a
    //dimension is limited to 65535
    const groupCount = Math.ceil(pixelCount / this.workGroupSize);
    const groupCountX = Math.min(groupCount, 65535);
    this.renderer.dispatchCompute(this.fillShader, groupCountX, Math.ceil(groupCount / groupCountX));

    [this.depthTarget, this.depthOld] = [this.depthOld, this.depthTarget];
    this.viewProjectionOld.copy(camera.viewProjection);
    this.inverseViewOld.copy(this.inverseView);
    this.inverseProjectionOld.copy(this.inverseProjection);
    this.cameraPositionOld.copy(camera.worldPosition);
  }

  public get lighting(): Texture{
    return this.lightingTarget;
  }
}
