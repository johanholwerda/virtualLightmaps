import RenderTexture from "monks-webgpu/renderer/texture/RenderTexture";
import Renderer from "monks-webgpu/renderer/render/Renderer";
import ParamGroup from "monks-webgpu/utils/params/ParamGroup";
import Matrix4x4 from "monks-webgpu/renderer/math/Matrix4x4";
import Camera from "monks-webgpu/renderer/camera/Camera";
import hash from 'monks-webgpu/shaders/hash.wgsl?raw';
import Texture from "monks-webgpu/renderer/texture/Texture";
import Texture2D from "monks-webgpu/renderer/texture/Texture2D";
import ComputeShader from "monks-webgpu/renderer/material/ComputeShader";
import VertexFragmentShader from "monks-webgpu/renderer/material/VertexFragmentShader";
import Vector2 from "monks-webgpu/renderer/math/Vector2";
import Vector3 from "monks-webgpu/renderer/math/Vector3";
import toneMap from 'monks-webgpu/shaders/toneMap.wgsl?raw';
import normalPacking from 'monks-webgpu/shaders/normalPacking.wgsl?raw';
import accumulate from './accumulate.wgsl?raw';
import upsample from './upsample.wgsl?raw';
import rayTraceScene from "../includes/rayTraceScene.wgsl?raw";
import pathTrace from "../includes/pathTrace.wgsl?raw";
import gBufferPacking from '../includes/gBuffer.wgsl?raw';
import cameraRay from '../includes/cameraRay.wgsl?raw';
import Scene from "../scene/Scene";
import r2 from 'monks-webgpu/shaders/r2.wgsl?raw';

//TemporalAccumulate at half resolution: a quarter of the paths, the path trace of every pixel
//every frame is what makes TemporalAccumulate slow. accumulate.wgsl reprojects and samples every half res pixel in
//one pass, a half res pixel stands for the top left full res pixel of its 2x2. upsample.wgsl is the resolve: it
//interpolates the half res lighting per full res pixel, over the half res pixels on the same surface, and applies the
//full res albedo. The albedo and the geometry stay sharp, the lighting is half res: that includes the sun shadows
export default class TemporalAccumulateHalfRes {
  private renderer: Renderer;
  private scene!: Scene;
  private inverseView: Matrix4x4 = new Matrix4x4();
  private inverseProjection: Matrix4x4 = new Matrix4x4();
  private viewProjectionOld: Matrix4x4 = new Matrix4x4();
  //the camera of the previous frame, the reprojection gets the positions of the history pixels from it and normalDepthOld
  private inverseViewOld: Matrix4x4 = new Matrix4x4();
  private inverseProjectionOld: Matrix4x4 = new Matrix4x4();
  private cameraPositionOld: Vector3 = new Vector3();
  private readonly workGroupSizeX = 8;
  private readonly workGroupSizeY = 8;
  private halfResolution: Vector2 = new Vector2();
  //the packed normalDepth of the full res pixels the half res pixels stand for, of this frame and the previous one.
  //Swapped every frame
  private normalDepthTarget!: Texture;
  private normalDepthOld!: Texture;
  //the running average of the light arriving at the pixels, without their albedo (rgb) and its sample count (a).
  //Swapped every frame
  private lightingTarget!: Texture;
  private lightingOld!: Texture;
  private gBufferWidth = 0;
  private gBufferHeight = 0;
  public accumulateShader: ComputeShader;
  private upsampleMaterial: VertexFragmentShader;

  constructor(renderer: Renderer){
    this.renderer = renderer;

    this.accumulateShader = new ComputeShader(this.renderer, 'accumulateHalfRes', accumulate, {includes: [rayTraceScene, pathTrace, r2, hash, normalPacking, gBufferPacking, cameraRay]});
    this.accumulateShader.setConstant('WORK_GROUP_SIZE_X', this.workGroupSizeX);
    this.accumulateShader.setConstant('WORK_GROUP_SIZE_Y', this.workGroupSizeY);

    this.upsampleMaterial = new VertexFragmentShader(this.renderer, 'upsampleHalfRes', upsample, {includes: [toneMap, normalPacking, gBufferPacking, cameraRay]});
  }

  public get exposureMaterial() {
    return this.upsampleMaterial;
  }

  public init(paramGroup:ParamGroup, scene:Scene){
    this.scene = scene;
    const pgAccumulateParams = { materials: [this.accumulateShader] };
    const pgAccumulate = paramGroup.addGroup('accumulate');
    pgAccumulate.addParamFloat('depthThreshold', .02, 0, .1, pgAccumulateParams);
    pgAccumulate.addParamBool('filterLinear', true, pgAccumulateParams);
    pgAccumulate.addParamBool('filterSharp', true, pgAccumulateParams);
    //the frames the exponential average is about long while the camera moves, and the samples at which a pixel
    //stops getting them while it is static
    pgAccumulate.addParamFloat('accumulationMax', 256, 1, 256, pgAccumulateParams);
    scene.bindPathTrace(this.accumulateShader);

    const pgUpsampleParams = { materials: [this.upsampleMaterial] };
    const pgUpsample = paramGroup.addGroup('upsample');
    scene.bindSkyColor(this.upsampleMaterial);
    pgUpsample.addParamFloat('planeTolerance', .02, 0, .2, pgUpsampleParams);
    pgUpsample.addParamBool('showSampleCount', false, pgUpsampleParams);
    //the sample count that shows white, accumulationMax
    pgUpsample.addParamFloat('sampleCountMax', 256, 1, 256, pgUpsampleParams);
  }

  private setSize(width: number, height: number){
    this.gBufferWidth = width;
    this.gBufferHeight = height;
    //a half res pixel stands for full res pixel 2c, which is inside the gBuffer for an odd size too
    this.halfResolution.x = Math.ceil(width / 2);
    this.halfResolution.y = Math.ceil(height / 2);
    const halfWidth = this.halfResolution.x;
    const halfHeight = this.halfResolution.y;
    this.normalDepthTarget?.destruct();
    this.normalDepthOld?.destruct();
    this.lightingTarget?.destruct();
    this.lightingOld?.destruct();
    const usage = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING;
    this.normalDepthTarget = new Texture2D(this.renderer, halfWidth, halfHeight, {format: 'rg32uint', usage});
    this.normalDepthOld = new Texture2D(this.renderer, halfWidth, halfHeight, {format: 'rg32uint', usage});
    this.lightingTarget = new Texture2D(this.renderer, halfWidth, halfHeight, {format: 'rgba16float', usage});
    this.lightingOld = new Texture2D(this.renderer, halfWidth, halfHeight, {format: 'rgba16float', usage});

    const resolution = new Vector2(width, height);
    this.accumulateShader.setVector2('resolution', resolution);
    this.accumulateShader.setVector2('halfResolution', this.halfResolution);
    this.upsampleMaterial.setVector2('resolution', resolution);
  }

  public draw(camera:Camera, gBuffer: RenderTexture, destination: RenderTexture | null = null){
    if(!this.lightingTarget || this.gBufferWidth !== this.renderer.width || this.gBufferHeight !== this.renderer.height) {
      this.setSize(this.renderer.width, this.renderer.height);
    }

    Matrix4x4.invert(camera.viewMatrix, this.inverseView);
    Matrix4x4.invert(camera.projectionMatrix, this.inverseProjection);

    //static: the pixels at accumulationMax stop getting samples. Moving: every pixel gets a sample, the count stays
    //at accumulationMax, which makes the running average an exponential one
    //not the view projection, that has the TAA jitter. A moving light changes the lighting like a moving camera
    const isStatic = !camera.changed && !this.scene.lightChanging;

    this.accumulateShader.setTexture('albedoTexture', gBuffer.views[0]);
    this.accumulateShader.setTexture('normalDepthTexture', gBuffer.views[1]);
    this.accumulateShader.setTexture('normalDepthOld', this.normalDepthOld);
    this.accumulateShader.setTexture('normalDepthTarget', this.normalDepthTarget);
    this.accumulateShader.setTexture('lightingOld', this.lightingOld);
    this.accumulateShader.setTexture('lighting', this.lightingTarget);
    this.accumulateShader.setVector3('cameraPosition', camera.worldPosition);
    this.accumulateShader.setFloat('frame', this.renderer.time.frameNumber);
    this.accumulateShader.setFloat('isStatic', isStatic ? 1 : 0);
    this.accumulateShader.setMatrix('viewProjectionOld', this.viewProjectionOld);
    this.accumulateShader.setMatrix('inverseView', this.inverseView);
    this.accumulateShader.setMatrix('inverseProjection', this.inverseProjection);
    this.accumulateShader.setVector3('cameraPositionOld', this.cameraPositionOld);
    this.accumulateShader.setMatrix('inverseViewOld', this.inverseViewOld);
    this.accumulateShader.setMatrix('inverseProjectionOld', this.inverseProjectionOld);
    this.renderer.dispatchCompute(this.accumulateShader, Math.ceil(this.halfResolution.x / this.workGroupSizeX), Math.ceil(this.halfResolution.y / this.workGroupSizeY));

    this.upsampleMaterial.setTexture('albedoTexture', gBuffer.views[0]);
    this.upsampleMaterial.setTexture('normalDepthTexture', gBuffer.views[1]);
    this.upsampleMaterial.setTexture('halfNormalDepthTexture', this.normalDepthTarget);
    this.upsampleMaterial.setVector3('cameraPosition', camera.worldPosition);
    this.upsampleMaterial.setMatrix('inverseView', this.inverseView);
    this.upsampleMaterial.setMatrix('inverseProjection', this.inverseProjection);
    this.renderer.blit(this.lightingTarget, destination, this.upsampleMaterial);

    [this.normalDepthTarget, this.normalDepthOld] = [this.normalDepthOld, this.normalDepthTarget];
    [this.lightingTarget, this.lightingOld] = [this.lightingOld, this.lightingTarget];
    this.viewProjectionOld.copy(camera.viewProjection);
    this.inverseViewOld.copy(this.inverseView);
    this.inverseProjectionOld.copy(this.inverseProjection);
    this.cameraPositionOld.copy(camera.worldPosition);
  }
}
