import ComputeShader from "monks-webgpu/renderer/material/ComputeShader";
import VertexFragmentShader from "monks-webgpu/renderer/material/VertexFragmentShader";
import Renderer from "monks-webgpu/renderer/render/Renderer";
import ParamGroup from "monks-webgpu/utils/params/ParamGroup";
import RenderTexture from "monks-webgpu/renderer/texture/RenderTexture";
import Camera from "monks-webgpu/renderer/camera/Camera";
import ParamLog from "monks-webgpu/utils/params/ParamLog";
import updateSpatialHash from './updateSpatialHash.wgsl?raw';
import resolveSpatialHash from './resolveSpatialHash.wgsl?raw';
import spatialHash from './spatialHash.wgsl?raw';
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
the diffuse lighting cached in a world space spatial hash map, instead of in the octree of OctreeUpdater.
The update is the one of OctreeUpdater: a random pixel per screen cell gets sampleCount paths, which end in the cached lighting
of their first hit, and the result is accumulated in the hash cell of the pixel and in the coarser cells above it.
-a cell is a world space cube, its size the power of 2 that covers about cellSize pixels at its distance
-the normal is part of the key, so the faces at a hard edge or the 2 sides of a wall never share lighting
-no bounds, and nothing to split or collapse: cells that no pixel used for evictFrames are taken over by new ones
The resolve interpolates the 4 nearest cells of 2 levels, see resolveSpatialHash.wgsl
 */
export default class SpatialHash {
  private renderer: Renderer;
  public updateShader: ComputeShader;
  private resolveMaterial: VertexFragmentShader;
  public hashBuffer: GPUBuffer;
  public hashTimeBuffer: GPUBuffer;
  public lightingBuffer: GPUBuffer;
  //the slots in use: an evicted cell is replaced, so only the inserts in empty slots count
  private cellCount: GPUBuffer;
  public readonly hashMapSize = Math.pow(2, 22);
  //screen pixels along a side of the cells that each get a sample per frame, at a screen height of referenceHeight. Small
  //cells with few samples each, like OctreeUpdater. They follow the resolution like the hash cells do, see getCellSize
  private readonly referenceCellSize = 4;
  //the CELL_SIZE of the shader, in pixels of the screen as it is
  private cellSize = 4;
  //the screen height the hash cell size (the cellSize param, in pixels) is for, see getPixelSize
  private readonly referenceHeight = 1080;
  private readonly workGroupSizeX = 8;
  private readonly workGroupSizeY = 8;
  private reset = true;
  private cellCountLog!: ParamLog;
  private cellCountReading = false;

  public constructor(renderer: Renderer) {
    this.renderer = renderer;

    this.updateShader = new ComputeShader(this.renderer, 'updateSpatialHash', updateSpatialHash, {includes: [hash, spatialHash, rayTraceScene, pathTrace, r2, normalPacking, gBufferPacking, cameraRay]});
    this.updateShader.setConstant('WORK_GROUP_SIZE_X', this.workGroupSizeX);
    this.updateShader.setConstant('WORK_GROUP_SIZE_Y', this.workGroupSizeY);
    this.updateShader.setConstant('CELL_SIZE', this.cellSize);

    this.resolveMaterial = new VertexFragmentShader(this.renderer, 'resolveSpatialHash', resolveSpatialHash, {includes: [spatialHash, toneMap, normalPacking, gBufferPacking, cameraRay]});
    this.resolveMaterial.cullMode = 'none';

    this.hashBuffer = this.renderer.createBuffer(this.hashMapSize * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);
    this.hashTimeBuffer = this.renderer.createBuffer(this.hashMapSize * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    //rgb lighting + sample count as 4 x f16 per cell
    this.lightingBuffer = this.renderer.createBuffer(this.hashMapSize * 8, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);

    [this.updateShader, this.resolveMaterial].forEach(s => {
      s.setConstant('HASHMAP_SIZE', this.hashMapSize);
      s.setBuffer('hashBuffer', this.hashBuffer);
      s.setBuffer('lightingBuffer', this.lightingBuffer);
    });
    this.updateShader.setBuffer('hashTimeBuffer', this.hashTimeBuffer);
    this.cellCount = this.renderer.createBuffer(4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);
    this.updateShader.setBuffer('cellCount', this.cellCount);
  }

  public get exposureMaterial() {
    return this.resolveMaterial;
  }

  public init(paramGroup:ParamGroup, scene:Scene) {
    scene.bindSkyColor(this.resolveMaterial);

    //the cells, the same in the update and the resolve
    const pgParams = { materials: [this.updateShader, this.resolveMaterial] };
    const pg = paramGroup.addGroup('spatialHash');
    pg.addButton('reset', { callback: () => {
      this.reset = true;
    } });
    //the size of a cell in pixels, at a screen height of referenceHeight whatever the resolution
    pg.addParamFloat('cellSize', 2, 1, 64, pgParams);
    //the size of the smallest cells
    pg.addParamFloat('sizeMin', .01, .001, .2, pgParams);
    //slots probed before a cell gives up
    pg.addParamInt('searchCount', 10, 1, 32, pgParams);
    this.cellCountLog = pg.addLog('cells: 0');

    const pgUpdateParams = { materials: [this.updateShader] };
    const pgUpdate = pg.addGroup('spatialHashUpdate');
    //lighting samples per screen cell per frame
    pgUpdate.addParamInt('sampleCount', 8, 1, 64, pgUpdateParams);
    //frames the running average of a cell holds. Lower reacts faster to changes, higher is less noisy
    pgUpdate.addParamFloat('accumulationMax', 256, 1, 2048, pgUpdateParams);
    //the paths end at their first hit, in the lighting the hash map has there, instead of tracing bounceCount bounces
    pgUpdate.addParamBool('cacheBounces', true, pgUpdateParams);
    //coarser levels that accumulate the samples too. The resolve needs 2 to blend between levels
    pgUpdate.addParamInt('ancestorLevels', 2, 0, 4, pgUpdateParams);
    //the samples of the coarser lighting a new cell starts with, instead of starting from its first noisy sample
    pgUpdate.addParamFloat('seedCount', 4, 0, 64, pgUpdateParams);
    //the frames after which a cell no pixel uses can be taken by another one
    pgUpdate.addParamFloat('evictFrames', 120, 1, 1000, pgUpdateParams);

    const pgResolveParams = { materials: [this.resolveMaterial] };
    const pgResolve = pg.addGroup('spatialHashResolve');
    //the samples the coarser level counts as in the lighting shown, so cells with few samples show the smoother coarser lighting
    pgResolve.addParamFloat('ancestorWeight', 8, 0, 64, pgResolveParams);
    //the ratio of the 2 largest normal components from which the 2 nearest cell directions blend, on curved surfaces
    pgResolve.addParamFloat('directionBlend', .5, 0, 1, pgResolveParams);
    //the pixels no cell covers in red
    pgResolve.addParamBool('showHoles', false, pgResolveParams);
    pgResolve.addParamBool('showLevels', false, pgResolveParams);

    scene.bindPathTrace(this.updateShader);
  }

  public update(gBuffer:RenderTexture, camera:Camera) {
    if(this.reset){
      this.renderer.clearBuffer(this.hashBuffer);
      this.renderer.clearBuffer(this.hashTimeBuffer);
      this.renderer.clearBuffer(this.lightingBuffer);
      this.renderer.clearBuffer(this.cellCount);
      this.reset = false;
    }

    const pixelSize = this.getPixelSize(camera);
    const cellSize = this.getCellSize(gBuffer);
    if(cellSize != this.cellSize){
      this.cellSize = cellSize;
      this.updateShader.setConstant('CELL_SIZE', cellSize);
    }

    this.updateShader.setVector3('cameraPosition', camera.worldPosition);
    this.updateShader.setMatrix('inverseView', camera.viewMatrix.inverse);
    this.updateShader.setMatrix('inverseProjection', camera.projectionMatrix.inverse);
    this.updateShader.setFloat('frame', this.renderer.time.frameNumber);
    this.updateShader.setFloat('pixelSize', pixelSize);
    this.updateShader.setVector2('resolution', gBuffer.size);
    this.updateShader.setTexture('normalDepthTexture', gBuffer.views[1]);

    const countX = Math.ceil(gBuffer.width / (this.workGroupSizeX * this.cellSize));
    const countY = Math.ceil(gBuffer.height / (this.workGroupSizeY * this.cellSize));
    this.renderer.dispatchCompute(this.updateShader, countX, countY, 1);

    //the used slots of the hash map
    if(this.renderer.time.frameNumber % 30 == 0 && !this.cellCountReading){
      this.cellCountReading = true;
      this.renderer.readBuffer(this.cellCount, Uint32Array).then(([count]) => {
        this.cellCountLog.setValue(`cells: ${count} / ${this.hashMapSize} (${(count / this.hashMapSize * 100).toFixed(1)}%)`);
        this.cellCountReading = false;
      });
    }
  }

  public draw(camera:Camera, gBuffer:RenderTexture, destination:RenderTexture | null = null) {
    this.resolveMaterial.setVector3('cameraPosition', camera.worldPosition);
    this.resolveMaterial.setMatrix('inverseView', camera.viewMatrix.inverse);
    this.resolveMaterial.setMatrix('inverseProjection', camera.projectionMatrix.inverse);
    this.resolveMaterial.setVector2('resolution', gBuffer.size);
    this.resolveMaterial.setFloat('pixelSize', this.getPixelSize(camera));
    this.resolveMaterial.setTexture('normalDepthTexture', gBuffer.views[1]);
    this.resolveMaterial.setTexture('albedoTexture', gBuffer.views[0]);
    this.renderer.blit(null, destination, this.resolveMaterial);
  }

  //referenceCellSize at the height of the screen, in whole pixels: 4 at 1080p, 5 at 1440p, 8 at 4K. So the samples per hash
  //cell per frame don't change with the resolution. A constant of the shader, so a resize that changes it compiles it again
  private getCellSize(gBuffer:RenderTexture) {
    return Math.max(1, Math.round(this.referenceCellSize * gBuffer.height / this.referenceHeight));
  }

  //the world size of a pixel at distance 1, of a screen referenceHeight pixels high: not of the real screen, so the hash cells
  //don't follow the resolution. Otherwise 4K has 4x the cells of 1080p, each with the same samples per frame, in a hash map that
  //is 4x as full: longer probes, more inserts that give up after searchCount (holes), more evictions
  private getPixelSize(camera:Camera) {
    return Math.tan(camera.projection.FOV * .5) * 2 / this.referenceHeight;
  }
}
