import ComputeShader from "monks-webgpu/renderer/material/ComputeShader";
import VertexFragmentShader from "monks-webgpu/renderer/material/VertexFragmentShader";
import Renderer from "monks-webgpu/renderer/render/Renderer";
import ParamGroup from "monks-webgpu/utils/params/ParamGroup";
import RenderTexture from "monks-webgpu/renderer/texture/RenderTexture";
import Camera from "monks-webgpu/renderer/camera/Camera";
import ParamLog from "monks-webgpu/utils/params/ParamLog";
import updateVirtualLightmap from './updateVirtualLightmap.wgsl?raw';
import resolveVirtualLightmap from './resolveVirtualLightmap.wgsl?raw';
import virtualLightmap from './virtualLightmap.wgsl?raw';
import hash from 'monks-webgpu/shaders/hash.wgsl?raw';
import toneMap from 'monks-webgpu/shaders/toneMap.wgsl?raw';
import rayTraceScene from "../includes/rayTraceScene.wgsl?raw";
import pathTrace from "../includes/pathTrace.wgsl?raw";
import normalPacking from 'monks-webgpu/shaders/normalPacking.wgsl?raw';
import gBufferPacking from '../includes/gBuffer.wgsl?raw';
import cameraRay from '../includes/cameraRay.wgsl?raw';
import Scene from "../scene/Scene";
import ParamFloat from "monks-webgpu/utils/params/ParamFloat";
import ParamBool from "monks-webgpu/utils/params/ParamBool";
import ParamInt from "monks-webgpu/utils/params/ParamInt";
import ParamSelect from "monks-webgpu/utils/params/ParamSelect";
import Vector3 from "monks-webgpu/renderer/math/Vector3";
import Vector4 from "monks-webgpu/renderer/math/Vector4";
import r2 from 'monks-webgpu/shaders/r2.wgsl?raw';

//the lightmap sizes of the dropdown, see sizeParam
const SIZE_OPTIONS = [1024, 1536, 2048, 3072, 4096];

//the settings of the quality dropdown, see qualityParam. Picking one sets these params, which can still be changed one
//by one after. A lower lightmapSize needs a lower texelsPerPixel, otherwise the lightmap is continuously full
type QualitySetting = {lightmapSize: number, texelsPerPixel: number, sampleCount: number, referenceCellSize: number, traceSamplesMax: number, accumulationMax: number};
const QUALITY_NAMES = ['low', 'medium', 'high', 'max'];
const QUALITY_SETTINGS: QualitySetting[] = [
  {lightmapSize: 1536, texelsPerPixel: .1, sampleCount: 8, referenceCellSize: 4, traceSamplesMax: 1024, accumulationMax: 512},
  {lightmapSize: 2048, texelsPerPixel: .2, sampleCount: 16, referenceCellSize: 3, traceSamplesMax: 2048, accumulationMax: 1024},
  {lightmapSize: 3072, texelsPerPixel: .4, sampleCount: 16, referenceCellSize: 2, traceSamplesMax: 4096, accumulationMax: 1024},
  {lightmapSize: 4096, texelsPerPixel: .6, sampleCount: 32, referenceCellSize: 1, traceSamplesMax: 8192, accumulationMax: 2048},
];
const QUALITY_DEFAULT = QUALITY_NAMES.indexOf('high');

/*
the diffuse lighting cached in a lightmap of a fixed size, which only holds what is (and was) in view.
-the charts are the faces of the models: every primitive has 6, see getSurface in virtualLightmap.wgsl. The lighting is
 interpolated on a face, never from one face to another
-a face has texels at power of 2 sizes (levels), the level in use is the one with about texelsPerPixel texels per screen pixel.
 The level 0 texels are texelSizeMin, which is the limit on the texels per m
-the lightmap is cut in tiles of tileSize x tileSize texels. A tile of a face at a level only gets a place in the lightmap when a
 pixel uses it, so a large floor takes no more than the part that is seen, at the resolution it is seen at. The place is a slot
 of a hash map on (face, level, tile), so finding the texels of a pixel is 1 lookup for the whole tile
-a tile that is out of view keeps its place and its lighting, until another tile needs the place: then a tile that was
 unused for long makes room, see findOrInsert in updateVirtualLightmap.wgsl
The update is like the one of SpatialHash: a random pixel per screen cell gets sampleCount paths, which end in the cached lighting
of their first hit, and the result is accumulated in the texel of the pixel and in the texels of the coarser levels above it.
But only 1 cell per texel traces per frame, the winner of a random bid, and a texel that holds accumulationMax paths takes no
more while the light doesn't change. So the cost follows the texels that are still converging, not the pixels: a low
texelsPerPixel or a converged view costs little. See claim_main in updateVirtualLightmap.wgsl.
The resolve interpolates the 4 nearest texels of 2 levels, see resolveVirtualLightmap.wgsl
 */
export default class VirtualLightmap {
  private renderer: Renderer;
  //the passes of the update, see claim_main in updateVirtualLightmap.wgsl. updateShader traces the paths
  private claimShader: ComputeShader;
  private selectShader: ComputeShader;
  public updateShader: ComputeShader;
  private refreshShader: ComputeShader;
  private updateShaders: ComputeShader[];
  private resolveMaterial: VertexFragmentShader;
  //the buffers that follow the size of the lightmap, see createLightmapBuffers
  private tileBuffer!: GPUBuffer;
  private tileTimeBuffer!: GPUBuffer;
  private lightingBuffer!: GPUBuffer;
  //the slots in use: an evicted tile is replaced, so only the inserts in empty slots count
  private tileCountBuffer: GPUBuffer;
  //per texel the highest bid of this frame
  private claimBuffer!: GPUBuffer;
  //the cells that trace this frame, and after those per screen cell the texel it bids on (from the claim pass to the
  //select pass): 2 slots per screen cell. Grows with the resolution
  private traceCellsBuffer: GPUBuffer | null = null;
  //the workgroups of the trace pass and the cell count, written by the select pass. Then the cells that wanted to trace,
  //of the even and of the odd frames: see traceArgs in updateVirtualLightmap.wgsl
  private traceArgsBuffer: GPUBuffer;
  //a copy of traceArgsBuffer: a buffer can't be the indirect buffer and storage of the same dispatch
  private traceIndirectBuffer: GPUBuffer;
  //the texels along a side of the lightmap, the LIGHTMAP_SIZE constant of the shaders. Set by sizeParam, which recreates
  //the buffers and resets the lightmap, see createLightmapBuffers
  public size = 4096;
  private readonly tileSize = 8;
  public tileCount = 0;
  private sizeParam!: ParamSelect;
  private texelsPerPixel!: ParamFloat;
  //screen pixels along a side of the cells that each get a sample per frame, like SpatialHash. At a screen height of
  //referenceHeight: the cells follow the resolution like the texels do, so a 4K screen has the cells (and the paths per
  //frame) of a 1080p one, see getCellSize
  private referenceCellSize!: ParamInt;
  //the CELL_SIZE of the shaders, in pixels of the screen as it is
  private cellSize = 4;
  //the EVICT_FIRST of the shaders, see evictFirstParam
  private evictFirst = false;
  private evictFirstParam!: ParamBool;
  private evictFrames!: ParamFloat;
  private evictDistance!: ParamFloat;
  //the frames a tile is unused at least before it can go, however far the camera went. See getEvictFrames
  private readonly evictFramesMin = 2;
  //where the camera was in the last frames, the newest last: at most evictFrames of them
  private cameraHistory: {frame: number, position: Vector3}[] = [];
  //the screen height texelsPerPixel is for, see getPixelSize
  private readonly referenceHeight = 1080;
  private readonly workGroupSizeX = 8;
  private readonly workGroupSizeY = 8;
  //the most threads that share the paths of a cell, see getTraceLanes and trace_main in updateVirtualLightmap.wgsl
  private readonly traceLanesMax = 8;
  //the samples a resolved texel goes back to when the light stops changing, as a part of accumulationMax: the lighting
  //it holds lags behind the light, the new samples refresh it. Twice, so the square of this part of the lagging lighting
  //stays in the texel. See refresh_main
  private readonly refreshFraction = 1 / 8;
  //frames until the second refresh, 0 when none is due
  private refreshFrames = 0;
  private reset = true;
  private scene!: Scene;
  private accumulationMax!: ParamFloat;
  private sampleCount!: ParamInt;
  private traceSamplesMax!: ParamInt;
  private changeFrames!: ParamFloat;
  //Scene.sunChanging of the previous frame
  private sunChanging = false;
  //the last drags, the newest first (the oldest ones merged, see mergeOldestStrokes), each as a capsule around where the
  //model went from the start of the drag to its end. A texel in its influence that was last sampled before the end of
  //the drag is reset, see resetStaleTexel in updateVirtualLightmap.wgsl. points: where the model went, for the width of
  //the capsule
  private dragStrokes: {points: Vector3[], from: Vector3, to: Vector3, radius: number, model: number, endFrame: number}[] = [];
  //the DRAG_STROKE_COUNT of the shaders
  private readonly dragStrokeCount = 4;
  //the first of dragStrokes is of the drag that is going on
  private dragStrokeActive = false;
  //during a drag a texel in its influence that had no sample for this many times changeFrames frames is reset: it is
  //not following the drag
  private readonly dragStaleFactor = 4;
  private dragModelIndex = 0;
  private dragModelRadius = 0;
  //the first and the last frame of the last time the sun moved, 0 when it never did. Tells how far the lighting of
  //a tile that was out of view is behind, see getStaleFrames in updateVirtualLightmap.wgsl
  private lightChangeStart = 0;
  private lightChangeEnd = 0;
  private tileCountLog!: ParamLog;
  private tileCountReading = false;
  public showLightmap!: ParamBool;
  private stochasticLevels!: ParamBool;

  public constructor(renderer: Renderer) {
    this.renderer = renderer;

    const includes = [hash, rayTraceScene, virtualLightmap, pathTrace, r2, normalPacking, gBufferPacking, cameraRay];
    this.claimShader = new ComputeShader(this.renderer, 'claimVirtualLightmap', updateVirtualLightmap, {includes, kernel: 'claim_main'});
    this.selectShader = new ComputeShader(this.renderer, 'selectVirtualLightmap', updateVirtualLightmap, {includes, kernel: 'select_main'});
    this.updateShader = new ComputeShader(this.renderer, 'updateVirtualLightmap', updateVirtualLightmap, {includes, kernel: 'trace_main'});
    this.refreshShader = new ComputeShader(this.renderer, 'refreshVirtualLightmap', updateVirtualLightmap, {includes, kernel: 'refresh_main'});
    this.updateShaders = [this.claimShader, this.selectShader, this.updateShader, this.refreshShader];
    this.updateShaders.forEach(s => {
      s.setConstant('WORK_GROUP_SIZE_X', this.workGroupSizeX);
      s.setConstant('WORK_GROUP_SIZE_Y', this.workGroupSizeY);
      s.setConstant('TRACE_GROUP_SIZE', this.workGroupSizeX * this.workGroupSizeY);
      s.setConstant('CELL_SIZE', this.cellSize);
    });
    //the model of a ray hit, for getCachedLighting. See rayTraceScene.wgsl
    this.updateShader.setConstant('TRACK_HIT_MODEL', true);

    this.resolveMaterial = new VertexFragmentShader(this.renderer, 'resolveVirtualLightmap', resolveVirtualLightmap, {includes: [rayTraceScene, virtualLightmap, toneMap, normalPacking, gBufferPacking, cameraRay]});
    this.resolveMaterial.cullMode = 'none';

    [...this.updateShaders, this.resolveMaterial].forEach(s => s.setConstant('TILE_SIZE', this.tileSize));
    this.createLightmapBuffers();

    this.tileCountBuffer = this.renderer.createBuffer(4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);
    this.traceArgsBuffer = this.renderer.createBuffer(32, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);
    this.traceIndirectBuffer = this.renderer.createBuffer(16, GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST);
    this.updateShaders.forEach(s => {
      s.setBuffer('tileCount', this.tileCountBuffer);
      s.setBuffer('traceArgs', this.traceArgsBuffer);
    });
  }

  //the buffers of the texels and the tiles, for the current size, and the constants that follow it. Replaces the ones
  //of a previous size, so the lightmap starts over
  private createLightmapBuffers() {
    this.tileBuffer?.destroy();
    this.tileTimeBuffer?.destroy();
    this.lightingBuffer?.destroy();
    this.claimBuffer?.destroy();

    this.tileCount = Math.pow(this.size / this.tileSize, 2);
    this.tileBuffer = this.renderer.createBuffer(this.tileCount * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    this.tileTimeBuffer = this.renderer.createBuffer(this.tileCount * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    //rgb lighting + sample count as 4 x f16 per texel
    this.lightingBuffer = this.renderer.createBuffer(this.size * this.size * 8, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    this.claimBuffer = this.renderer.createBuffer(this.size * this.size * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);

    [...this.updateShaders, this.resolveMaterial].forEach(s => {
      s.setConstant('LIGHTMAP_SIZE', this.size);
      s.setConstant('TILE_COUNT', this.tileCount);
      s.setBuffer('tileBuffer', this.tileBuffer);
      s.setBuffer('lightingBuffer', this.lightingBuffer);
    });
    this.updateShaders.forEach(s => {
      s.setBuffer('tileTimeBuffer', this.tileTimeBuffer);
      s.setBuffer('claimBuffer', this.claimBuffer);
    });
    this.reset = true;
  }

  public get exposureMaterial() {
    return this.resolveMaterial;
  }

  public init(paramGroup:ParamGroup, scene:Scene) {
    scene.bindSkyColor(this.resolveMaterial);

    //the texels, the same in the update and the resolve
    const pg = paramGroup.addGroup('virtualLightmap');
    pg.addButton('reset', { callback: () => {
      this.reset = true;
    } });
    //sets the params of a QualitySetting
    pg.addParamSelect('quality', QUALITY_DEFAULT, QUALITY_NAMES, { callback: (index) => this.applyQuality(QUALITY_SETTINGS[index]) });
    //the texels along a side of the lightmap. A change recreates its buffers and starts it over, see update
    //does not influence the performance a lot

    const pgSharedParams = { materials: [...this.updateShaders, this.resolveMaterial] };
    const pgShared = pg.addGroup('virtualLightmapShared');
    this.sizeParam = pgShared.addParamSelect('lightmapSize', SIZE_OPTIONS.indexOf(QUALITY_SETTINGS[QUALITY_DEFAULT].lightmapSize), SIZE_OPTIONS.map(String));
    //lightmap texels per screen pixel along a side, at a screen height of referenceHeight whatever the resolution. Lower takes less of the lightmap, so more of what was seen before stays in it
    //lower this when lowering lightmapSize, otherwise the lightmap will be continuously full
    this.texelsPerPixel = pgShared.addParamFloat('texelsPerPixel', QUALITY_SETTINGS[QUALITY_DEFAULT].texelsPerPixel, .05, 1, pgSharedParams);
    //the size of the smallest texels in m, so 1 / the most texels per m: a surface the camera is close to takes no more than this
    pgShared.addParamFloat('texelSizeMin', .002, .001, .1, pgSharedParams);
    //how much the texels grow where the surface is seen at a grazing angle, see getLevel in virtualLightmap.wgsl. 0: their
    //size follows the distance only. .5: the same number of texels per pixel at every angle. 1: as long as the pixel is on the surface
    pgShared.addParamFloat('levelAngle', .5, 0, 1, pgSharedParams);
    //slots probed before a tile gives up
    pgShared.addParamInt('searchCount', 16, 1, 32, pgSharedParams);
    this.tileCountLog = pgShared.addLog('tiles: 0');
    pgShared.collapsed = true;

    const pgUpdate = pg.addGroup('virtualLightmapUpdate');
    //lighting samples per screen cell per frame
    const params2 = { materials: this.updateShaders };
    this.sampleCount = pgUpdate.addParamInt('sampleCount', QUALITY_SETTINGS[QUALITY_DEFAULT].sampleCount, 8, 64, { ...params2, step:8 });
    //screen pixels along a side of a cell, at a screen height of referenceHeight: see getCellSize. Lower is more cells,
    //so more texels get their sampleCount paths per frame: half the size is up to 4 times the paths. A change compiles
    //the update shaders again, as it is a constant there
    this.referenceCellSize = pgUpdate.addParamInt('referenceCellSize', QUALITY_SETTINGS[QUALITY_DEFAULT].referenceCellSize, 1, 16);
    //the most paths traced per frame, times 1000: the cells that trace times sampleCount. The cells over it wait for a
    //next frame, so a view of new texels (the camera turns fast) takes more frames to fill in instead of a long frame.
    //Also slows down the start and how fast the lighting follows the sun. A 16:9 screen has about 130000 cells at any
    //resolution (see getCellSize), so from 2074 at a sampleCount of 16 nothing is limited. See select_main in
    //updateVirtualLightmap.wgsl
    this.traceSamplesMax = pgUpdate.addParamInt('traceSamplesMax', QUALITY_SETTINGS[QUALITY_DEFAULT].traceSamplesMax, 64, 8192, { step:64 });
    //paths the running average of a texel holds, so a higher sampleCount gets there in fewer frames. Lower reacts faster to
    //changes, higher is less noisy. A texel that holds them gets no more paths while the light doesn't change, it is resolved.
    //At most 2048: the count is an f16, which has no gaps between its integers up to there.
    //The resolve too, for showConverging
    const setResolveAccumulationMax = (value: number) => this.resolveMaterial.setFloat('accumulationMax', value);
    this.accumulationMax = pgUpdate.addParamInt('accumulationMax', QUALITY_SETTINGS[QUALITY_DEFAULT].accumulationMax, 8, 2048, { ...params2, step:8, callback: setResolveAccumulationMax });
    setResolveAccumulationMax(this.accumulationMax.value);
    this.scene = scene;
    //the paths end at their first hit, in the lighting the lightmap has there, instead of tracing bounceCount bounces
    pgUpdate.addParamBool('cacheBounces', true, params2);
    //coarser levels that accumulate the samples too. The resolve needs 2 to blend between levels
    pgUpdate.addParamInt('ancestorLevels', 2, 0, 4, params2);
    //the paths of the coarser lighting a new texel starts with, instead of starting from its first noisy sample
    pgUpdate.addParamFloat('seedCount', 32, 0, 512, params2);
    //the frames a tile has to be unused before another tile can take its place: when the lightmap has no room left, or
    //none near the first slot of the new tile (see EVICT_NEAR in updateVirtualLightmap.wgsl)
    //While the camera stays where it is, so also when it only rotates: see evictDistance. The uniform is set in update
    this.evictFrames = pgUpdate.addParamFloat('evictFrames', 30, 1, 1000);
    //the distance in m the camera has to go before the tiles it saw from where it was can make place sooner than after
    //evictFrames frames, down to evictFramesMin: what was seen from somewhere else is at another level now, or behind
    //the camera. Without it a camera that moves fast fills the lightmap within evictFrames frames, the new tiles then
    //find no tile to replace near their first slot and the claim pass stays up to 2.5 times slower. See getEvictFrames
    this.evictDistance = pgUpdate.addParamFloat('evictDistance', 2, .1, 50);
    //a new tile takes the place of the first unused tile of its probe sequence, instead of the one that was unused the
    //longest. The claim pass of a full lightmap is then as fast as that of an empty one (about 1.2 times faster than
    //without this), but which of the tiles that are out of view goes no longer follows how long they are. A change
    //compiles the update shaders again, as it is a constant there. See findOrInsert in updateVirtualLightmap.wgsl
    this.evictFirstParam = pgUpdate.addParamBool('evictFirst', false);
    //the frames of light a texel holds while the light changes (the day cycle), instead of accumulationMax paths: lower
    //follows the light closer, higher is less noisy. Also the frames a tile can be out of view before its lighting
    //counts for less when it comes back. See getChangingCountMax and markUsed in updateVirtualLightmap.wgsl
    this.changeFrames = pgUpdate.addParamFloat('changeFrames', 30, 4, 120, params2);
    //how far a model that is dragged changes the lighting: times the part of the hemisphere of a texel it covers. The
    //texels there get the short history of changeFrames, the rest of the scene stays as it is. Higher reaches further
    //but is more noise around the model, 0 is only the model itself and its sun shadow. See getDragInfluence
    pgUpdate.addParamFloat('dragInfluence', 2, 0, 16, params2);
    //true: a model that is dragged only changes the lighting where it has influence, see dragInfluence. False: everywhere,
    //so every texel in view follows the drag and every texel starts over after it. Slower and more noise, but it misses
    //nothing: the influence is an estimate, light that bounces more than once from what the model changed is not in it
    pgUpdate.addParamBool('dragLocal', true, params2);
    pgUpdate.collapsed = true;

    const pgResolveParams = { materials: [this.resolveMaterial] };
    const pgResolve = pg.addGroup('virtualLightmapResolve');
    //the paths the coarser level counts as in the lighting shown, so texels with few paths show the smoother coarser lighting.
    //Less as the texels get more paths, none from 8 times this or from 3/4 of accumulationMax: see combine and
    //getPriorEnd in resolveVirtualLightmap.wgsl
    pgResolve.addParamFloat('ancestorWeight', 64, 0, 512, pgResolveParams);
    //a pixel reads 1 of the 2 levels around its level, picked by noise, instead of blending both: the TAA pass averages
    //it to the same blend, so only when that pass runs (see draw). About 1.5 times faster, see getLightingStochastic in
    //resolveVirtualLightmap.wgsl
    this.stochasticLevels = pgResolve.addParamBool('stochasticLevels', true);
    //the pixels no texel covers in red
    pgResolve.addParamBool('showHoles', false, pgResolveParams);
    pgResolve.addParamBool('showLevels', false, pgResolveParams);
    //the tiles in a color each, their texels as a checkerboard
    pgResolve.addParamBool('showTiles', false, pgResolveParams);
    //the pixels of texels that still take samples, that hold less than accumulationMax, in yellow
    pgResolve.addParamBool('showConverging', false, pgResolveParams);
    //the lightmap itself instead of the scene, as a square with black borders next to it. It doesn't move with the camera,
    //so it gets no TAA pass, see usesTAAPass in IrradianceCacheDraw
    this.showLightmap = pgResolve.addParamBool('showLightmap', false, pgResolveParams);
    pgResolve.collapsed = true;

    //the layout of a shader has every binding of the source, so the passes that don't trace need them too
    this.updateShaders.forEach(s => scene.bindPathTrace(s));
    scene.bind(this.resolveMaterial);
  }

  //through the params, so the ui follows and their callbacks run. The lightmapSize takes effect in update
  private applyQuality(setting:QualitySetting) {
    this.sizeParam.setValue(SIZE_OPTIONS.indexOf(setting.lightmapSize));
    this.texelsPerPixel.setValue(setting.texelsPerPixel);
    this.sampleCount.setValue(setting.sampleCount);
    this.referenceCellSize.setValue(setting.referenceCellSize);
    this.traceSamplesMax.setValue(setting.traceSamplesMax);
    this.accumulationMax.setValue(setting.accumulationMax);
  }

  public update(gBuffer:RenderTexture, camera:Camera) {
    //here and not in the callback of sizeParam, so the update and the draw of a frame use the same buffers
    const size = SIZE_OPTIONS[this.sizeParam.value];
    if(size != this.size){
      this.size = size;
      this.createLightmapBuffers();
    }
    if(this.reset){
      this.renderer.clearBuffer(this.tileBuffer);
      this.renderer.clearBuffer(this.tileTimeBuffer);
      this.renderer.clearBuffer(this.lightingBuffer);
      this.renderer.clearBuffer(this.tileCountBuffer);
      this.renderer.clearBuffer(this.claimBuffer);
      this.renderer.clearBuffer(this.traceArgsBuffer);
      this.refreshFrames = 0;
      this.lightChangeStart = 0;
      this.lightChangeEnd = 0;
      this.sunChanging = false;
      this.dragStrokes.length = 0;
      this.dragStrokeActive = false;
      this.cameraHistory.length = 0;
      this.reset = false;
    }

    if(this.evictFirstParam.value != this.evictFirst){
      this.evictFirst = this.evictFirstParam.value;
      this.updateShaders.forEach(s => s.setConstant('EVICT_FIRST', this.evictFirst));
    }
    const cellSize = this.getCellSize(gBuffer);
    if(cellSize != this.cellSize){
      this.cellSize = cellSize;
      this.updateShaders.forEach(s => s.setConstant('CELL_SIZE', cellSize));
    }
    const countX = Math.ceil(gBuffer.width / (this.workGroupSizeX * this.cellSize));
    const countY = Math.ceil(gBuffer.height / (this.workGroupSizeY * this.cellSize));
    const cellCount = countX * this.workGroupSizeX * countY * this.workGroupSizeY;
    if(!this.traceCellsBuffer || this.traceCellsBuffer.size < cellCount * 8){
      this.traceCellsBuffer?.destroy();
      this.traceCellsBuffer = this.renderer.createBuffer(cellCount * 8, GPUBufferUsage.STORAGE);
      this.updateShaders.forEach(s => s.setBuffer('traceCells', this.traceCellsBuffer!));
    }

    //the sun only: a dragged model is local, see updateDrag
    const sunChanging = this.scene.sunChanging;
    const frame = this.renderer.time.frameNumber;
    if(sunChanging){
      if(!this.sunChanging) this.lightChangeStart = frame;
      this.lightChangeEnd = frame;
    }
    this.updateDrag(frame);
    const evictFrames = this.getEvictFrames(camera, frame);
    const traceLanes = this.getTraceLanes(this.sampleCount.value);
    const traceCellsMax = Math.max(1, Math.floor(this.traceSamplesMax.value * 1000 / this.sampleCount.value));
    this.updateShaders.forEach(s => {
      s.setFloat('traceCellsMax', traceCellsMax);
      s.setFloat('evictFrames', evictFrames);
      s.setFloat('lightChangeStart', this.lightChangeStart);
      s.setFloat('lightChangeEnd', this.lightChangeEnd);
      s.setFloat('pixelScale', gBuffer.height / this.referenceHeight);
      s.setVector3('cameraPosition', camera.worldPosition);
      s.setMatrix('inverseView', camera.viewMatrix.inverse);
      s.setMatrix('inverseProjection', camera.projectionMatrix.inverse);
      s.setFloat('frame', this.renderer.time.frameNumber);
      s.setFloat('lightChanging', sunChanging ? 1 : 0);
      s.setFloat('traceLanes', traceLanes);
      s.setFloat('pixelSize', this.getPixelSize(camera, gBuffer));
      s.setVector2('resolution', gBuffer.size);
      s.setTexture('normalDepthTexture', gBuffer.views[1]);
      s.setTexture('modelTexture', gBuffer.views[2]);
    });

    if(sunChanging){
      this.refreshFrames = 0;
    }else if(this.sunChanging){
      //the light stopped: the resolved texels take samples again, of the light as it is now
      this.refresh(false);
      //and once more when the texels in view are resolved again. Twice the frames that takes at a sample per frame: the
      //random pixel of a cell is not in the same texel every frame
      this.refreshFrames = Math.ceil(this.accumulationMax.value * (1 - this.refreshFraction) / this.sampleCount.value) * 2;
    }else if(this.refreshFrames > 0 && --this.refreshFrames == 0){
      this.refresh(true);
    }
    this.sunChanging = sunChanging;

    //the args of the trace pass and the wanted count of this frame. The one of the previous frame stays, for select_main
    const wantedIndex = 4 + (frame & 1);
    this.renderer.clearBuffer(this.traceArgsBuffer, 0, 16);
    this.renderer.clearBuffer(this.traceArgsBuffer, wantedIndex * 4, 4);
    this.renderer.dispatchCompute(this.claimShader, countX, countY, 1);
    this.renderer.dispatchCompute(this.selectShader, countX, countY, 1);
    this.renderer.copyBufferToBuffer(this.traceArgsBuffer, this.traceIndirectBuffer, 0, 0, 16);
    this.renderer.dispatchComputeIndirect(this.updateShader, this.traceIndirectBuffer);

    //the used slots of the lightmap
    if(this.renderer.time.frameNumber % 30 == 0 && !this.tileCountReading){
      this.tileCountReading = true;
      const cellCount = countX * this.workGroupSizeX * countY * this.workGroupSizeY;
      Promise.all([this.renderer.readBuffer(this.tileCountBuffer, Uint32Array), this.renderer.readBuffer(this.traceArgsBuffer, Uint32Array)]).then(([[count], traceArgs]) => {
        //the cells that traced, and the ones that wanted to: 1 per texel that is still converging. Less traced when
        //that is over traceSamplesMax
        this.tileCountLog.setValue(`tiles: ${count} / ${this.tileCount} (${(count / this.tileCount * 100).toFixed(1)}%), tracing: ${Math.min(traceArgs[3], traceCellsMax)} of ${traceArgs[wantedIndex]} / ${cellCount} cells`);
        this.tileCountReading = false;
      });
    }
  }

  //temporal: a TAA pass follows, which averages the noise of stochasticLevels
  public draw(camera:Camera, gBuffer:RenderTexture, destination:RenderTexture | null = null, temporal = false) {
    this.resolveMaterial.setFloat('stochasticLevel', temporal && this.stochasticLevels.value ? 1 : 0);
    this.resolveMaterial.setFloat('frame', this.renderer.time.frameNumber);
    this.resolveMaterial.setVector3('cameraPosition', camera.worldPosition);
    this.resolveMaterial.setMatrix('inverseView', camera.viewMatrix.inverse);
    this.resolveMaterial.setMatrix('inverseProjection', camera.projectionMatrix.inverse);
    this.resolveMaterial.setVector2('resolution', gBuffer.size);
    this.resolveMaterial.setFloat('pixelSize', this.getPixelSize(camera, gBuffer));
    this.resolveMaterial.setTexture('normalDepthTexture', gBuffer.views[1]);
    this.resolveMaterial.setTexture('albedoTexture', gBuffer.views[0]);
    this.resolveMaterial.setTexture('modelTexture', gBuffer.views[2]);
    this.renderer.blit(null, destination, this.resolveMaterial);
  }

  /*
  a model that is dragged (Scene.movingModel) changes the lighting around it and in its sun shadow, not everywhere like
  the sun. Every drag is a stroke: a capsule around where the model went, from the start of the drag to its end.
  -during the drag the texels in view in the influence of its stroke get a short history, see getCountMax in
   updateVirtualLightmap.wgsl. The rest of the scene stays resolved
  -after the drag every texel in the influence of the stroke is reset, the first time a cell uses it again: now when
   it is in view, else when it comes back. Until then it is no cached lighting for a path. See resetStaleTexel
   */
  private updateDrag(frame:number) {
    const moving = this.scene.movingModel;
    if(moving){
      const m = moving.transform.worldMatrix.m;
      const center = new Vector3(m[12], m[13], m[14]);
      this.dragModelIndex = this.scene.models.indexOf(moving);
      this.dragModelRadius = this.scene.getBoundingRadius(moving);

      if(!this.dragStrokeActive){
        this.dragStrokes.unshift({points: [center], from: center, to: center, radius: 0, model: 0, endFrame: 0});
        if(this.dragStrokes.length > this.dragStrokeCount) this.mergeOldestStrokes();
        this.dragStrokeActive = true;
      }
      const stroke = this.dragStrokes[0];
      const points = stroke.points;
      //not every frame: only when the model is somewhere else
      if(Vector3.distance(points[points.length - 1], center) > this.dragModelRadius * .25) points.push(center);
      stroke.to = center;
      stroke.radius = this.dragModelRadius + this.getDeviation(points, stroke.from, center);
      stroke.model = this.dragModelIndex;
      stroke.endFrame = frame;
    }else{
      this.dragStrokeActive = false;
    }

    const strokes = this.dragStrokes;
    const active = this.dragStrokeActive ? strokes[0] : null;
    //the frame before which a texel has to be sampled last to be reset: any frame of a drag that is over, some frames
    //ago for the one that goes on
    const staleFrames = Math.ceil(this.changeFrames.value * this.dragStaleFactor);
    const resetFrames = strokes.map(stroke => stroke === active ? Math.max(frame - staleFrames, 0) : stroke.endFrame + 1);
    const models = new Vector4(strokes[0]?.model ?? 0, strokes[1]?.model ?? 0, strokes[2]?.model ?? 0, strokes[3]?.model ?? 0);
    this.updateShaders.forEach(s => {
      for(let i = 0; i < this.dragStrokeCount; i++){
        const stroke = strokes[i];
        //a reset frame of 0 is no stroke
        s.setVector4('dragStrokeA' + i, stroke ? new Vector4(stroke.from.x, stroke.from.y, stroke.from.z, stroke.radius) : Vector4.ZERO);
        s.setVector4('dragStrokeB' + i, stroke ? new Vector4(stroke.to.x, stroke.to.y, stroke.to.z, resetFrames[i]) : Vector4.ZERO);
      }
      s.setVector4('dragStrokeModels', models);
      s.setFloat('dragStrokeEnd', Math.max(0, ...resetFrames));
      s.setFloat('dragState', active ? 1 : 0);
      s.setFloat('dragModel', this.dragModelIndex);
      s.setFloat('dragRadius', active ? active.radius : 0);
      s.setVector3('dragFrom', active ? active.from : Vector3.ZERO);
      s.setVector3('dragTo', active ? active.to : Vector3.ZERO);
    });
  }

  /*
  the evictFrames of the shaders: the frames a tile has to be unused before another tile can take its place.
  The frames since the camera was last further than evictDistance from where it is now, between evictFramesMin and the
  evictFrames param. So a tile can go when it was last used from more than evictDistance away:
  -a camera that rotates in place keeps what it saw for evictFrames frames, it is likely to look there again
  -a camera that moves leaves the tiles of where it was behind. Those are of little use from where it is now: the level
   of a surface follows the distance to the camera, so the same surface gets other tiles
   */
  private getEvictFrames(camera:Camera, frame:number) {
    const position = camera.worldPosition.clone();
    const history = this.cameraHistory;
    const framesMax = Math.ceil(this.evictFrames.value);
    while(history.length > 0 && frame - history[0].frame > framesMax) history.shift();
    let evictFrames = framesMax;
    for(let i = history.length - 1; i >= 0; i--){
      if(Vector3.distance(history[i].position, position) > this.evictDistance.value){
        evictFrames = frame - history[i].frame;
        break;
      }
    }
    history.push({frame, position});
    return Math.min(Math.max(evictFrames, this.evictFramesMin), framesMax);
  }

  //the 2 oldest strokes become 1, a sphere around both: every grab of an axis of the pivot is a stroke, so there are soon
  //more than the shaders take. Dropping the oldest would leave the tiles that are out of view since then with their old
  //lighting. The sphere is larger than the 2 strokes, so it resets more than needed, never less
  private mergeOldestStrokes() {
    const b = this.dragStrokes.pop()!;
    const a = this.dragStrokes.pop()!;
    const ends: [Vector3, number][] = [[a.from, a.radius], [a.to, a.radius], [b.from, b.radius], [b.to, b.radius]];
    const center = new Vector3();
    for(const [p] of ends) center.add(p);
    center.multiplyScalar(1 / ends.length);
    const radius = Math.max(...ends.map(([p, r]) => Vector3.distance(p, center) + r));
    //a was the newer one: its end frame, as a tile has to be reset when it missed either
    this.dragStrokes.push({points: [center], from: center, to: center, radius, model: a.model, endFrame: a.endFrame});
  }

  //how far the points are from the line from from to to at most. The path of a model is not straight when it was dragged
  //along more than 1 axis: the capsule around it is wider by this
  private getDeviation(points:Vector3[], from:Vector3, to:Vector3) {
    if(points.length <= 2) return 0;
    const ab = to.clone().subtract(from);
    const lengthSquared = Math.max(Vector3.dot(ab, ab), 1e-8);
    let deviation = 0;
    for(const p of points){
      const ap = p.clone().subtract(from);
      const t = Math.min(Math.max(Vector3.dot(ap, ab) / lengthSquared, 0), 1);
      deviation = Math.max(deviation, ap.subtract(ab.clone().multiplyScalar(t)).length());
    }
    return deviation;
  }

  //see refresh_main in updateVirtualLightmap.wgsl. second: the texels that took no samples since the first refresh go
  //back further
  private refresh(second:boolean) {
    const count = Math.max(1, Math.floor(this.accumulationMax.value * this.refreshFraction));
    this.refreshShader.setFloat('refreshCount', count);
    this.refreshShader.setFloat('refreshCountIdle', second ? Math.max(1, Math.floor(count * this.refreshFraction)) : count);
    this.renderer.dispatchCompute(this.refreshShader, this.size / this.workGroupSizeX, this.size / this.workGroupSizeY, 1);
  }

  //the threads of trace_main per cell: a power of 2, so it divides the workgroup. A lane traces every lanes-th path, so
  //with a sampleCount that is no multiple of the lanes some lanes have a path less, or none, and the threads of a wave
  //wait on the others: 12 paths on 8 lanes are 2 rounds in which 12 of the 16 threads trace, on 4 lanes 3 full rounds.
  //Fewer lanes is more paths one after the other in a thread though, which a frame with few cells to trace waits on.
  //So the most lanes that are within 1/8 of the best occupancy
  private getTraceLanes(sampleCount:number) {
    const occupancy = (lanes:number) => sampleCount / (lanes * Math.ceil(sampleCount / lanes));
    const options = [];
    for(let lanes = this.traceLanesMax; lanes >= 1; lanes /= 2) options.push(lanes);
    const best = Math.max(...options.map(occupancy));
    return options.find(lanes => occupancy(lanes) >= best - 1 / 8)!;
  }

  //referenceCellSize at the height of the screen, in whole pixels: at 4 that is 4 at 1080p, 5 at 1440p, 8 at 4K. A
  //constant of the shaders, so a resize that changes it compiles them again
  private getCellSize(gBuffer:RenderTexture) {
    return Math.max(1, Math.round(this.referenceCellSize.value * gBuffer.height / this.referenceHeight));
  }

  //the world size of a pixel at distance 1, of a screen referenceHeight pixels high: not of the real screen, so the texels
  //and the tiles they take don't follow the resolution. Only the FOV and the aspect ratio change what is in view
  private getPixelSize(camera:Camera, _gBuffer:RenderTexture) {
    return Math.tan(camera.projection.FOV * .5) * 2 / this.referenceHeight;
  }
}
