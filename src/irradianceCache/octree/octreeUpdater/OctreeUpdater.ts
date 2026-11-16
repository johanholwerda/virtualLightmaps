import ComputeShader from "webGPU-framework/renderer/material/ComputeShader";
import Renderer from "webGPU-framework/renderer/render/Renderer";
import ParamGroup from "webGPU-framework/utils/params/ParamGroup";
import RenderTexture from "webGPU-framework/renderer/texture/RenderTexture";
import updateOctree from './updateOctree.wgsl?raw';
import collapseOctree from './collapseOctree.wgsl?raw';
import hash from 'webGPU-framework/shaders/hash.wgsl?raw';
import rayTraceScene from "../../includes/rayTraceScene.wgsl?raw";
import pathTrace from "../../includes/pathTrace.wgsl?raw";
import Scene from "../../scene/Scene";
import Camera from "webGPU-framework/renderer/camera/Camera";
import OctreeRayMarch from "../octreeRayMarch/OctreeRayMarch";
import ParamBool from "webGPU-framework/utils/params/ParamBool";
import OctreeRayResolve from "../octreeResolve/OctreeRayResolve";
import OctreeSplat from "../octreeSplat/OctreeSplat";
import OctreeDebug from "../octreeDebug/OctreeDebug";
import octreeNode from '../../includes/octreeNode.wgsl?raw';
import ParamLog from "webGPU-framework/utils/params/ParamLog";
import ParamFloat from "webGPU-framework/utils/params/ParamFloat";
import ParamInt from "webGPU-framework/utils/params/ParamInt";
import normalPacking from 'webGPU-framework/shaders/normalPacking.wgsl?raw';
import gBufferPacking from '../../includes/gBuffer.wgsl?raw';
import cameraRay from '../../includes/cameraRay.wgsl?raw';
import r2 from 'webGPU-framework/shaders/r2.wgsl?raw';

/*
steps:
-g-buffer is divided into 2d cells
-each cell samples the octree at the 3d position of the center of the cell
-if the octree leaf node level is too coarse for the depth of the pixel, subdivide the octree and add a node to a list for ray tracing
-an octree node has:
  -3d position of sample within cell? 4 bytes
  -normal of sample? 4 bytes - this is very little
  -color of sample: 4 bytes
  -index to children: 4 bytes. This might be too little: 16M nodes
  -world position of sample, node size: 8 bytes
  ----------------------
  24 bytes
  ----------------------

  what is needed for a leaf node?
  -radiance 8 bytes
  -normal of sample 4 bytes
  -world position of sample, node size: 8 bytes
  -albedo?: 4 bytes
  ----------------------------------
   24 bytes. Nodes in 256MB buffer: 12M leaf nodes max in 256MB

  what is needed for a node?
  -index to children or leaf node: 31 bits. index is 0 for an empty leaf node
  -isLeafNode: 1 bit
  -------------------
  4 bytes

  problem: thin surfaces in the distance. If a node is bigger than a wall thickness, only one of the 2 sides would be stored in it
-divide the node when it encounters a normal which is very different from the current one
-problem: it will divide the octree a lot in places with lots of geometric detail. Even in the distance
-drawing would need to happen with splatting?

  how will nodes be collapsed? (because they are too small for their distance)
  run a thread per node. Node gets the average lighting of the samples in it. If the node is too small, it becomes a leaf node
  can't average the normals of the leaf nodes - 2 opposite normals would cancel each other out
  the node first child index is added to an available first child index list.

  //TODO: lighting needs a lot of bits. Give leaf nodes a separate buffer for lighting? - adds complexity

  //TODO: get coverage from splat shader -> reproject -> use pixel with least coverage to create nodes

  How to initialize a low depth (low detail) octree? Shooting rays from random cameras?
  -areas will be missed
  -or: during lighting, when a ray does not hit anything, save the ray origin and ray direction into a buffer to be used later, then return
  -run a compute shader to create nodes from the ray origin and direction using scene ray trace. Store the albedo, normal, etc

  -when a ray does hit the octree, cast a ray into the scene and check the normal against the scene normal.
   If it is too different, split the node. This will throw away the lighting information of the previous leaf :(
   This is expensive. Should not be done continuously

 */
export enum OctreeDrawMode {
  SPLAT,
  RESOLVE,
  RAY_MARCH,
  DEBUG,
}

export default class OctreeUpdater {
  private renderer: Renderer;
  public updateShader: ComputeShader;
  public octreeBuffer: GPUBuffer;
  public lastWrittenIndex: GPUBuffer;
  public lightingQueueBuffer: GPUBuffer;
  public lightingQueueCount: GPUBuffer;
  public nodeSplitCount: GPUBuffer;
  public lightingBuffer: GPUBuffer;
  //the blocks of 8 children that collapseShader freed, reused by the splits of updateShader
  public freeBlocks: GPUBuffer;
  public freeBlockCount: GPUBuffer;
  //freeBlockCount after the update, for collapseShader. See isUnderPressure in collapseOctree.wgsl
  private freeBlockCountSnapshot: GPUBuffer;
  //per node: the frame a g-buffer sample last landed in it, so collapseShader can free the nodes that are offscreen or occluded
  public lastSeen: GPUBuffer;
  public collapseShader: ComputeShader;
  private readonly collapseWorkGroupSize = 256;
  private collapse!: ParamBool;
  //the draw modes use it to skip the leaves that are too coarse for their distance
  public scale!: ParamFloat;
  private accumulationMax!: ParamFloat;
  private edgeNormalThreshold!: ParamFloat;
  private edgeSplitDepth!: ParamInt;
  public readonly octreeDepthMax: number;
  //half the size of the root node, which is centered on the origin
  public readonly octreeRootNodeSize: number;
  public readonly nodeByteSize = 6 * 4;
  public readonly nodeCountMax = Math.pow(2, 22);
  //the slots of the feed hash map, 4 words each after the split counts in nodeSplitCount, see getFeedKey in updateOctree.wgsl
  public readonly feedHashMapSize = Math.pow(2, 20);
  //screen pixels along a side of the cells that each get a sample per frame, at a screen height of referenceHeight.
  //Small cells with few samples each: the same rays spread over more nodes, so each node gets samples more often
  private readonly referenceCellSize = 4;
  //the CELL_SIZE of the shader, in pixels of the screen as it is, see getCellSize
  private cellSize = 4;
  private readonly referenceHeight = 1080;
  private readonly workGroupSizeX = 8;
  private readonly workGroupSizeY = 8;
  private reset: boolean = true;
  private octreeRayResolve: OctreeRayResolve;
  private octreeRayMarch!: OctreeRayMarch;
  private octreeSplat!: OctreeSplat;
  private octreeDebug: OctreeDebug;
  private nodeCountLog!: ParamLog;
  private nodeCountReading = false;

  public constructor(renderer: Renderer, octreeRootNodeSize: number) {
    this.renderer = renderer;
    this.octreeRootNodeSize = octreeRootNodeSize;
    //the smallest nodes are 16 / 1024 at a root size of 8, whatever the root size
    this.octreeDepthMax = 10 + Math.round(Math.log2(octreeRootNodeSize / 8));

    this.updateShader = new ComputeShader(this.renderer, 'updateOctree', updateOctree, {includes: [hash, octreeNode, rayTraceScene, pathTrace, r2, normalPacking, gBufferPacking, cameraRay]});
    this.updateShader.setConstant('WORK_GROUP_SIZE_X', this.workGroupSizeX);
    this.updateShader.setConstant('WORK_GROUP_SIZE_Y', this.workGroupSizeY);
    this.updateShader.setConstant('CELL_SIZE', this.cellSize);
    this.updateShader.setConstant('OCTREE_ROOT_NODE_SIZE', this.octreeRootNodeSize);
    this.updateShader.setConstant('OCTREE_DEPTH_MAX', this.octreeDepthMax);
    this.updateShader.setConstant('FEED_HASHMAP_SIZE', this.feedHashMapSize);

    this.octreeBuffer = this.renderer.createBuffer(this.nodeCountMax *  this.nodeByteSize, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);
    this.updateShader.setBuffer('octreeBuffer', this.octreeBuffer);

    this.lightingQueueBuffer = this.renderer.createBuffer(1024 * 1024 * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);
    this.lightingQueueCount = this.renderer.createBuffer(4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);

    this.lastWrittenIndex = this.renderer.createBuffer(4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);
    this.updateShader.setBuffer('lastWrittenIndex', this.lastWrittenIndex);

    //per node, then the feed hash map: in 1 buffer as updateShader is at the limit of storage buffers
    this.nodeSplitCount = this.renderer.createBuffer((this.nodeCountMax + this.feedHashMapSize * 4) * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);
    this.updateShader.setBuffer('nodeSplitCount', this.nodeSplitCount);

    //rgb lighting + sample count as 4 x f16 per node
    this.lightingBuffer = this.renderer.createBuffer(this.nodeCountMax * 8, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    this.updateShader.setBuffer('lightingBuffer', this.lightingBuffer);

    this.freeBlocks = this.renderer.createBuffer(this.nodeCountMax / 8 * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    this.updateShader.setBuffer('freeBlocks', this.freeBlocks);
    this.freeBlockCount = this.renderer.createBuffer(4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);
    this.updateShader.setBuffer('freeBlockCount', this.freeBlockCount);
    this.freeBlockCountSnapshot = this.renderer.createBuffer(4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    this.lastSeen = this.renderer.createBuffer(this.nodeCountMax * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    this.updateShader.setBuffer('lastSeen', this.lastSeen);

    this.collapseShader = new ComputeShader(this.renderer, 'collapseOctree', collapseOctree, {includes: [octreeNode]});
    this.collapseShader.setConstant('WORK_GROUP_SIZE', this.collapseWorkGroupSize);
    this.collapseShader.setConstant('OCTREE_ROOT_NODE_SIZE', this.octreeRootNodeSize);
    this.collapseShader.setConstant('OCTREE_DEPTH_MAX', this.octreeDepthMax);
    this.collapseShader.setBuffer('octreeBuffer', this.octreeBuffer);
    this.collapseShader.setBuffer('nodeSplitCount', this.nodeSplitCount);
    this.collapseShader.setBuffer('lightingBuffer', this.lightingBuffer);
    this.collapseShader.setBuffer('freeBlocks', this.freeBlocks);
    this.collapseShader.setBuffer('freeBlockCount', this.freeBlockCount);
    this.collapseShader.setBuffer('freeBlockCountSnapshot', this.freeBlockCountSnapshot);
    this.collapseShader.setBuffer('lastWrittenIndex', this.lastWrittenIndex);
    this.collapseShader.setBuffer('lastSeen', this.lastSeen);

    this.octreeRayMarch = new OctreeRayMarch(this.renderer, this.octreeDepthMax, this.octreeRootNodeSize);
    this.octreeRayResolve = new OctreeRayResolve(this.renderer, this.octreeDepthMax, this.octreeRootNodeSize);
    this.octreeSplat = new OctreeSplat(this.renderer, this.nodeCountMax, this.octreeDepthMax, this.octreeRootNodeSize);
    this.octreeDebug = new OctreeDebug(this.renderer, this.nodeCountMax, this.octreeDepthMax, this.octreeRootNodeSize);
  }

  public get exposureMaterials() {
    return [this.octreeRayResolve.exposureMaterial, this.octreeRayMarch.exposureMaterial, this.octreeSplat.exposureMaterial];
  }

  public init(paramGroup:ParamGroup, scene:Scene) {
    const pgParams = { materials: [this.updateShader] };
    const pg = paramGroup.addGroup('OctreeSplit');
    pg.addButton('reset', { callback: () => {
      this.reset = true;
    } });
    //the node size at a distance is proportional to octreeRootNodeSize / scale, so scale grows with the root to keep the node sizes
    const scaleFactor = this.octreeRootNodeSize / 8;
    this.scale = pg.addParamFloat('scale', 10000 * scaleFactor, 1, 20000 * scaleFactor, pgParams);
    //lighting samples per cell per frame, accumulated in the leaf node the cell hits
    pg.addParamInt('sampleCount', 8, 1, 64, pgParams);
    //frames the running average of a node holds. Lower reacts faster to changes, higher is less noisy
    this.accumulationMax = pg.addParamFloat('accumulationMax', 256, 1, 2048, pgParams);
    //nodes containing normals with a dot product below this are on a hard edge
    this.edgeNormalThreshold = pg.addParamFloat('edgeNormalThreshold', .8, -1, 1, pgParams);
    //extra octree levels for nodes on a hard edge, to make the bleeding across the edge thinner
    this.edgeSplitDepth = pg.addParamInt('edgeSplitDepth', 2, 0, 4, pgParams);
    //the paths end at their first hit, in the lighting the octree has there, instead of tracing bounceCount bounces
    pg.addParamBool('cacheBounces', true, pgParams);
    //the feed: world space cells the rays fill themselves, for the hit points the octree has no lighting for (what is
    //out of view). A path then ends at its first hit in the feed cell there, or feeds it with 1 more ray: at most 2
    //rays and a sun ray, and bounceCount is not used. See getIndirectPathSum in updateOctree.wgsl.
    //The paths a feed cell holds: a path ends in a cell with the chance of the part of them it has. Higher is less
    //noise in the bounce light of what is out of view, but takes longer to fill and follows changes slower. 0: no feed,
    //a path goes on for up to bounceCount bounces where the octree has no lighting
    pg.addParamInt('rayFeedPaths', 128, 0, 1024, { ...pgParams, step:8 });
    //the size of the feed cells in m. Larger fills faster and takes fewer slots, but the bounce light of what is out of
    //view is blurrier and leaks between parallel surfaces closer than a cell
    pg.addParamFloat('feedCellSize', .25, .02, 2, pgParams);
    //the part of the workgroups whose paths feed a feed cell that is full too, instead of ending in it: a full cell
    //keeps following the light around it as it fills, and as the sun moves. See feedRefreshing in updateOctree.wgsl
    pg.addParamFloat('feedRefresh', 1 / 16, 0, 1, pgParams);
    //the frames after which a feed cell no ray looked at can be taken by another one
    pg.addParamFloat('feedEvictFrames', 120, 1, 1000, pgParams);
    //ancestors of the leaf that accumulate its samples too. An ancestor covers ~4x the screen area per level, so it converges faster
    pg.addParamInt('ancestorLevels', 2, 0, 4, pgParams);
    //the samples the nearest ancestor counts as in the lighting shown for a leaf, so leaves with few samples show the smoother ancestor
    pg.addParamFloat('ancestorWeight', 8, 0, 64, pgParams);
    //the samples of the ancestor lighting a new leaf starts with, instead of starting from its first noisy sample
    pg.addParamFloat('seedCount', 4, 0, 64, pgParams);
    this.nodeCountLog = pg.addLog('nodes: 0');

    //frees the nodes that are finer than the view needs
    const pgCollapseParams = { materials: [this.collapseShader] };
    const pgCollapse = paramGroup.addGroup('OctreeCollapse');
    this.collapse = pgCollapse.addParamBool('collapse', true);
    //levels finer than needed that are kept, so zooming back in doesn't split them again
    pgCollapse.addParamInt('keepLevels', 1, 0, 4, pgCollapseParams);
    //levels coarser than needed for the nodes that are not seen (offscreen or occluded), once the nodes run out
    pgCollapse.addParamInt('offscreenLevels', 2, 0, 12, pgCollapseParams);
    //frames without a sample after which a node counts as not seen. The samples are 1 random pixel per cell, so a node
    //that covers a few pixels is missed for some frames while it is visible
    pgCollapse.addParamInt('unseenFrames', 60, 1, 600, pgCollapseParams);
    //the fraction of the blocks kept free once the end of the buffer is reached. The nodes that are not seen are only
    //collapsed below it, so looking back at something finds its nodes still there when the buffer wasn't full
    pgCollapse.addParamFloat('freeReserve', .02, 0, .2, pgCollapseParams);

    this.octreeSplat.init(paramGroup, scene);
    this.octreeRayResolve.init(paramGroup, scene);
    this.octreeRayMarch.init(pg, scene);
    this.octreeDebug.init(paramGroup);

    scene.bindPathTrace(this.updateShader);
  }

  public update(gBuffer:RenderTexture, camera:Camera) {
    if(this.reset){
      this.renderer.clearBuffer(this.octreeBuffer);
      this.renderer.clearBuffer(this.nodeSplitCount);
      this.renderer.clearBuffer(this.lightingBuffer);
      //the root node already exists
      this.renderer.writeBuffer(this.lastWrittenIndex,  new Uint32Array([1]));
      // this.renderer.clearBuffer(this.lastWrittenIndex);
      this.renderer.clearBuffer(this.freeBlockCount);
      this.renderer.clearBuffer(this.lastSeen);
      this.reset = false;
    }

    this.updateShader.setMatrix('viewProjection', camera.viewProjection);
    this.updateShader.setVector3('cameraPosition', camera.worldPosition);
    this.updateShader.setMatrix('inverseView', camera.viewMatrix.inverse);
    this.updateShader.setMatrix('inverseProjection', camera.projectionMatrix.inverse);
    this.updateShader.setFloat('frame', this.renderer.time.frameNumber);

    this.updateShader.setVector2('resolution', gBuffer.size);

    this.updateShader.setTexture('normalDepthTexture', gBuffer.views[1]);

    const cellSize = this.getCellSize(gBuffer);
    if(cellSize != this.cellSize){
      this.cellSize = cellSize;
      this.updateShader.setConstant('CELL_SIZE', cellSize);
    }
    const countX = Math.ceil(gBuffer.width / (this.workGroupSizeX * this.cellSize));
    const countY = Math.ceil(gBuffer.height / (this.workGroupSizeY * this.cellSize));
    this.renderer.dispatchCompute(this.updateShader, countX, countY, 1);

    if(this.collapse.value) {
      this.renderer.copyBufferToBuffer(this.freeBlockCount, this.freeBlockCountSnapshot);
      this.collapseShader.setVector3('cameraPosition', camera.worldPosition);
      this.collapseShader.setFloat('scale', this.scale.value);
      this.collapseShader.setFloat('accumulationMax', this.accumulationMax.value);
      this.collapseShader.setFloat('edgeNormalThreshold', this.edgeNormalThreshold.value);
      this.collapseShader.setFloat('edgeSplitDepth', this.edgeSplitDepth.value);
      this.collapseShader.setFloat('parity', this.renderer.time.frameNumber % 2);
      //the frame of updateShader, which writes it to lastSeen
      this.collapseShader.setFloat('frame', this.renderer.time.frameNumber);
      this.renderer.dispatchCompute(this.collapseShader, Math.ceil(this.nodeCountMax / this.collapseWorkGroupSize), 1, 1);
    }

    //the nodes in use: the ones ever allocated minus the free blocks. Splits that don't fit leave the node a leaf
    if(this.renderer.time.frameNumber % 30 == 0 && !this.nodeCountReading){
      this.nodeCountReading = true;
      Promise.all([this.renderer.readBuffer(this.lastWrittenIndex, Uint32Array), this.renderer.readBuffer(this.freeBlockCount, Int32Array)]).then(([allocated, free]) => {
        const count = Math.min(allocated[0], this.nodeCountMax) - free[0] * 8;
        const full = allocated[0] + 8 > this.nodeCountMax ? ' (end reached, reusing freed nodes)' : '';
        this.nodeCountLog.setValue(`nodes: ${count} / ${this.nodeCountMax} (${(count / this.nodeCountMax * 100).toFixed(1)}%)${full}`);
        this.nodeCountReading = false;
      });
    }

    // this.renderer.readBuffer(this.octreeBuffer, Float32Array, 'octreeBuffer',  true);
    // this.renderer.readBuffer(this.octreeBuffer, Uint32Array, 'octreeBuffer',  true);
    // this.renderer.readBuffer(this.lastWrittenIndex, Uint32Array, 'lastWrittenIndex',  true);
    // this.renderer.readBuffer(this.nodeSplitCount, Uint32Array, 'nodeSplitCount',  true);
  }

  //referenceCellSize at the height of the screen, in whole pixels: 4 at 1080p, 5 at 1440p, 8 at 4K. The node sizes only
  //follow the distance (see getOctreeDepth), so a node covers 4x the pixels at 4K: with a fixed cell size it got 4x the
  //samples per frame, and the update cost 4x as much. A constant of the shader, so a resize that changes it compiles it again
  private getCellSize(gBuffer:RenderTexture) {
    return Math.max(1, Math.round(this.referenceCellSize * gBuffer.height / this.referenceHeight));
  }

  public draw(drawMode:OctreeDrawMode, camera:Camera, gBuffer:RenderTexture, destination:RenderTexture| null = null){
    if(drawMode == OctreeDrawMode.SPLAT) {
      this.octreeSplat.draw(camera, this, gBuffer, destination);
    }else if(drawMode == OctreeDrawMode.RESOLVE){
      this.octreeRayResolve.draw(camera, gBuffer, this, destination);
    }else if(drawMode == OctreeDrawMode.RAY_MARCH){
      this.octreeRayMarch.draw(camera, this, gBuffer, destination);
    }else if(drawMode == OctreeDrawMode.DEBUG){
      this.octreeDebug.draw(camera, this, gBuffer, destination);
    }
  }
}
