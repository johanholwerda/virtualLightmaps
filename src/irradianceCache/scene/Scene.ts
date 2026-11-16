import Vector3 from "monks-webgpu/renderer/math/Vector3";
import Transform from "monks-webgpu/renderer/scene/Transform";
import Renderer from "monks-webgpu/renderer/render/Renderer";
import WebGPUPreLoader from "monks-webgpu/renderer/core/WebGPUPreLoader";
import ShaderBase from "monks-webgpu/renderer/material/ShaderBase";
import LogGL from "monks-webgpu/renderer/core/LogGL";
import ModelBVH from "./ModelBVH";
import BlueNoiseTexture2D16Bits from "monks-webgpu/utils/texture/BlueNoiseTexture2D16Bits";
import JSONLoader from "monks-webgpu/utils/loaders/JSONLoader";
import LightmapMesh from "./LightmapMesh";
import Ray from "monks-webgpu/renderer/math/Ray";

export enum PrimitiveType {
  SPHERE = 0,
  CUBE = 1,
  CYLINDER = 2,
  //the triangles of meshBuffer, see LightmapMesh
  MESH = 3,
}

export class Model{
  public primitiveType:PrimitiveType = PrimitiveType.SPHERE
  public color:Vector3 = new Vector3(1, 1, 1);
  public transform: Transform = new Transform();
}

//the building of createBuilding, in m. x is the width, z the length. The pillars stand on a grid of pillarSpacing (a tile)
//from the corner, so the sizes are multiples of the spacing and the block is centered on the grid like the building.
//An outer building of 12 x 9 tiles around an atrium, with an inner building of 6 x 3 tiles in the middle of it. Between them
//a ring of 2 tiles that is open to the sky. The camera goes around the inner building, it hides what was seen before
export const buildingLayout = {
  pillarSpacing: 4,
  width: 48,
  length: 36,
  floorHeight: 4,
  floorCount: 2,
  //the galleries around the atrium and around the block, with floors and a roof over them
  galleryDepth: 4,
  //the solid core of the inner building, from the ground to the roof
  blockWidth: 16,
  blockLength: 4,
};

/*
A building of about 1700 boxes, more surface than the irradiance caches can cover.
sceneBuffer holds the BVH models in BVH order, then the other models, then the sky.
The BVH models are the static axis aligned cubes: their box is the BVH leaf (see ModelBVH). The other static models and the
dynamic ones (so they can be animated) are traced one by one, and the sky when nothing else is hit.
Every model but the sky can be moved, see moveModel. The models keep their index, the virtual lightmap uses it
 */
export default class Scene {
  private renderer:Renderer;
  //the sky sphere encloses the building, see createSky
  private readonly skyRadius: number;
  //half the size of the cube around the origin that contains the scene, sky included. A power of 2, the root of the octree
  public readonly size: number;
  public models: Model[] = [];
  private staticModels: Model[] = [];
  //the static axis aligned cubes, in the order of the leaves of the first BVH. The first models of sceneBuffer
  private bvhModels: Model[] = [];
  //the world bounding boxes of bvhModels, see createBVH
  private bvhBoxMin: Vector3[] = [];
  private bvhBoxMax: Vector3[] = [];
  private bvhNodeData!: ArrayBuffer;
  private bvh!: ModelBVH;
  //a box moved and the BVH was only refitted, see moveModel and rebuildBVH
  private bvhRefitted = false;
  private dynamicModels: Model[] = [];
  private sky!: Model;
  public sunDirection = new Vector3(.75, 1, .5).normalize();
  //the sun and the sky move (see DayCycle) or a model is dragged (see SceneEditor): the accumulations keep taking samples
  //where they would stop at their maximum
  public lightChanging = false;
  //the part of lightChanging that changes the light everywhere: the sun and the sky move
  public sunChanging = false;
  //the model that is dragged (see SceneEditor), null when none is: it changes the light around it and in its shadow only,
  //see getDragInfluence in updateVirtualLightmap.wgsl
  public movingModel: Model | null = null;
  public sceneBuffer!:GPUBuffer;
  public bvhBuffer!:GPUBuffer;
  //model indices grouped by primitive type, for instanced drawing
  public modelIndicesByType!:GPUBuffer;
  //the triangle mesh of the MESH models, see LightmapMesh
  public meshBuffer!:GPUBuffer;
  public mesh!:LightmapMesh;
  private meshLoader: JSONLoader;
  public readonly instanceOffsets: number[] = [];
  public readonly instanceCounts: number[] = [];
  //the sky is a single instance after all types
  public skyInstanceOffset = 0;
  //the content of modelIndicesByType, see sortFrontToBack
  private instanceIndices!: Uint32Array;
  private modelDistances!: Float32Array;
  //the Model struct of rayTraceScene.wgsl: 2 matrices, the quaternion, color + type, scale + padding
  private floatsPerPrimitive = 16+16+4+4+4;
  private sceneData!: Float32Array;
  //the start of the random numbers of the path tracer per pixel, see getBlueNoise in pathTrace.wgsl
  private blueNoise: BlueNoiseTexture2D16Bits;
  private pathTraceShaders: ShaderBase[] = [];
  //the shaders that got a copy of sunDirection (bind) and of the sky color (bindSkyColor), updated by setSun
  private boundShaders = new Set<ShaderBase>();
  private skyColorShaders = new Set<ShaderBase>();

  constructor(renderer:Renderer, preloader:WebGPUPreLoader){
    this.renderer = renderer;
    const {width, length, floorHeight, floorCount} = buildingLayout;
    //past the farthest corner of the roof. That corner has to stay within SKY_RADIUS of rayTraceScene.wgsl (.99 of the
    //radius), or isSky takes it for the sky
    this.skyRadius = Math.hypot(width / 2, floorHeight * floorCount + .1, length / 2) + .5;
    this.size = Math.pow(2, Math.ceil(Math.log2(this.skyRadius)));
    this.blueNoise = new BlueNoiseTexture2D16Bits(renderer, preloader);
    //has lightmap UVs (aUV0), mirrored: see LightmapMesh.createCharts
    this.meshLoader = new JSONLoader(preloader, import('../data/BattleHelmet.json?url'));
  }

  //cameraPath: the random cubes stay clear of it, see RANDOM_CUBES_CLEAR_PATH of IrradianceCacheDraw
  public init(cameraPath:Vector3[] = []) {
    this.mesh = LightmapMesh.fromJSON(this.meshLoader.data);
    this.meshBuffer = this.renderer.createBuffer(this.mesh.data.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    this.renderer.writeBuffer(this.meshBuffer, this.mesh.data);

    this.createBuilding(cameraPath);

    const isBVHModel = (model:Model) => model.primitiveType == PrimitiveType.CUBE && this.isAxisAligned(model);
    //ordered by a first build, so the boxes that are close in the BVH are close in sceneBuffer. Their indices are the
    //leaves of the BVH, also of the ones made again when a box moves
    const bvhModels = this.staticModels.filter(isBVHModel);
    this.bvhModels = this.createBVH(bvhModels).order.map(i => bvhModels[i]);
    const buildStart = performance.now();
    const bvh = this.bvh = this.createBVH(this.bvhModels);
    const buildTime = performance.now() - buildStart;
    this.models = [...this.bvhModels, ...this.staticModels.filter(model => !isBVHModel(model)), ...this.dynamicModels, this.sky];
    //the model target of the gBuffer packs both into 16 bits, see packModelTriangle in gBuffer.wgsl
    if(this.models.length > 0x10000) throw new Error(`Scene: ${this.models.length} models, the gBuffer holds at most 65536`);
    if(this.mesh.triangleCount > 0x10000) throw new Error(`Scene: ${this.mesh.triangleCount} mesh triangles, the gBuffer holds at most 65536`);

    this.sceneData = new Float32Array(this.models.length * this.floatsPerPrimitive);
    this.sceneBuffer = this.renderer.createBuffer( this.sceneData.length * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    this.writeSceneBuffer();

    //the node count only depends on the box count, a BVH made again fits
    this.bvhNodeData = new ArrayBuffer(Math.max(bvh.nodeCount, 1) * ModelBVH.bytesPerNode);
    this.bvhBuffer = this.renderer.createBuffer(this.bvhNodeData.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    this.writeBVH(bvh);
    if(LogGL.ENABLED) LogGL.log('scene: models', this.models.length, 'BVH nodes', bvh.nodeCount, 'depth', bvh.depth, 'build', buildTime.toFixed(2), 'ms');
    if(bvh.nodeCount > 0xffff) LogGL.error('scene: more BVH nodes than the stack entries of rayTraceScene.wgsl can index');

    //the sky is left out of the spheres and goes last: it covers the whole screen and is drawn after everything else, see SceneRendererForward
    const skyIndex = this.models.length - 1;
    const indices: number[] = [];
    for(const type of Scene.instancedTypes){
      this.instanceOffsets[type] = indices.length;
      this.models.forEach((model, i) => {
        if(model.primitiveType == type && i != skyIndex) indices.push(i);
      });
      this.instanceCounts[type] = indices.length - this.instanceOffsets[type];
    }
    this.skyInstanceOffset = indices.length;
    indices.push(skyIndex);
    this.instanceIndices = new Uint32Array(indices);
    this.modelDistances = new Float32Array(this.models.length);
    this.modelIndicesByType = this.renderer.createBuffer(indices.length * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    this.renderer.writeBuffer(this.modelIndicesByType, this.instanceIndices);
  }

  //the closest model whose box is hit by the ray, not the sky. The box of a model is its -1 to 1 cube, of a mesh the box
  //around its triangles
  public pick(ray:Ray):Model | null {
    const origin = new Vector3();
    const direction = new Vector3();
    let closest = Infinity;
    let picked: Model | null = null;
    for(const model of this.models){
      if(model === this.sky) continue;
      //into the space of the model. The direction is not normalized, so the distance along it is the world distance
      const inverse = model.transform.worldMatrix.inverse;
      origin.copy(ray.origin).transform(inverse);
      direction.copy(ray.origin).add(ray.direction).transform(inverse).subtract(origin);
      const half = model.primitiveType == PrimitiveType.MESH ? this.mesh.halfExtents : Scene.unitHalfSize;
      //slabs
      let near = 0, far = closest;
      for(let axis = 0; axis < 3 && near <= far; axis++){
        const o = axis == 0 ? origin.x : axis == 1 ? origin.y : origin.z;
        const d = axis == 0 ? direction.x : axis == 1 ? direction.y : direction.z;
        const h = axis == 0 ? half.x : axis == 1 ? half.y : half.z;
        const t0 = (-h - o) / d, t1 = (h - o) / d;
        near = Math.max(near, Math.min(t0, t1));
        far = Math.min(far, Math.max(t0, t1));
      }
      if(near <= far){
        closest = near;
        picked = model;
      }
    }
    return picked;
  }

  private static readonly unitHalfSize = new Vector3(1, 1, 1);

  //the radius of a sphere around the world position of the model that contains it: of a sphere its longest axis, else
  //half the diagonal of its box
  public getBoundingRadius(model:Model):number {
    const m = model.transform.worldMatrix.m;
    const x = Math.hypot(m[0], m[1], m[2]), y = Math.hypot(m[4], m[5], m[6]), z = Math.hypot(m[8], m[9], m[10]);
    if(model.primitiveType == PrimitiveType.SPHERE) return Math.max(x, y, z);
    const half = model.primitiveType == PrimitiveType.MESH ? this.mesh.halfExtents : Scene.unitHalfSize;
    return Math.hypot(half.x * x, half.y * y, half.z * z);
  }

  //after the transform of the model changed: writes it to sceneBuffer, and refits the BVH when it is in it: the boxes
  //above it follow, the tree stays the same. Cheap enough for every frame of a drag, see rebuildBVH for after it. A BVH
  //model has to stay axis aligned
  public moveModel(model:Model){
    const index = this.models.indexOf(model);
    if(index < 0) return;
    this.writeModel(index);
    const o = index * this.floatsPerPrimitive;
    this.renderer.device.queue.writeBuffer(this.sceneBuffer, o * 4, this.sceneData, o, this.floatsPerPrimitive);

    if(index < this.bvhModels.length){
      this.getBounds(model, this.bvhBoxMin[index], this.bvhBoxMax[index]);
      this.bvh.refit(index);
      this.writeBVH(this.bvh);
      this.bvhRefitted = true;
    }
  }

  //makes the BVH again when it was refitted by moveModel, after a drag: a box that moved far is in the wrong part of the
  //refitted tree, which makes the boxes above it large
  public rebuildBVH(){
    if(!this.bvhRefitted) return;
    this.bvhRefitted = false;
    const buildStart = performance.now();
    this.bvh = new ModelBVH(this.bvhBoxMin, this.bvhBoxMax, Scene.bvhStackSize + 1);
    if(LogGL.ENABLED) LogGL.log('scene: BVH rebuilt in', (performance.now() - buildStart).toFixed(2), 'ms, depth', this.bvh.depth);
    this.writeBVH(this.bvh);
  }

  //sorts the instances of every primitive type front to back, so the depth test rejects the hidden fragments before they are shaded.
  //on the distance from the camera to the world bounding box of the model, so a large floor the camera stands on goes first
  public sortFrontToBack(cameraPosition:Vector3){
    const m = this.sceneData;
    const distances = this.modelDistances;
    const cx = cameraPosition.x, cy = cameraPosition.y, cz = cameraPosition.z;
    for(let i = 0; i < this.models.length; i++){
      //the world matrix is the first in the model data. The primitives fit in the -1 to 1 cube,
      //so the half size of the bounding box is the sum of the absolute matrix columns
      const o = i * this.floatsPerPrimitive;
      const dx = Math.max(Math.abs(m[o + 12] - cx) - (Math.abs(m[o]) + Math.abs(m[o + 4]) + Math.abs(m[o + 8])), 0);
      const dy = Math.max(Math.abs(m[o + 13] - cy) - (Math.abs(m[o + 1]) + Math.abs(m[o + 5]) + Math.abs(m[o + 9])), 0);
      const dz = Math.max(Math.abs(m[o + 14] - cz) - (Math.abs(m[o + 2]) + Math.abs(m[o + 6]) + Math.abs(m[o + 10])), 0);
      distances[i] = dx * dx + dy * dy + dz * dz;
    }
    const compare = (a:number, b:number) => distances[a] - distances[b];
    for(const type of Scene.instancedTypes){
      const offset = this.instanceOffsets[type];
      this.instanceIndices.subarray(offset, offset + this.instanceCounts[type]).sort(compare);
    }
    this.renderer.writeBuffer(this.modelIndicesByType, this.instanceIndices);
  }

  //BVH_STACK_SIZE of rayTraceScene.wgsl. A larger stack costs registers: 32 made updateOctree 1.09x slower
  private static readonly bvhStackSize = 16;

  //the types that are drawn instanced, in modelIndicesByType
  public static readonly instancedTypes = [PrimitiveType.SPHERE, PrimitiveType.CUBE, PrimitiveType.CYLINDER, PrimitiveType.MESH];

  //emissive, see isSky in rayTraceScene.wgsl
  public get skyColor():Vector3{
    return this.sky.color;
  }

  //binds what the ray tracer in rayTraceScene.wgsl needs
  public bind(shader:ShaderBase){
    shader.setBuffer('sceneBuffer', this.sceneBuffer);
    shader.setBuffer('bvhBuffer', this.bvhBuffer);
    shader.setBuffer('meshBuffer', this.meshBuffer);
    shader.setFloat('modelCount', this.models.length);
    shader.setFloat('bvhModelCount', this.bvhModels.length);
    shader.setVector3('lightDirection', this.sunDirection);
    this.boundShaders.add(shader);
  }

  //for the shaders that show the sky color where nothing is hit
  public bindSkyColor(shader:ShaderBase){
    shader.setVector3('skyColor', this.sky.color);
    this.skyColorShaders.add(shader);
  }

  //moves the sun and changes the color of the sky, in all the bound shaders and in sceneBuffer (the sky is emissive)
  public setSun(direction:Vector3, skyColor:Vector3){
    this.sunDirection.copy(direction);
    for(const shader of this.boundShaders) shader.setVector3('lightDirection', this.sunDirection);

    this.sky.color.copy(skyColor);
    for(const shader of this.skyColorShaders) shader.setVector3('skyColor', this.sky.color);
    //only the color of the sky, the last model. See writeSceneBuffer
    const o = (this.models.length - 1) * this.floatsPerPrimitive + 16 + 16 + 4;
    this.sceneData[o] = skyColor.x;
    this.sceneData[o + 1] = skyColor.y;
    this.sceneData[o + 2] = skyColor.z;
    this.renderer.device.queue.writeBuffer(this.sceneBuffer, o * 4, this.sceneData, o, 3);
  }

  //binds what the path tracer in pathTrace.wgsl needs, on top of bind
  public bindPathTrace(shader:ShaderBase){
    this.bind(shader);
    shader.setTexture('blueNoiseTexture', this.blueNoise);
    this.pathTraceShaders.push(shader);
  }

  //false: the path tracers use white noise, to compare with
  public setBlueNoise(enabled:boolean){
    for(const shader of this.pathTraceShaders) shader.setConstant('BLUE_NOISE', enabled);
  }

  //floors with pillars and outer walls, around an atrium that is open to the sky with a block in the middle. All boxes are
  //axis aligned, so they all go in the BVH
  private createBuilding(cameraPath:Vector3[]){
    const {pillarSpacing, width, length, floorHeight, floorCount, galleryDepth, blockWidth, blockLength} = buildingLayout;
    const stone = new Vector3(.8, .78, .72);
    const slab = .2;
    const wall = .2;
    //half sizes of the atrium, the inner building and its block
    const atriumX = width / 2 - galleryDepth, atriumZ = length / 2 - galleryDepth;
    const innerX = blockWidth / 2 + galleryDepth, innerZ = blockLength / 2 + galleryDepth;
    const blockX = blockWidth / 2, blockZ = blockLength / 2;

    //the ground, its top is at slab / 2
    this.addBox(new Vector3(0, 0, 0), new Vector3(width, slab, length), stone);

    for(let floor = 0; floor < floorCount; floor++){
      const y = floor * floorHeight;
      //the outer walls
      this.addBox(new Vector3(width / 2, y + floorHeight / 2, 0), new Vector3(wall, floorHeight, length), stone);
      this.addBox(new Vector3(-width / 2, y + floorHeight / 2, 0), new Vector3(wall, floorHeight, length), stone);
      this.addBox(new Vector3(0, y + floorHeight / 2, length / 2), new Vector3(width, floorHeight, wall), stone);
      this.addBox(new Vector3(0, y + floorHeight / 2, -length / 2), new Vector3(width, floorHeight, wall), stone);
      //the floor above, or the roof: a ring around the atrium. The long sides over the full length, the short sides between them
      this.addBox(new Vector3(width / 2 - galleryDepth / 2, y + floorHeight, 0), new Vector3(galleryDepth, slab, length), stone);
      this.addBox(new Vector3(-width / 2 + galleryDepth / 2, y + floorHeight, 0), new Vector3(galleryDepth, slab, length), stone);
      this.addBox(new Vector3(0, y + floorHeight, length / 2 - galleryDepth / 2), new Vector3(width - 2 * galleryDepth, slab, galleryDepth), stone);
      this.addBox(new Vector3(0, y + floorHeight, -length / 2 + galleryDepth / 2), new Vector3(width - 2 * galleryDepth, slab, galleryDepth), stone);
      //the floor above or the roof of the inner building, through the block. The roof has the hole of the Cornell box
      if(floor == floorCount - 1){
        const {holeX, holeZ, holeHalf} = Scene.cornellBox;
        this.addSlabWithHole(-innerX, innerX, -innerZ, innerZ, y + floorHeight, slab,
          holeX - holeHalf, holeX + holeHalf, holeZ - holeHalf, holeZ + holeHalf, stone);
      }
      else{
        this.addBox(new Vector3(0, y + floorHeight, 0), new Vector3(2 * innerX, slab, 2 * innerZ), stone);
      }
    }

    //on the grid, except in the open part of the atrium and in the block. Those on the edges of the block stay, half in
    //its walls like on the outer walls. One pillar over all floors, through the slabs
    for(let x = -width / 2; x <= width / 2; x += pillarSpacing){
      for(let z = -length / 2; z <= length / 2; z += pillarSpacing){
        const ax = Math.abs(x), az = Math.abs(z);
        const inAtrium = ax < atriumX && az < atriumZ;
        const inInner = ax <= innerX && az <= innerZ;
        const inBlock = ax < blockX && az < blockZ;
        if((!inAtrium || inInner) && !inBlock) this.addPillar(x, z, floorHeight, floorCount);
      }
    }

    //a ledge along the edges of the floors and the roof that face the atrium, 1 box per side over its full length. As high
    //as a pillar base (half the double one at a floor) and a third of its width. It sticks out of the slab edge less than
    //the bases do, so it ends inside them at the pillars on the edge. On the roof it is centered on the slab like on the
    //floors, so it stands above the roof and the top bases. The sides touch at the corners, they don't overlap
    const ledgeHeight = .5, ledge = .2;
    for(let floor = 1; floor <= floorCount; floor++){
      const y = floor * floorHeight;
      //the outer building, on the inner edge of its ring: a concave corner, the z sides between the x sides
      for(const s of [-1, 1]){
        this.addBox(new Vector3(s * (atriumX - ledge / 2), y, 0), new Vector3(ledge, ledgeHeight, 2 * atriumZ), stone);
        this.addBox(new Vector3(0, y, s * (atriumZ - ledge / 2)), new Vector3(2 * (atriumX - ledge), ledgeHeight, ledge), stone);
      }
      //the inner building, on the outer edge of its floor: a convex corner, the x sides over the corners
      for(const s of [-1, 1]){
        this.addBox(new Vector3(s * (innerX + ledge / 2), y, 0), new Vector3(ledge, ledgeHeight, 2 * (innerZ + ledge)), stone);
        this.addBox(new Vector3(0, y, s * (innerZ + ledge / 2)), new Vector3(2 * innerX, ledgeHeight, ledge), stone);
      }
    }

    //the block, from the ground to the underside of the roof. Not to its top, that z-fights with the roof
    const height = floorCount * floorHeight;
    this.addBox(new Vector3(0, height / 2, 0), new Vector3(blockWidth, height - slab, blockLength), stone);

    this.addCornellBox();

    //in the open ring, on either side of the inner building. The ground is at 0.1
    const sphere = new Model();
    sphere.transform.scaleScalar = 1;
    sphere.transform.position = new Vector3(-16, 1.1, -1);
    sphere.color = new Vector3(.9, .1, .1);
    sphere.primitiveType = PrimitiveType.SPHERE;
    this.dynamicModels.push(sphere);

    const blueCylinder = new Model();
    blueCylinder.transform.scale = new Vector3(.6, 1, .6);
    blueCylinder.transform.position = new Vector3(16, 1.1, 1);
    blueCylinder.color = new Vector3(.1, .1, .9);
    blueCylinder.primitiveType = PrimitiveType.CYLINDER;
    this.dynamicModels.push(blueCylinder);

    //in the open ring, where the sun reaches the ground: the buildings are 8 m high, so their shadows reach about 6 m to -x
    //and 4 m to -z (see sunDirection). Next to RANDOM_CUBES_CLEAR_PATH of IrradianceCacheDraw, not on it.
    //Axis aligned, so they go in the BVH with the building
    const cubes: [x:number, z:number, halfSize:number, color:Vector3][] = [
      //front, z 6 to 10 is in the sun
      [-3, 7.5, .6, new Vector3(1, .45, .05)],
      [-18.5, 8.5, .7, new Vector3(.95, .85, .1)],
      //back, z -14 to -10
      [-6, -12.6, .7, new Vector3(.1, .8, .2)],
      [5, -12.8, .9, new Vector3(.1, .3, .95)],
      //right, x 12 to 14
      [13.2, -3, .6, new Vector3(.9, .1, .7)],
      [13.2, 4, .7, new Vector3(.1, .8, .85)],
      //left, x -20 to -18, the shadow of the inner building covers the rest
      [-18.8, -3.5, .6, new Vector3(.9, .1, .1)],
      [-19, 3.5, .5, new Vector3(.5, .15, .9)],
    ];
    //a mesh with UV charts for the virtual lightmap, on the ground floor under the center of the Cornell
    //box. Static but not a cube, so it is traced on its own after the BVH
    const helmet = new Model();
    helmet.primitiveType = PrimitiveType.MESH;
    const helmetScale = .8;
    const helmetHalf = Math.max(this.mesh.halfExtents.x, this.mesh.halfExtents.z) * helmetScale;
    const helmetX = (Scene.cornellBox.minX + Scene.cornellBox.maxX) / 2;
    const helmetZ = (Scene.cornellBox.minZ + Scene.cornellBox.maxZ) / 2 + .5;
    helmet.transform.scaleScalar = helmetScale;
    helmet.transform.position = new Vector3(helmetX, .1 + this.mesh.halfExtents.y * helmetScale, helmetZ);
    helmet.color = new Vector3(.8, .8, .8);
    this.staticModels.push(helmet);

    const placed: [x:number, y:number, z:number, halfSize:number][] = cubes.map(([x, z, halfSize]) => [x, .1 + halfSize, z, halfSize]);
    //keep the helmet clear, as a cube around it
    placed.push([helmetX, 0, helmetZ, helmetHalf]);
    //light bleeding test, in front of and to the right of the start view: plates of .01 x 1 x 1 m on the ground, in the sun.
    //Red and green .02 m apart, blue 1 m from green. Kept clear of the random cubes, as a cube around them
    {
      const plateHalf = .005, plateZ = 8.5, plateY = .1 + .5;
      const redX = 3.5, greenX = redX + 2 * plateHalf + .02, blueX = greenX + 2 * plateHalf + 1;
      const plateSize = new Vector3(2 * plateHalf, 1, 1);
      this.addBox(new Vector3(redX, plateY, plateZ), plateSize, new Vector3(.9, .1, .1));
      this.addBox(new Vector3(greenX, plateY, plateZ), plateSize, new Vector3(.1, .9, .1));
      this.addBox(new Vector3(blueX, plateY, plateZ), plateSize, new Vector3(.1, .1, .9));
      placed.push([(redX + blueX) / 2, plateY, plateZ, Math.max((blueX - redX) / 2 + plateHalf, .5)]);
    }
    //keep the Cornell box clear, as a cube around it
    {
      const {minX, maxX, minZ, maxZ} = Scene.cornellBox;
      placed.push([(minX + maxX) / 2, floorHeight + floorHeight / 2, (minZ + maxZ) / 2, Math.max(maxX - minX, maxZ - minZ) / 2]);
    }
    const randomCubes = this.createRandomCubes(cameraPath, placed, cubes.length);
    for(const [x, z, halfSize, color] of cubes) randomCubes.push([x, .1 + halfSize, z, halfSize, color]);
    for(const [x, y, z, halfSize, color] of randomCubes){
      const cube = new Model();
      cube.primitiveType = PrimitiveType.CUBE;
      cube.transform.position = new Vector3(x, y, z);
      cube.transform.scaleScalar = halfSize;
      cube.color = color;
      this.staticModels.push(cube);
    }

    this.sky = this.createSky(this.skyRadius);
  }

  //randomly colored axis aligned cubes, half on the ground and half on the first floor, up to cubeCount with the placedCubeCount
  //ones. Clear of the walls, the pillars, the block, the placed cubes and the camera path. Seeded, so every run has the
  //same scene
  private createRandomCubes(cameraPath:Vector3[], placed:[x:number, y:number, z:number, halfSize:number][], placedCubeCount:number){
    const {pillarSpacing, width, length, floorHeight, galleryDepth, blockWidth, blockLength} = buildingLayout;
    const cubeCount = 32;
    //placed also holds the helmet and the Cornell box
    const count = cubeCount - placedCubeCount;
    const cubes: [x:number, y:number, z:number, halfSize:number, color:Vector3][] = [];
    const atriumX = width / 2 - galleryDepth, atriumZ = length / 2 - galleryDepth;
    const innerX = blockWidth / 2 + galleryDepth, innerZ = blockLength / 2 + galleryDepth;
    //mulberry32
    let seed = 12345;
    const random = () => {
      seed = (seed + 0x6D2B79F5) | 0;
      let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
    const distanceToSegment = (p:Vector3, a:Vector3, b:Vector3) => {
      const ab = b.clone().subtract(a);
      const t = Math.min(Math.max(Vector3.dot(p.clone().subtract(a), ab) / Math.max(Vector3.dot(ab, ab), 1e-6), 0), 1);
      return a.clone().add(ab.multiplyScalar(t)).subtract(p).length();
    };

    const gap = .1;
    for(let attempt = 0; cubes.length < count && attempt < 10000; attempt++){
      const floor = cubes.length % 2;
      const halfSize = .3 + random() * .4;
      const x = (random() * 2 - 1) * width / 2;
      const z = (random() * 2 - 1) * length / 2;
      //on the ground or the first floor, whose tops are at slab / 2 = 0.1 above the floor
      const y = floor * floorHeight + .1 + halfSize;
      const ax = Math.abs(x), az = Math.abs(z);
      const reach = halfSize + gap;

      //inside the outer walls
      if(ax + reach > width / 2 - .1 || az + reach > length / 2 - .1) continue;
      //outside the block
      if(ax - reach < blockWidth / 2 && az - reach < blockLength / 2) continue;
      //on the first floor: fully on the gallery around the atrium or on the floor of the inner building
      if(floor == 1){
        const onGallery = ax - reach > atriumX || az - reach > atriumZ;
        const onInner = ax + reach < innerX && az + reach < innerZ;
        if(!onGallery && !onInner) continue;
      }
      //clear of the nearest pillar, whose base is .6 wide. The grid starts at the corner of the building
      const px = Math.round((x + width / 2) / pillarSpacing) * pillarSpacing - width / 2;
      const pz = Math.round((z + length / 2) / pillarSpacing) * pillarSpacing - length / 2;
      if(Math.abs(x - px) < .3 + reach && Math.abs(z - pz) < .3 + reach) continue;
      //clear of the other cubes
      const overlaps = (ox:number, oy:number, oz:number, oh:number) =>
        Math.abs(x - ox) < halfSize + oh + .3 && Math.abs(y - oy) < halfSize + oh + .3 && Math.abs(z - oz) < halfSize + oh + .3;
      if(placed.some(([ox, oy, oz, oh]) => overlaps(ox, oy, oz, oh))) continue;
      if(cubes.some(([ox, oy, oz, oh]) => overlaps(ox, oy, oz, oh))) continue;
      //clear of the camera path
      const center = new Vector3(x, y, z);
      let nearPath = false;
      for(let i = 0; i + 1 < cameraPath.length && !nearPath; i++){
        nearPath = distanceToSegment(center, cameraPath[i], cameraPath[i + 1]) < halfSize * Math.SQRT2 + .8;
      }
      if(nearPath) continue;

      //a saturated color of a random hue
      const hue = random() * 6;
      const channel = (offset:number) => {
        const k = (offset + hue) % 6;
        return 1 - Math.max(Math.min(k, 4 - k, 1), 0);
      };
      const saturation = .7 + random() * .25, value = .85 + random() * .1;
      const color = new Vector3(channel(5), channel(3), channel(1)).multiplyScalar(saturation)
        .addScalar(1 - saturation).multiplyScalar(value);
      cubes.push([x, y, z, halfSize, color]);
    }
    if(cubes.length < count) LogGL.error('scene: placed', cubes.length, 'of', count, 'random cubes');
    return cubes;
  }

  //the Cornell box on the first floor of the inner building, in its south (+z) gallery between the pillars at x 0 and 4.
  //The block is its back wall, it is open to the atrium. Seen from there the left wall (-x) is red and the right wall (+x)
  //green. The roof of the inner building is its ceiling, with a hole in the middle that lets in the sun (see createBuilding)
  private static readonly cornellBox = (() => {
    const minX = 0, maxX = 4;
    const minZ = buildingLayout.blockLength / 2, maxZ = minZ + buildingLayout.galleryDepth;
    return {minX, maxX, minZ, maxZ, holeX: (minX + maxX) / 2, holeZ: (minZ + maxZ) / 2, holeHalf: .6};
  })();

  private addCornellBox(){
    const {floorHeight, floorCount} = buildingLayout;
    const {minX, maxX, minZ, maxZ} = Scene.cornellBox;
    const white = new Vector3(.8, .8, .8);
    const red = new Vector3(.75, .08, .06);
    const green = new Vector3(.12, .6, .1);
    const slab = .2, wall = .2;
    //on the first floor, whose top is at slab / 2 above floorHeight, up to the roof, the slab of the last floor
    const floorTop = floorHeight + slab / 2;
    const ceiling = floorCount * floorHeight - slab / 2;
    const wallHeight = ceiling - floorTop;
    const wallY = (floorTop + ceiling) / 2;
    const centerZ = (minZ + maxZ) / 2;

    //on the grid lines, between the pillars
    this.addBox(new Vector3(minX, wallY, centerZ), new Vector3(wall, wallHeight, maxZ - minZ), red);
    this.addBox(new Vector3(maxX, wallY, centerZ), new Vector3(wall, wallHeight, maxZ - minZ), green);
    const innerMinX = minX + wall / 2, innerMaxX = maxX - wall / 2;
    //clear of the bases of the pillars at the back, which reach .3 from the grid
    const innerMinZ = minZ + .3;

    //a white cube turned like in the classic box, at the back left, and a white sphere at the front right. Static, but the
    //cube is not axis aligned so both are traced on their own after the BVH
    const cubeHalf = .55;
    const cube = new Model();
    cube.primitiveType = PrimitiveType.CUBE;
    cube.transform.position = new Vector3(innerMinX + .9, floorTop + cubeHalf, innerMinZ + 1.1);
    cube.transform.eulerY = 20 * Math.PI / 180;
    cube.transform.scaleScalar = cubeHalf;
    cube.color = white.clone();
    this.staticModels.push(cube);

    const sphereRadius = .5;
    const sphere = new Model();
    sphere.primitiveType = PrimitiveType.SPHERE;
    sphere.transform.position = new Vector3(innerMaxX - .8, floorTop + sphereRadius, maxZ - 1.2);
    sphere.transform.scaleScalar = sphereRadius;
    sphere.color = white.clone();
    this.staticModels.push(sphere);
  }

  //a horizontal slab from min to max with a rectangular hole (hole min to max), as 4 boxes: the full depth left and right
  //of the hole, and in front of and behind it
  private addSlabWithHole(minX:number, maxX:number, minZ:number, maxZ:number, y:number, thickness:number,
                          holeMinX:number, holeMaxX:number, holeMinZ:number, holeMaxZ:number, color:Vector3){
    const centerZ = (minZ + maxZ) / 2, holeCenterX = (holeMinX + holeMaxX) / 2, holeWidth = holeMaxX - holeMinX;
    this.addBox(new Vector3((minX + holeMinX) / 2, y, centerZ), new Vector3(holeMinX - minX, thickness, maxZ - minZ), color);
    this.addBox(new Vector3((holeMaxX + maxX) / 2, y, centerZ), new Vector3(maxX - holeMaxX, thickness, maxZ - minZ), color);
    this.addBox(new Vector3(holeCenterX, y, (minZ + holeMinZ) / 2), new Vector3(holeWidth, thickness, holeMinZ - minZ), color);
    this.addBox(new Vector3(holeCenterX, y, (holeMaxZ + maxZ) / 2), new Vector3(holeWidth, thickness, maxZ - holeMaxZ), color);
  }

  //a pillar from the ground through all floors: a base at the bottom and the top, a double one at every floor in between
  //(the slab goes through it), the shaft and 8 ornaments along its edges per floor. The shaft is 1 box over all floors,
  //the bases hide it where they cross. Fewer BVH leaves than a pillar per floor, and no seam between the floors
  private addPillar(x:number, z:number, floorHeight:number, floorCount:number){
    const stone = new Vector3(.8, .78, .72);
    const ornament = stone;
    const height = floorHeight * floorCount;
    this.addBox(new Vector3(x, .25, z), new Vector3(.6, .5, .6), stone);
    for(let floor = 1; floor < floorCount; floor++){
      this.addBox(new Vector3(x, floor * floorHeight, z), new Vector3(.6, 1, .6), stone);
    }
    this.addBox(new Vector3(x, height - .25, z), new Vector3(.6, .5, .6), stone);
    this.addBox(new Vector3(x, height / 2, z), new Vector3(.5, height - 1, .5), stone);
    const ornamentSize = new Vector3(.1, floorHeight - 1.2, .1);
    const {width, length, blockWidth, blockLength} = buildingLayout;
    for(const [dx, dz] of [[.1, .25], [-.1, .25], [.1, -.25], [-.1, -.25], [.25, .1], [-.25, .1], [.25, -.1], [-.25, -.1]]){
      const ox = Math.abs(x + dx), oz = Math.abs(z + dz);
      //not on the outside of the outer walls, the camera doesn't go there
      if(ox > width / 2 || oz > length / 2) continue;
      //not inside the block
      if(ox < blockWidth / 2 && oz < blockLength / 2) continue;
      for(let floor = 0; floor < floorCount; floor++){
        this.addBox(new Vector3(x + dx, (floor + .5) * floorHeight, z + dz), ornamentSize, ornament);
      }
    }
  }

  //a static axis aligned cube, size is the full size
  private addBox(center:Vector3, size:Vector3, color:Vector3){
    const model = new Model();
    model.primitiveType = PrimitiveType.CUBE;
    model.transform.position = center;
    //the ray traced and drawn cube is 2 wide
    model.transform.scale = size.clone().multiplyScalar(.5);
    model.color = color;
    this.staticModels.push(model);
  }

  //the sky sphere encloses the scene, so every ray hits something. See isSky in rayTraceScene.wgsl
  private createSky(radius:number):Model{
    const sky = new Model();
    sky.transform.scaleScalar = radius;
    sky.color = new Vector3(0.25, 0.5, 1).multiplyScalar(4.);
    sky.primitiveType = PrimitiveType.SPHERE;
    return sky;
  }

  //true when every axis of the model is along a world axis, so the world bounding box of a cube is the cube
  private isAxisAligned(model:Model):boolean{
    const m = model.transform.worldMatrix.m;
    for(let column = 0; column < 3; column++){
      const x = Math.abs(m[column * 4]), y = Math.abs(m[column * 4 + 1]), z = Math.abs(m[column * 4 + 2]);
      const max = Math.max(x, y, z);
      if(x + y + z - max > max * 1e-5) return false;
    }
    return true;
  }

  //the world bounding box of the model. The primitives fit in the -1 to 1 cube
  private getBounds(model:Model, min:Vector3, max:Vector3){
    const corner = new Vector3();
    min.setValues(Infinity, Infinity, Infinity);
    max.setValues(-Infinity, -Infinity, -Infinity);
    for(let i = 0; i < 8; i++){
      corner.setValues(i & 1 ? 1 : -1, i & 2 ? 1 : -1, i & 4 ? 1 : -1).transform(model.transform.worldMatrix);
      min.min(corner);
      max.max(corner);
    }
  }

  //over the boxes of the models, its leaves are indices into models. Keeps the boxes for moveModel
  private createBVH(models:Model[]):ModelBVH{
    this.bvhBoxMin = models.map(() => new Vector3());
    this.bvhBoxMax = models.map(() => new Vector3());
    models.forEach((model, i) => this.getBounds(model, this.bvhBoxMin[i], this.bvhBoxMax[i]));
    //limited to the stack of the traversal, a node pushes at most one child
    return new ModelBVH(this.bvhBoxMin, this.bvhBoxMax, Scene.bvhStackSize + 1);
  }

  private writeBVH(bvh:ModelBVH){
    bvh.setNodeData(this.bvhNodeData);
    this.renderer.writeBuffer(this.bvhBuffer, this.bvhNodeData);
    //a node pushes at most one child on the stack, a leaf none
    if(bvh.depth - 1 > Scene.bvhStackSize) LogGL.error('scene: the BVH is deeper than the traversal stack in rayTraceScene.wgsl');
  }

  private writeSceneBuffer(){
    for(let i = 0; i < this.models.length; i++) this.writeModel(i);
    this.renderer.writeBuffer(this.sceneBuffer, this.sceneData);
  }

  //into sceneData
  private writeModel(i:number){
    const model = this.models[i];
    const sceneData = this.sceneData;
    let o = i * this.floatsPerPrimitive;
    sceneData.set(model.transform.worldMatrix.m, o);
    sceneData.set(model.transform.worldMatrix.inverse.m, o + 16);
    const q = model.transform.worldRotation;
    o += 32;
    sceneData[o++] = q.x;
    sceneData[o++] = q.y;
    sceneData[o++] = q.z;
    sceneData[o++] = q.w;
    sceneData[o++] = model.color.x;
    sceneData[o++] = model.color.y;
    sceneData[o++] = model.color.z;
    sceneData[o++] = model.primitiveType;
    //the lengths of the axes of the world matrix, so a shader doesn't take them per pixel
    const m = model.transform.worldMatrix.m;
    sceneData[o++] = Math.hypot(m[0], m[1], m[2]);
    sceneData[o++] = Math.hypot(m[4], m[5], m[6]);
    sceneData[o] = Math.hypot(m[8], m[9], m[10]);
  }

}
