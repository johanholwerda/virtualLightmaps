import Renderer from "webGPU-framework/renderer/render/Renderer";
import WebGPUPreLoader from "webGPU-framework/renderer/core/WebGPUPreLoader";
import FlyCamera from "webGPU-framework/utils/camera/FlyCamera";
import ParamGroup from "webGPU-framework/utils/params/ParamGroup";
import ParamsGui from "webGPU-framework/utils/paramsGui/ParamsGui";
import RenderTexture from "webGPU-framework/renderer/texture/RenderTexture";
import ParamSelect from "webGPU-framework/utils/params/ParamSelect";
import Scene from "./scene/Scene";
import Vector3 from "webGPU-framework/renderer/math/Vector3";
import SceneRendererForward from "./sceneRendererForward/SceneRendererForward";
import ReprojectAndFill from "./reprojectAndFill/ReprojectAndFill";
import TemporalAccumulate from "./temporalAccumulate/TemporalAccumulate";
import Resolve from "./shared/Resolve";
import TemporalAccumulateHalfRes from "./temporalAccumulateHalfRes/TemporalAccumulateHalfRes";
import OctreeUpdater, { OctreeDrawMode } from "./octree/octreeUpdater/OctreeUpdater";
import SpatialHash from "./spatialHash/SpatialHash";
import VirtualLightmap from "./virtualLightmap/VirtualLightmap";
import VertexFragmentShader from "webGPU-framework/renderer/material/VertexFragmentShader";
import showNormal from "./showNormal.wgsl?raw";
import showModel from "./showModel.wgsl?raw";
import showTriangle from "./showTriangle.wgsl?raw";
import normalPacking from 'webGPU-framework/shaders/normalPacking.wgsl?raw';
import gBufferPacking from './includes/gBuffer.wgsl?raw';
import GroundTruth from "./groundTruth/GroundTruth";
import TAA from "./taa/TAA";
import DayCycle from "./scene/DayCycle";
import SceneEditor from "./scene/SceneEditor";

enum DrawMode {
  OCTREE_SPLAT,
  OCTREE_RESOLVE,
  OCTREE_RAY_MARCH,
  OCTREE_DEBUG,
  SPATIAL_HASH,
  VIRTUAL_LIGHTMAP,
  REPROJECT_AND_FILL,
  TEMPORAL_ACCUMULATION,
  TEMPORAL_ACCUMULATE_HALF_RES,
  ALBEDO,
  NORMAL,
  MODEL_ID,
  TRIANGLE_INDEX,
  GROUND_TRUTH,
}

//the random cubes of the scene stay clear of this path (see Scene.createRandomCubes). It is the camera path of the
//performance test of the original example, kept so the scene is the same. It starts at the start view and ends at an
//overview, the pillars stand every 4 m and the path crosses their rows halfway between two. See Scene.createBuilding
const RANDOM_CUBES_CLEAR_PATH: Vector3[] = [
  //once around the inner building in the middle of the atrium, it hides what was seen before
  new Vector3(2, 1.5, 12),
  new Vector3(-8, 1.6, 10),
  new Vector3(-16, 1.6, 6),
  new Vector3(-14, 1.6, -10),
  new Vector3(8, 1.6, -10),
  new Vector3(16, 1.6, -6),
  new Vector3(14, 1.6, 10),
  //into the gallery of the inner building, between the pillars at x 4 and 8, and along its front
  new Vector3(6, 1.8, 8),
  new Vector3(6, 1.8, 4),
  new Vector3(-10, 1.8, 4),
  //up through the open ring to the left side of the first floor of the outer building
  new Vector3(-16, 3.2, 4),
  new Vector3(-22, 5.8, 0),
  new Vector3(-22, 5.8, -12),
  //out over the atrium to the overview
  new Vector3(-12, 7, -12),
  new Vector3(-14, 13, -18),
];

export default class IrradianceCacheDraw {
  private renderer: Renderer;
  private camera: FlyCamera;
  private scene:Scene;
  private sceneRendererForward:SceneRendererForward;
  private gBuffer:RenderTexture;
  private reprojectAndFill:ReprojectAndFill;
  private drawMode!: ParamSelect;
  private showNormalMaterial:VertexFragmentShader;
  private showModelMaterial:VertexFragmentShader;
  private showTriangleMaterial:VertexFragmentShader;
  private resolve!: Resolve;
  private temporalAccumulation: TemporalAccumulate;
  private temporalAccumulationResolve: Resolve;
  private temporalAccumulateHalfRes: TemporalAccumulateHalfRes;
  private indirectLightingOctree!: OctreeUpdater;
  private spatialHash: SpatialHash;
  private virtualLightmap: VirtualLightmap;
  private groundTruth: GroundTruth;
  private taa: TAA;
  private dayCycle!: DayCycle;
  private sceneEditor: SceneEditor;
  //the draw mode the TAA history was made with
  private taaDrawMode: DrawMode | null = null;
  // private rayTraceUniformBuffer:GPUBuffer;

  constructor(renderer:Renderer, preloader: WebGPUPreLoader){
    this.renderer = renderer;

    this.scene = new Scene(this.renderer, preloader);

    //looking at the building from the front
    this.camera = new FlyCamera(this.renderer, { fov: 1, nearPlane: 0.1, farPlane: 100 });
    this.camera.lookAt(new Vector3(2, 1.5, 12), new Vector3(2, 1.6, 0));
    this.camera.setTranslateSpeed(0.2);
    //a full stop soon after the input, so groundTruth starts accumulating
    this.camera.stopDuration = .5;

    this.sceneRendererForward = new SceneRendererForward(this.renderer);

    //albedo, normalDepth packed into 8 bytes (see includes/gBuffer.wgsl). The depth is the analytic ray distance,
    //the depth buffer can't replace it: that is the depth of the tessellated triangles. And the model index and triangle
    //packed into an r32uint (see packModelTriangle), for the virtual lightmap
    this.gBuffer = new RenderTexture(this.renderer, 1, 1, {scaleToCanvas: true, depth:true, formats: ['rgba8unorm', 'rg32uint', 'r32uint']});

    this.reprojectAndFill = new ReprojectAndFill(this.renderer);

    this.resolve = new Resolve(this.renderer);

    this.temporalAccumulation = new TemporalAccumulate(this.renderer);
    this.temporalAccumulationResolve = new Resolve(this.renderer);

    this.temporalAccumulateHalfRes = new TemporalAccumulateHalfRes(this.renderer);

    this.indirectLightingOctree = new OctreeUpdater(this.renderer, this.scene.size);

    this.spatialHash = new SpatialHash(this.renderer);

    this.virtualLightmap = new VirtualLightmap(this.renderer);

    this.groundTruth = new GroundTruth(this.renderer);

    this.taa = new TAA(this.renderer);

    this.showNormalMaterial = new VertexFragmentShader(this.renderer, 'showNormal', showNormal, {includes: [normalPacking, gBufferPacking]});
    this.showModelMaterial = new VertexFragmentShader(this.renderer, 'showModel', showModel, {includes: [normalPacking, gBufferPacking]});
    this.showTriangleMaterial = new VertexFragmentShader(this.renderer, 'showTriangle', showTriangle, {includes: [normalPacking, gBufferPacking]});

    this.sceneEditor = new SceneEditor(this.renderer, this.scene, this.camera);
  }

  public init(uiVisible:boolean): void {
    //the materials of the exposure param
    const paramGroupMaterials = [
      ...this.indirectLightingOctree.exposureMaterials,
      this.spatialHash.exposureMaterial,
      this.virtualLightmap.exposureMaterial,
      this.resolve.exposureMaterial,
      this.temporalAccumulationResolve.exposureMaterial,
      this.temporalAccumulateHalfRes.exposureMaterial,
      this.groundTruth.exposureMaterial
    ];
    const paramGroup: ParamGroup = new ParamGroup('controls');
    //the order of DrawMode
    const drawModeNames = ['octreeSplat', 'octreeResolve', 'octreeRayMarch', 'octreeDebug', 'spatialHash', 'virtualLightmap', 'reprojectAndFill', 'temporalAccumulation', 'temporalAccumulateHalfRes', 'albedo', 'normal', 'modelID', 'triangleIndex', 'groundTruth'];
    this.drawMode = paramGroup.addParamSelect('drawMode', DrawMode.VIRTUAL_LIGHTMAP, drawModeNames);
    //one exposure for all the draw modes, the lighting is HDR
    paramGroup.addParamFloat('exposure', .7, 0, 1, { materials: paramGroupMaterials });
    this.taa.init(paramGroup);
    this.scene.init(RANDOM_CUBES_CLEAR_PATH);
    const cachedPathMaterials = [
      this.sceneRendererForward.material,
      this.indirectLightingOctree.updateShader,
      this.spatialHash.updateShader,
      this.virtualLightmap.updateShader,
      this.reprojectAndFill.fillShader,
      this.temporalAccumulation.fillShader,
      this.temporalAccumulateHalfRes.accumulateShader
    ];
    const pgRayTraceMaterials = [...cachedPathMaterials, this.groundTruth.accumulateShader];
    const pgRayTrace = paramGroup.addGroup('rayTrace');
    //groundTruth has its own bounceCount, so this one only sets the cached path materials
    const bounceCountMaterials = cachedPathMaterials.filter(m => m.hasUniform('bounceCount'));
    const setBounceCount = (value: number) => bounceCountMaterials.forEach(m => m.setFloat('bounceCount', value));
    setBounceCount(pgRayTrace.addParamInt('bounceCount', 4, 0, 1, { callback: setBounceCount }).value);
    const params = { materials: pgRayTraceMaterials };
    pgRayTrace.addParamFloat('directionalBrightness', 10, 0.0, 1.0, params);
    pgRayTrace.addParamFloat('directionalProbability', .3, 0.0, 1.0, params);
    //the color of the noon sun, the day cycle makes it redder and dimmer near the horizon
    const directionalColor = pgRayTrace.addParamColor('directionalColor', new Vector3(1,.8,.6), params);
    //angular radius of the sun in radians, softens the shadows. The real sun is .0047
    pgRayTrace.addParamFloat('sunRadius', .02, 0, .2, params);
    //morning to evening and a quick night, see DayCycle
    this.dayCycle = new DayCycle(this.scene, pgRayTraceMaterials, directionalColor);
    this.dayCycle.init(pgRayTrace);
    //the noon sky, emissive: color times brightness, so the color picker stays in 0-1. Scene.createSky starts at (.25, .5, 1) * 4
    const params2 = { callback: () => updateSky() };
    const skyColor = pgRayTrace.addParamColor('skyColor', new Vector3(.25, .5, 1), params2);
    const skyBrightness = pgRayTrace.addParamFloat('skyBrightness', 4, 0, 10, params2);
    const updateSky = () => this.dayCycle.setDaySky(skyColor.value.clone().multiplyScalar(skyBrightness.value));
    //blue noise instead of white noise for the first bounce and the sun sample, see getBlueNoise in pathTrace.wgsl
    pgRayTrace.addParamBool('blueNoise', true, { callback: (value) => this.scene.setBlueNoise(value) });

    this.sceneRendererForward.init(this.scene);
    this.sceneEditor.init();


    const pgReprojectAndFill = paramGroup.addGroup('reprojectAndFill');
    this.reprojectAndFill.init(pgReprojectAndFill, this.scene);
    this.resolve.init(pgReprojectAndFill, this.scene);

    //without the denoise by default: the plain baseline. The sample count shows white where the pixels stopped getting
    //samples, at the accumulationMax default
    const pgTemporalAccumulation = paramGroup.addGroup('temporalAccumulation');
    this.temporalAccumulation.init(pgTemporalAccumulation, this.scene);
    this.temporalAccumulationResolve.init(pgTemporalAccumulation, this.scene, 1, 64);

    this.temporalAccumulateHalfRes.init(paramGroup.addGroup('temporalAccumulateHalfRes'), this.scene);

    const pgOctree = paramGroup.addGroup('octree');
    this.indirectLightingOctree.init(pgOctree, this.scene);

    this.spatialHash.init(paramGroup, this.scene);

    this.virtualLightmap.init(paramGroup, this.scene);

    this.groundTruth.init(paramGroup, this.scene);

    //everything collapsed except the virtual lightmap (and its subgroups)
    const collapse = (group:ParamGroup) => group.children.forEach(child => {
      if(child.name === 'virtualLightmap') return;
      child.collapsed = true;
      collapse(child);
    });
    collapse(paramGroup);
    if(uiVisible) new ParamsGui(paramGroup);
  }

  public update(finalDestination:RenderTexture | null = null): void {
    this.dayCycle.update();
    //before the camera update: a drag of the pivot doesn't rotate the camera
    this.sceneEditor.update();
    this.camera.mouseRotationEnabled = !this.sceneEditor.dragging;
    //the caches keep taking samples while the sun or a model moves
    this.scene.sunChanging = this.dayCycle.sunChanging;
    this.scene.lightChanging = this.scene.sunChanging || this.sceneEditor.dragging;

    const drawMode = this.drawMode.value as DrawMode;
    const octreeDrawMode = this.getOctreeDrawMode(drawMode);
    const taaEnabled = this.taa.enabled.value;
    const taaPass = taaEnabled && this.usesTAAPass(drawMode);
    //groundTruth has the TAA integrated: the jittered samples average in its accumulation
    const jitter = taaPass || (taaEnabled && drawMode == DrawMode.GROUND_TRUTH);
    if(!taaPass || drawMode !== this.taaDrawMode) this.taa.invalidate();
    this.taaDrawMode = taaPass ? drawMode : null;

    //without the jitter of the previous frame, so camera.changed only sees the camera motion
    this.camera.projection.resetJitter();
    this.camera.update();
    this.taa.setCamera(this.camera, jitter);

    //with the TAA pass the draw mode renders into its source
    const destination = taaPass ? this.taa.source : finalDestination;

    this.renderer.startRenderPass(this.gBuffer);
    this.sceneRendererForward.draw(this.camera, this.scene);

    if(octreeDrawMode !== null) {
      this.indirectLightingOctree.update(this.gBuffer, this.camera);
    }else if(drawMode == DrawMode.SPATIAL_HASH){
      this.spatialHash.update(this.gBuffer, this.camera);
    }else if(drawMode == DrawMode.VIRTUAL_LIGHTMAP){
      this.virtualLightmap.update(this.gBuffer, this.camera);
    }else if(drawMode == DrawMode.GROUND_TRUTH){
      this.groundTruth.update(this.gBuffer, this.camera);
    }

    this.renderer.startRenderPass(destination);

    if(octreeDrawMode !== null) {
      this.indirectLightingOctree.draw(octreeDrawMode, this.camera, this.gBuffer, destination);
    }else if(drawMode == DrawMode.SPATIAL_HASH){
      this.spatialHash.draw(this.camera, this.gBuffer, destination);
    }else if(drawMode == DrawMode.VIRTUAL_LIGHTMAP){
      this.virtualLightmap.draw(this.camera, this.gBuffer, destination, taaPass);
    }else if(drawMode == DrawMode.REPROJECT_AND_FILL){
      this.reprojectAndFill.draw(this.camera, this.gBuffer);
      this.resolve.draw(this.reprojectAndFill.lighting, this.gBuffer, this.camera, destination);
    }else if(drawMode == DrawMode.TEMPORAL_ACCUMULATION){
      this.temporalAccumulation.draw(this.camera, this.gBuffer);
      this.temporalAccumulationResolve.draw(this.temporalAccumulation.lighting, this.gBuffer, this.camera, destination);
    }else if(drawMode == DrawMode.TEMPORAL_ACCUMULATE_HALF_RES){
      this.temporalAccumulateHalfRes.draw(this.camera, this.gBuffer, destination);
    }else if(drawMode == DrawMode.ALBEDO){
      this.renderer.blit(this.gBuffer.views[0], destination);
    }else if(drawMode == DrawMode.NORMAL){
      this.renderer.blit(this.gBuffer.views[1], destination, this.showNormalMaterial);
    }else if(drawMode == DrawMode.MODEL_ID){
      this.renderer.blit(this.gBuffer.views[2], destination, this.showModelMaterial);
    }else if(drawMode == DrawMode.TRIANGLE_INDEX){
      this.renderer.blit(this.gBuffer.views[2], destination, this.showTriangleMaterial);
    }else if(drawMode == DrawMode.GROUND_TRUTH){
      this.groundTruth.draw(this.gBuffer, destination);
    }

    if(taaPass) this.taa.draw(this.camera, this.gBuffer, finalDestination);

    //the pivot is drawn after the TAA, without the jitter so it doesn't shake. The next frame resets it anyway
    this.camera.projection.resetJitter();
    this.sceneEditor.draw();
  }

  //the draw modes that resolve the lighting per pixel get a TAA pass. Not the views of the raw gBuffer and the octree
  //debug, and not groundTruth: that accumulates the jittered samples itself. Not the lightmap view of the virtual lightmap
  //either: that is in lightmap space, the reprojection with the camera would smear it
  private usesTAAPass(drawMode:DrawMode): boolean {
    if(drawMode == DrawMode.VIRTUAL_LIGHTMAP && this.virtualLightmap.showLightmap.value) return false;
    return drawMode != DrawMode.OCTREE_DEBUG && drawMode != DrawMode.ALBEDO && drawMode != DrawMode.NORMAL && drawMode != DrawMode.MODEL_ID && drawMode != DrawMode.TRIANGLE_INDEX && drawMode != DrawMode.GROUND_TRUTH;
  }

  private getOctreeDrawMode(drawMode:DrawMode): OctreeDrawMode | null {
    if(drawMode == DrawMode.OCTREE_SPLAT) return OctreeDrawMode.SPLAT;
    if(drawMode == DrawMode.OCTREE_RESOLVE) return OctreeDrawMode.RESOLVE;
    if(drawMode == DrawMode.OCTREE_RAY_MARCH) return OctreeDrawMode.RAY_MARCH;
    if(drawMode == DrawMode.OCTREE_DEBUG) return OctreeDrawMode.DEBUG;
    return null;
  }
}
