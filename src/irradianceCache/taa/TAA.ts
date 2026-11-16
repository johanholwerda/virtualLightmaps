import Renderer from "monks-webgpu/renderer/render/Renderer";
import RenderTexture from "monks-webgpu/renderer/texture/RenderTexture";
import VertexFragmentShader from "monks-webgpu/renderer/material/VertexFragmentShader";
import ParamGroup from "monks-webgpu/utils/params/ParamGroup";
import ParamBool from "monks-webgpu/utils/params/ParamBool";
import Matrix4x4 from "monks-webgpu/renderer/math/Matrix4x4";
import Vector2 from "monks-webgpu/renderer/math/Vector2";
import Camera from "monks-webgpu/renderer/camera/Camera";
import normalPacking from 'monks-webgpu/shaders/normalPacking.wgsl?raw';
import gBufferPacking from '../includes/gBuffer.wgsl?raw';
import cameraRay from '../includes/cameraRay.wgsl?raw';
import shader from './taa.wgsl?raw';

//temporal anti aliasing, after deferredRenderer/taa. The draw modes render into source, see taa.wgsl. setCamera jitters
//the camera, only while the TAA is used: disabled the camera has no jitter
export default class TAA {
  public enabled!: ParamBool;
  //the image of the draw mode, the TAA resolves it into the history. With depth: the octree ray march draws a mesh
  public readonly source: RenderTexture;
  private renderer: Renderer;
  private material: VertexFragmentShader;
  private histories: RenderTexture[];
  private historyIndex = 0;
  private historyValid = false;
  private readonly resolution = new Vector2();
  private readonly viewProjectionNoJitter = new Matrix4x4();
  private readonly viewProjectionOldNoJitter = new Matrix4x4();
  private readonly inverseView = new Matrix4x4();
  private readonly inverseProjection = new Matrix4x4();

  constructor(renderer: Renderer) {
    this.renderer = renderer;
    this.material = new VertexFragmentShader(renderer, 'taa', shader, {includes: [normalPacking, gBufferPacking, cameraRay]});
    this.material.setSampler('linearSampler', renderer.device.createSampler({
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge',
      magFilter: 'linear',
      minFilter: 'linear',
    }));
    this.source = new RenderTexture(renderer, 1, 1, {scaleToCanvas: true, depth: true, formats: ['rgba16float']});
    this.histories = [0, 1].map(() => new RenderTexture(renderer, 1, 1, {scaleToCanvas: true, depth: false, formats: ['rgba16float']}));
  }

  public init(paramGroup: ParamGroup) {
    const params = { materials: [this.material] };
    const pg = paramGroup.addGroup('TAA');
    //off: no camera jitter either
    this.enabled = pg.addParamBool('enabled', true);
    pg.addParamFloat('feedback', .9, 0, .99, params);
    pg.addParamFloat('clipGamma', 1, .5, 3, params);
    pg.collapsed = true;
  }

  //every frame, after camera.update() with the jitter reset: keeps the camera without jitter for the reprojection, then
  //jitters it when jitter is true
  public setCamera(camera: Camera, jitter: boolean) {
    this.viewProjectionOldNoJitter.copy(this.viewProjectionNoJitter);
    this.viewProjectionNoJitter.copy(camera.viewProjection);
    if (jitter) camera.projection.jitter(this.renderer.size, 16);
  }

  //the history doesn't belong to the next frame: the TAA skipped a frame, or the draw mode changed
  public invalidate() {
    this.historyValid = false;
  }

  //resolves source into the history and copies that to destination
  public draw(camera: Camera, gBuffer: RenderTexture, destination: RenderTexture | null = null) {
    const size = this.renderer.size;
    if (size.x !== this.resolution.x || size.y !== this.resolution.y) {
      this.resolution.copy(size);
      this.historyValid = false;
    }
    this.historyIndex = 1 - this.historyIndex;
    const current = this.histories[this.historyIndex];

    //the jittered matrices, they reconstruct the positions of the gBuffer samples
    Matrix4x4.invert(camera.viewMatrix, this.inverseView);
    Matrix4x4.invert(camera.projectionMatrix, this.inverseProjection);
    this.material.setMatrix('inverseView', this.inverseView);
    this.material.setMatrix('inverseProjection', this.inverseProjection);
    this.material.setVector3('cameraPosition', camera.worldPosition);
    this.material.setMatrix('viewProjectionNoJitter', this.viewProjectionNoJitter);
    this.material.setMatrix('viewProjectionOldNoJitter', this.viewProjectionOldNoJitter);
    this.material.setVector2('resolution', this.resolution);
    this.material.setFloat('historyValid', this.historyValid ? 1 : 0);
    this.material.setTexture('normalDepthTexture', gBuffer.views[1]);
    this.material.setTexture('source', this.source);
    this.material.setTexture('history', this.histories[1 - this.historyIndex]);
    this.renderer.blit(null, current, this.material);

    this.renderer.blit(current, destination);
    this.historyValid = true;
  }
}
