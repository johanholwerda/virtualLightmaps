import Renderer from "monks-webgpu/renderer/render/Renderer";
import WebGPUPreLoader from "monks-webgpu/renderer/core/WebGPUPreLoader";
import LogGL from "monks-webgpu/renderer/core/LogGL";
import IrradianceCacheDraw from "./IrradianceCacheDraw";

export default class IrradianceCache {
  private renderer!: Renderer;
  private irradianceCacheDraw!: IrradianceCacheDraw;
  public static description: string = "virtual lightmaps. Use WASD and mousse for camera control. Click scene objects to drag";

  async start(canvasParent: HTMLElement, preloader: WebGPUPreLoader, uiVisible:boolean = true): Promise<void> {
    LogGL.ENABLED = uiVisible;
    this.renderer = new Renderer();

    const requiredFeatures: GPUFeatureName[] = ['float32-filterable'];
    //updateOctree.wgsl has 9 storage buffers, the default is 8. Lowered to what the adapter supports, see checkRequiredLimitsAgainstAdapterLimits
    const requiredLimits = {maxStorageBuffersPerShaderStage: 10};
    await this.renderer.init(canvasParent, {sampleCount: 1, requiredFeatures, requiredLimits});

    this.renderer.setClearColor(0, 0, 0, 0);

    this.irradianceCacheDraw = new IrradianceCacheDraw(this.renderer, preloader);
  }

  public init() {
    this.irradianceCacheDraw.init(LogGL.ENABLED);

    this.update();
  }

  public update(time = 0) {
    this.renderer.update(time);

    this.irradianceCacheDraw.update();

    this.renderer.finish();

    requestAnimationFrame((time) => this.update(time));
  }
}
