import Renderer from '../render/Renderer';
import ShaderBase, {ShaderBaseOptions} from './ShaderBase';
import {PipelineConstants} from './shaderUtils/OverrideUtils';
import LogGL from '../core/LogGL';

export type ComputeShaderOptions = ShaderBaseOptions & {
  kernel?: string;
}

export default class ComputeShader extends ShaderBase {
  //a constant that keeps changing (a value per frame) would otherwise grow the cache without end
  private static readonly MAX_PIPELINES = 16;
  private dirty = true;
  private kernel: string;
  private _pipeline!: GPUComputePipeline;
  private shaderModule: GPUShaderModule | null = null;
  //by the values of the constants, so switching back to values used before reuses their pipeline
  private pipelines: Map<string, GPUComputePipeline> = new Map();

  public constructor(renderer: Renderer, name: string, shaderCode: string, options: ComputeShaderOptions = {}) {
    super(renderer, name, shaderCode, options);
    this.kernel = options.kernel ?? 'compute_main';

    this.constants = new PipelineConstants(this.shaderCode, false);
    LogGL.log('overrides', name, this.constants.names);

    this.initBindGroups();
  }

  protected constantsChanged() {
    this.dirty = true;
  }

  public get pipeline(): GPUComputePipeline {
    if (this.dirty) {
      //a constant only changes the pipeline: the module and the layout are created once
      if (!this.shaderModule) {
        this.shaderModule = this.renderer.device.createShaderModule({
          code: this.shaderCode,
          label: this.name + 'Module',
        });
      }

      const key = this.constants.key;
      let pipeline = this.pipelines.get(key);
      if (pipeline) {
        //to the end: a Map iterates in insertion order, so the first entry is the least recently used one
        this.pipelines.delete(key);
      } else {
        if (this.pipelines.size >= ComputeShader.MAX_PIPELINES) {
          this.pipelines.delete(this.pipelines.keys().next().value!);
        }
        pipeline = this.renderer.device.createComputePipeline({
          label:   this.name + 'Pipeline',
          layout:  this.pipelineLayout,
          compute: {
            module: this.shaderModule,
            entryPoint: this.kernel,
            constants: this.constants.values,
          },
        });
      }
      this.pipelines.set(key, pipeline);
      this._pipeline = pipeline;

      this.dirty = false;
    }
    return this._pipeline;
  }

  public destruct(): void {
    super.destruct();
    this.pipelines.clear();
    this.dirty = true;
  }
}
