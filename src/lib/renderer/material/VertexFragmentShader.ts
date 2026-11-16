import Renderer from '../render/Renderer';
import ShaderBase, {ShaderBaseOptions} from './ShaderBase';
import RenderTexture from '../texture/RenderTexture';
import LogGL from '../core/LogGL';
import {PipelineConstants} from './shaderUtils/OverrideUtils';

//the vertex buffer layouts come deduplicated from the VertexBufferLayoutManager, so an id per object identifies them
const vertexBufferLayoutIds: WeakMap<GPUVertexBufferLayout, number> = new WeakMap();
let nextVertexBufferLayoutId = 0;
function getVertexBufferLayoutId(layout: GPUVertexBufferLayout): number {
  let id = vertexBufferLayoutIds.get(layout);
  if (id === undefined) {
    id = nextVertexBufferLayoutId++;
    vertexBufferLayoutIds.set(layout, id);
  }
  return id;
}

export default class VertexFragmentShader extends ShaderBase {
  //per material. the shared blit material gets one per render target format, sample count and depth stencil format
  private static readonly MAX_PIPELINES = 64;
  private shaderModule: GPUShaderModule;
  private _vertexBufferLayout: GPUVertexBufferLayout[] = [];
  //part of the pipeline id, so one material can be drawn with meshes of different layouts without rebuilding pipelines
  private vertexBufferLayoutKey: string = '';
  private _dirty: boolean = true;
  //incremented on every pipeline state change so cached render bundles (which bake in the
  //pipeline) can be invalidated. see Renderer.drawCachedBundle
  private _version: number = 0;
  //by render target, vertex buffer layout and pipeline state (see getID), so switching back to a state reuses its pipeline
  private pipelines: Map<string, GPURenderPipeline> = new Map();
  //the part of the id for the pipeline state, rebuilt when it changed
  private stateKey: string = '';
  private warnedAboutEviction: boolean = false;
  private _pipeline!: GPURenderPipeline;
  //what _pipeline was looked up for, so the pipeline getter can skip building the id when nothing changed
  private pipelineRenderTarget: RenderTexture | null = null;
  private pipelineMipLevel: number = -1;
  private pipelineVertexBufferLayoutKey: string = '';

  private _alphaToCoverageEnabled: boolean = false;
  private _depthWriteEnabled: boolean = true;
  private _topology: GPUPrimitiveTopology = 'triangle-list';
  //needed for primitive restart in an indexed strip
  private _stripIndexFormat: GPUIndexFormat | undefined = undefined;
  private _cullMode: GPUCullMode = 'back';
  private _depthCompare: GPUCompareFunction = 'less';
  //by color target index. null or missing: no blending. replaced, not changed, when the blending of a target changes
  private blendStates: (GPUBlendState | null)[] = [];
  private _depthBias: number = 0;
  private _stencilCompare: GPUCompareFunction = 'always';
  private _stencilFailOp: GPUStencilOperation = 'keep';
  private _stencilDepthFailOp: GPUStencilOperation = 'keep';
  private _stencilPassOp: GPUStencilOperation = 'keep';
  private _stencilReadMask: GPUStencilValue = 0xFF;
  private _stencilWriteMask: GPUStencilValue = 0xFF;
  private _colorWriteFlags: GPUColorWriteFlags = GPUColorWrite.ALL;

  public constructor(
    renderer: Renderer,
    name: string,
    shaderCode: string,
    options: ShaderBaseOptions = {},
  ) {
    super(renderer, name, shaderCode, options);

    this.constants = new PipelineConstants(this.shaderCode, true);

    this.initBindGroups();

    this.shaderModule = this.renderer.device.createShaderModule({
      code: this.shaderCode,
      label: this.name,
    });
  }

  public get dirty(): boolean {
    return this._dirty;
  }

  public get version(): number {
    return this._version;
  }

  private markDirty() {
    this._dirty = true;
    this._version++;
  }

  protected constantsChanged() {
    this.markDirty();
  }

  //no dirty flag or version bump here: the layout is part of the pipeline id, so switching between layouts
  //(one material drawn with different meshes) looks up the pipeline made for that layout instead of rebuilding it
  public set vertexBufferLayout(value: GPUVertexBufferLayout[]) {
    let changed = this._vertexBufferLayout.length != value.length;
    for (let i = 0; i < value.length && !changed; i++) {
      changed = value[i] != this._vertexBufferLayout[i];
    }
    if (!changed) return;

    this._vertexBufferLayout = value;
    this.vertexBufferLayoutKey = value.map(getVertexBufferLayoutId).join(',');
  }

  //the same as setting vertexBufferLayout to [value], without allocating an array when it didn't change.
  //called for every draw
  public setVertexBufferLayout(value: GPUVertexBufferLayout) {
    if (this._vertexBufferLayout.length == 1 && this._vertexBufferLayout[0] == value) return;
    this.vertexBufferLayout = [value];
  }

  //no vertex buffers, for the draws without a mesh. called for every such draw
  public clearVertexBufferLayout() {
    if (this._vertexBufferLayout.length == 0) return;
    this.vertexBufferLayout = [];
  }

  public set cullMode(value: GPUCullMode) {
    if (this._cullMode != value) {
      this._cullMode = value;
      this.markDirty();
    }
  }

  public set topology(value: GPUPrimitiveTopology) {
    if (this._topology != value) {
      this._topology = value;
      this.markDirty();
    }
  }

  public set stripIndexFormat(value: GPUIndexFormat | undefined) {
    if (this._stripIndexFormat != value) {
      this._stripIndexFormat = value;
      this.markDirty();
    }
  }

  public set alphaToCoverageEnabled(value: boolean) {
    if (this._alphaToCoverageEnabled != value) {
      this._alphaToCoverageEnabled = value;
      this.markDirty();
    }
  }

  public set depthWriteEnabled(value: boolean) {
    if (this._depthWriteEnabled != value) {
      this._depthWriteEnabled = value;
      this.markDirty();
    }
  }

  public set depthCompare(value: GPUCompareFunction) {
    if (this._depthCompare != value) {
      this._depthCompare = value;
      this.markDirty();
    }
  }

  public set depthBias(value: number) {
    if (this._depthBias != value) {
      this._depthBias = value;
      this.markDirty();
    }
  }

  public set stencilCompare(value: GPUCompareFunction) {
    if (this._stencilCompare != value) {
      this._stencilCompare = value;
      this.markDirty();
    }
  }

  public set stencilFailOp(value: GPUStencilOperation) {
    if (this._stencilFailOp != value) {
      this._stencilFailOp = value;
      this.markDirty();
    }
  }

  public set stencilDepthFailOp(value: GPUStencilOperation) {
    if (this._stencilDepthFailOp != value) {
      this._stencilDepthFailOp = value;
      this.markDirty();
    }
  }

  public set stencilPassOp(value: GPUStencilOperation) {
    if (this._stencilPassOp != value) {
      this._stencilPassOp = value;
      this.markDirty();
    }
  }

  public set stencilReadMask(value: GPUStencilValue) {
    if (this._stencilReadMask != value) {
      this._stencilReadMask = value;
      this.markDirty();
    }
  }

  public set stencilWriteMask(value: GPUStencilValue) {
    if (this._stencilWriteMask != value) {
      this._stencilWriteMask = value;
      this.markDirty();
    }
  }

  public setColorWriteMask(value: GPUColorWriteFlags) {
    if (this._colorWriteFlags != value) {
      this._colorWriteFlags = value;
      this.markDirty();
    }
  }

  //the blend setters take the index of the color target, with multiple render targets every target blends on its own
  public setDefaultBlending(target: number = 0) {
    if (!this.blendStates[target]) return;
    this.blendStates[target] = null;
    this.markDirty();
  }

  public setPremultipliedAlphaBlending(target: number = 0) {
    this.setBlending(target, 'one', 'one-minus-src-alpha', 'one', 'one-minus-src-alpha');
  }

  public setAlphaBlending(target: number = 0) {
    this.setBlending(target, 'src-alpha', 'one-minus-src-alpha', 'one', 'one-minus-src-alpha');
  }

  public setAdditiveBlending(target: number = 0) {
    this.setBlending(target, 'one', 'one', 'one', 'one');
  }

  public setMultipliedBlending(target: number = 0) {
    this.setBlending(target, 'zero', 'src', 'zero', 'src-alpha');
  }

  //any other blending, like the revealage target of weighted blended transparency:
  //{color: {srcFactor: 'zero', dstFactor: 'one-minus-src'}, alpha: {srcFactor: 'zero', dstFactor: 'one-minus-src-alpha'}}
  public setBlendState(state: GPUBlendState | null, target: number = 0) {
    if (!state) {
      this.setDefaultBlending(target);
      return;
    }
    //the defaults of webgpu
    this.setBlending(target,
      state.color.srcFactor ?? 'one', state.color.dstFactor ?? 'zero', state.alpha.srcFactor ?? 'one', state.alpha.dstFactor ?? 'zero',
      state.color.operation ?? 'add', state.alpha.operation ?? 'add');
  }

  //only marks the pipeline dirty when the blending changes, so calling a blend setter every frame is free
  private setBlending(target: number, colorSrc: GPUBlendFactor, colorDst: GPUBlendFactor, alphaSrc: GPUBlendFactor, alphaDst: GPUBlendFactor, colorOperation: GPUBlendOperation = 'add', alphaOperation: GPUBlendOperation = 'add') {
    const current = this.blendStates[target];
    if (current && current.color.srcFactor == colorSrc && current.color.dstFactor == colorDst && current.color.operation == colorOperation
      && current.alpha.srcFactor == alphaSrc && current.alpha.dstFactor == alphaDst && current.alpha.operation == alphaOperation) return;
    this.blendStates[target] = {
      color: {operation: colorOperation, srcFactor: colorSrc, dstFactor: colorDst},
      alpha: {operation: alphaOperation, srcFactor: alphaSrc, dstFactor: alphaDst},
    };
    this.markDirty();
  }

  private createPipeline(renderTarget: RenderTexture, mipLevel:number): GPURenderPipeline {
    const vertexState: GPUVertexState = {
      module:     this.shaderModule,
      entryPoint: 'vs_main',
      buffers:    this._vertexBufferLayout,
      constants:  this.constants.values,
    };

    const primitiveState: GPUPrimitiveState = {
      topology: this._topology,
      cullMode: this._cullMode,
    };
    //only allowed for strips
    if (this._stripIndexFormat && this._topology.endsWith('strip')) {
      primitiveState.stripIndexFormat = this._stripIndexFormat;
    }

    const targets: GPUColorTargetState[] = [];
    const formats = renderTarget.formats;
    for (let i = 0; i < formats.length; i++) {
      const target: GPUColorTargetState = {format: formats[i], writeMask: this._colorWriteFlags};
      const blend = this.blendStates[i];
      if (blend) target.blend = blend;
      targets.push(target);
    }
    const sampleCount = renderTarget.getSampleCount(mipLevel);

    const fragmentState: GPUFragmentState = {
      module:     this.shaderModule,
      entryPoint: 'fs_main',
      targets:    targets,
      constants:  this.constants.values,
    };

    let descriptor: GPURenderPipelineDescriptor =
          {
            layout:      this.pipelineLayout,
            vertex:      vertexState,
            fragment:    fragmentState,
            primitive:   primitiveState,
            multisample: {
              count: sampleCount,
              //only valid with msaa. so the material can also be drawn without, like into the higher mip levels
              alphaToCoverageEnabled: this._alphaToCoverageEnabled && sampleCount > 1,
            },
            label:       this.name,
          };
    if (renderTarget.depthStencilAttachment && mipLevel == 0) {
      const depthStencil: GPUDepthStencilState = {
        format:            renderTarget.depthStencilAttachment.format,
      };

      if(renderTarget.depthEnabled) {
        depthStencil.depthWriteEnabled = this._depthWriteEnabled;
        depthStencil.depthCompare = this._depthCompare;
        depthStencil.depthBias = this._depthBias;
      }

      if(renderTarget.stencilEnabled) {
        const stencilState: GPUStencilFaceState = {
          compare: this._stencilCompare,
          failOp: this._stencilFailOp,
          depthFailOp: this._stencilDepthFailOp,
          passOp: this._stencilPassOp,
        };
        depthStencil.stencilFront = stencilState;
        depthStencil.stencilBack = stencilState;
        depthStencil.stencilReadMask = this._stencilReadMask;
        depthStencil.stencilWriteMask =  this._stencilWriteMask;
      }

      descriptor.depthStencil = depthStencil;
    }
    const pipeline = this.renderer.device.createRenderPipeline(descriptor);
    LogGL.log('created pipeline', this.name, this.pipelines.size + 1, descriptor);
    return pipeline;
  }

  //everything createPipeline reads apart from the render target and the vertex buffer layout
  private getStateKey(): string {
    return [
      this._topology, this._stripIndexFormat, this._cullMode, this._alphaToCoverageEnabled,
      this._depthWriteEnabled, this._depthCompare, this._depthBias,
      this._stencilCompare, this._stencilFailOp, this._stencilDepthFailOp, this._stencilPassOp, this._stencilReadMask, this._stencilWriteMask,
      this._colorWriteFlags, JSON.stringify(this.blendStates),
      this.constants.key,
    ].join('_');
  }

  private getID(renderTarget: RenderTexture, mipLevel:number):string{
    let id = renderTarget.getSampleCount(mipLevel) + '_' + renderTarget.formats.join('_');
    //the mip level only matters for msaa and the depth stencil attachment, which are only used at mip level 0
    if (renderTarget.depthStencilAttachment && mipLevel == 0) {
      //whether depth and stencil are enabled changes the depth stencil state
      id += '_' + renderTarget.depthStencilAttachment.format + '_' + renderTarget.depthEnabled + '_' + renderTarget.stencilEnabled;
    }
    return id + '_' + this.vertexBufferLayoutKey + '|' + this.stateKey;
  }

  public get pipeline(): GPURenderPipeline {
    const renderTarget = this.renderer.renderTarget;
    const mipLevel =  this.renderer.currentRenderTextureMipLevel;
    if (!this._dirty && renderTarget === this.pipelineRenderTarget && mipLevel === this.pipelineMipLevel && this.vertexBufferLayoutKey === this.pipelineVertexBufferLayoutKey) {
      return this._pipeline;
    }
    this.pipelineRenderTarget = renderTarget;
    this.pipelineMipLevel = mipLevel;
    this.pipelineVertexBufferLayoutKey = this.vertexBufferLayoutKey;

    if (this._dirty) {
      this.stateKey = this.getStateKey();
      this._dirty = false;
    }

    //compare on the id, not on the RenderTexture instance: two RenderTextures can share a pipeline
    //and one RenderTexture can need different pipelines per mip level
    const id = this.getID(renderTarget, mipLevel);
    let pipeline = this.pipelines.get(id);
    if (pipeline) {
      //to the end: a Map iterates in insertion order, so the first entry is the least recently used one
      this.pipelines.delete(id);
    } else {
      //a state that keeps changing (an animated depth bias or constant) would otherwise grow the cache without end
      if (this.pipelines.size >= VertexFragmentShader.MAX_PIPELINES) {
        this.pipelines.delete(this.pipelines.keys().next().value!);
        if (!this.warnedAboutEviction) {
          this.warnedAboutEviction = true;
          LogGL.warn('VertexFragmentShader: more than', VertexFragmentShader.MAX_PIPELINES, 'pipelines in use, the least recently used ones are created again when needed', this.name);
        }
      }
      pipeline = this.createPipeline(renderTarget, mipLevel);
    }
    this.pipelines.set(id, pipeline);
    this._pipeline = pipeline;
    return this._pipeline;
  }

  public destruct(): void {
    super.destruct();
    this.pipelines.clear();
    this.pipelineRenderTarget = null;
  }
}
