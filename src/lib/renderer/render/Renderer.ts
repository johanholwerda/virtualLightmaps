import MouseListener from '../input/MouseListener';
import Mesh from '../mesh/Mesh';
import Quad from 'monks-webgpu/utils/mesh/geometries/Quad';
import VertexFragmentShader from '../material/VertexFragmentShader';
import ComputeShader from '../material/ComputeShader';
import RenderTexture from '../texture/RenderTexture';
import LogGL from '../core/LogGL';
import WebGPUMipmapGenerator from '../mipmapper/WebGPUMipmapGenerator';
import VertexBufferLayoutManager from './VertexBufferLayoutManager';
import Vector2 from '../math/Vector2';
import CanvasManager from '../core/CanvasManager';
import Time from '../core/Time';
import ShaderBase, { BindGroupState } from '../material/ShaderBase';
import DummyBindings from '../material/DummyBindings';
import Texture from '../texture/Texture';
import { RenderBundleCacheManager } from './RenderBundleCacheManager';
import Profiler from './Profiler';
import blitShader from 'monks-webgpu/shaders/blit.wgsl?raw';
import Color from '../math/Color';


export type RendererOptions = {
  format: GPUTextureFormat;
  sampleCount: number;
  depth: boolean;
  stencil: boolean;
  requiredLimits: Record<string, GPUSize64>;
  requiredFeatures: GPUFeatureName[];
  //initial size in bytes of the buffer that holds the uniforms of all draws and dispatches of a frame. it grows when a frame needs more
  uniformBufferSize: number;
  pixelRatio: number;
  //see RenderTextureOptions.discardMSAA. for the screen when sampleCount > 1
  discardMSAA: boolean;
}

export const defaultRendererOptions: RendererOptions = {
  format: typeof navigator !== 'undefined' && navigator.gpu ? navigator.gpu.getPreferredCanvasFormat() : 'bgra8unorm' as GPUTextureFormat,
  sampleCount: 1,
  depth: true,
  stencil: false,
  requiredLimits: {},
  requiredFeatures: [],
  //every draw or dispatch with changed uniforms takes minUniformBufferOffsetAlignment (usually 256) bytes or more,
  //so this fits about 1024 of them per frame before the buffer grows
  uniformBufferSize: 1024 * 256,
  //-1 means use devicePixelRatio
  pixelRatio: -1,
  discardMSAA: false,
};

export type DrawOptions = {
  instanceCount: number;
}

export type DrawVerticesOptions = {
  instanceCount: number;
  //without an index buffer the first vertex, with one the base vertex that is added to every index
  firstVertex: number;
  firstInstance: number;
  //only used with an index buffer
  firstIndex: number;
  indexFormat: GPUIndexFormat;
}

export type ClearOptions = {
  colorClear?: boolean;
  clearColor?: Color;
  depthClear?: boolean;
  depthClearValue?: number;
  //defaults to depthClear
  stencilClear?: boolean;
  stencilClearValue?: number;
}

const DEFAULT_DEPTH_CLEAR_VALUE = 1;

//what is currently set on the render pass encoder, so draws can skip setting the same state again
class RenderPassState {
  public pipeline: GPURenderPipeline | null = null;
  public vertexBuffer: GPUBuffer | null = null;
  public indexBuffer: GPUBuffer | null = null;
  public indexFormat: GPUIndexFormat | null = null;
  public bindGroups: BindGroupState = { bindGroups: [], offsets: [] };

  //a new pass starts without state and executeBundles clears it
  public reset() {
    this.pipeline = null;
    this.vertexBuffer = null;
    this.indexBuffer = null;
    this.indexFormat = null;
    this.bindGroups.bindGroups.length = 0;
    this.bindGroups.offsets.length = 0;
  }
}

//what is currently set on the compute pass encoder, so dispatches can skip setting the same state again
class ComputePassState {
  public pipeline: GPUComputePipeline | null = null;
  public bindGroups: BindGroupState = { bindGroups: [], offsets: [] };

  public reset() {
    this.pipeline = null;
    this.bindGroups.bindGroups.length = 0;
    this.bindGroups.offsets.length = 0;
  }
}

export default class Renderer {
  public canvasManager!: CanvasManager;
  public readonly time: Time = new Time();
  public canvas!: HTMLCanvasElement;
  public clearColor: Color = new Color(0, 0, 0, 1);
  // public limits!: GPUSupportedLimits;
  public device!: GPUDevice;
  public context!: GPUCanvasContext;
  private _commandEncoder: GPUCommandEncoder | null = null;
  private _renderPassEncoder: GPURenderPassEncoder | null = null;
  private _computePassEncoder: GPUComputePassEncoder | null = null;
  public vertexBufferLayoutManager!: VertexBufferLayoutManager;
  protected _mouseListener: MouseListener | undefined;
  public options!: RendererOptions;
  private blitMesh!: Mesh;
  private copyTextureMaterial!: VertexFragmentShader;
  //used to render to screen
  private renderTexture!: RenderTexture;
  private currentRenderTexture!: RenderTexture;
  public currentRenderTextureMipLevel: number = 0;
  private mipmapGenerator!: WebGPUMipmapGenerator;
  public resourceID: number = 0;
  public instanceID: number = 0;
  //the uniforms of all draws and dispatches of a command encoder, packed one after the other (see setUniformData) and
  //uploaded in finish. a single buffer is enough: queue.writeBuffer copies the data immediately and is ordered with
  //submit, so it never overwrites uniforms that previously submitted work still has to read
  public uniformBuffer!: GPUBuffer;
  private uniformData!: Float32Array;
  //where the next uniforms go. keeps counting past the end of the buffer, so it tells how big the buffer should be
  private uniformDataOffset: number = 0;
  private uniformUploadSize: number = 0;
  //incremented when the uniforms are uploaded and the packing starts over: offsets from older epochs are reused
  public uniformEpoch: number = 0;
  public minUniformBufferOffsetAlignment: number = -1;
  private renderPassState: RenderPassState = new RenderPassState();
  private computePassState: ComputePassState = new ComputePassState();
  private warnedAboutResumedPass: boolean = false;
  public dummyBindings!: DummyBindings;
  public renderBundleCacheManager: RenderBundleCacheManager = new RenderBundleCacheManager();
  private attachmentKeyCache: { rt: RenderTexture | null, mipLevel: number, depthStencilFormat: GPUTextureFormat | undefined, sampleCount: number, key: string } =
    { rt: null, mipLevel: 0, depthStencilFormat: undefined, sampleCount: 0, key: '' };
  //reused by drawCachedBundle
  private bindGroupIds: number[] = [];
  //run right after the next submit, see destroyAfterSubmit and readBuffer
  private afterSubmitCallbacks: (() => void)[] = [];

  public profiler: Profiler = new Profiler();

  //defines every shader gets, so a shader can #ifdef on what the device supports. a define of the shader itself wins.
  //set in init, after the device is created:
  //HAS_SUBGROUPS: the device has the "subgroups" feature, so "enable subgroups;" compiles
  //HAS_SUBGROUP_ID: also the subgroup_id and num_subgroups builtins (WGSL language feature subgroup_id, Chrome 144).
  //  the shader then needs "requires subgroup_id;" as well
  public readonly shaderDefines: Map<string, boolean> = new Map();

  public init = async (canvasParent: HTMLElement, options: Partial<RendererOptions> = {}) => {
    let adapter = null;
    if (typeof navigator !== 'undefined' && navigator.gpu) {
      adapter = await navigator.gpu.requestAdapter();
    }
    if (adapter == null) {
      this.showWebgpuNotSupportedMessage(canvasParent);
      return;
    }

    this.canvasManager = new CanvasManager(canvasParent, options.pixelRatio);
    this.canvas = this.canvasManager.canvas;

    LogGL.log('Renderer: adapter.features', Array.from(adapter.features));
    LogGL.log('Renderer: adapter.limits', adapter.limits);

    this.options = {
      ...defaultRendererOptions,
      ...options,
      // Copy the mutable fields so we never mutate the shared module-level defaults
      // (a shallow spread aliases these array/object references when the caller omits them).
      requiredFeatures: [...(options.requiredFeatures ?? defaultRendererOptions.requiredFeatures)],
      requiredLimits: { ...(options.requiredLimits ?? defaultRendererOptions.requiredLimits) },
    };

    this.options.requiredFeatures.push('rg11b10ufloat-renderable');
    this.options.requiredFeatures.push('shader-f16');
    this.options.requiredFeatures.push('texture-formats-tier1');
    this.options.requiredFeatures.push('texture-formats-tier2');
    this.options.requiredFeatures.push('float32-filterable');
    this.options.requiredFeatures.push('float32-blendable');
    this.options.requiredFeatures.push('timestamp-query');
    //RadixSort and PrefixSum use it when the adapter has it, see shaderDefines
    this.options.requiredFeatures.push('subgroups');

    this.options.requiredFeatures = this.checkRequiredFeaturesAgainstAdapterFeatures(adapter, this.options.requiredFeatures);

    // Request max buffer size to handle Chrome/Dawn staging buffer requirements
    this.options.requiredLimits.maxBufferSize = adapter.limits.maxBufferSize;

    this.options.requiredLimits = this.checkRequiredLimitsAgainstAdapterLimits(adapter, this.options.requiredLimits);

    this.device = await adapter.requestDevice({
      requiredFeatures: this.options.requiredFeatures,
      requiredLimits: this.options.requiredLimits,
    });

    LogGL.log('Renderer: requiredFeatures', this.options.requiredFeatures);
    // LogGL.log('Renderer: device.features', Object.fromEntries(Array.from(this.device.features).map(f => [f, true])));
    LogGL.log('Renderer: device.features', Array.from(this.device.features));
    LogGL.log('Renderer: device.limits', this.device.limits);

    const hasSubgroups = this.device.features.has('subgroups');
    this.shaderDefines.set('HAS_SUBGROUPS', hasSubgroups);
    this.shaderDefines.set('HAS_SUBGROUP_ID', hasSubgroups && navigator.gpu.wgslLanguageFeatures.has('subgroup_id'));
    LogGL.log('Renderer: shaderDefines', Object.fromEntries(this.shaderDefines));

    //@ts-ignore
    this.context = this.canvas.getContext('webgpu');

    this.context.configure({
      device: this.device,
      format: this.options.format,
    });

    this.profiler.init(this.device, canvasParent);

    this.dummyBindings = new DummyBindings(this);

    this.minUniformBufferOffsetAlignment = this.device.limits.minUniformBufferOffsetAlignment;
    this.uniformData = new Float32Array(this.options.uniformBufferSize / 4);
    this.uniformBuffer = this.createBuffer(this.options.uniformBufferSize, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, false);

    this.vertexBufferLayoutManager = new VertexBufferLayoutManager();

    this.blitMesh = this.createBlitTriangle();

    this.copyTextureMaterial = new VertexFragmentShader(this, 'blit', blitShader);
    this.copyTextureMaterial.depthWriteEnabled = false;
    this.copyTextureMaterial.depthCompare = 'always';

    this.renderTexture = new RenderTexture(this, this.canvas.width, this.canvas.height, {
      formats: [this.options.format],
      depth: this.options.depth,
      stencil: this.options.stencil,
      sampleCount: this.options.sampleCount,
      toScreen: true,
      scaleToCanvas: true,
      discardMSAA: this.options.discardMSAA,
    });
    this.currentRenderTexture = this.renderTexture;
  };

  private showWebgpuNotSupportedMessage(canvasParent: HTMLElement) {
    console.warn('WebGPU not supported');

    const div = document.createElement('div');
    div.style.position = 'absolute';
    div.style.top = '0';
    div.style.left = '0';
    div.style.width = '100%';
    div.style.height = '100%';
    div.style.display = 'flex';
    div.style.flexDirection = 'column';
    div.style.justifyContent = 'center';
    div.style.alignItems = 'center';
    div.style.backgroundColor = '#1a1a1a';
    div.style.color = '#ffffff';
    div.style.zIndex = '1000';
    div.style.fontFamily = 'Inter, system-ui, sans-serif';

    div.innerHTML = `
        <h1 style="font-size: 2rem; margin-bottom: 1rem;">WebGPU Not Supported</h1>
        <p style="font-size: 1rem; color: #888;">Your browser does not appear to support WebGPU.</p>
      `;

    if (getComputedStyle(canvasParent).position === 'static') {
      canvasParent.style.position = 'relative';
    }
    canvasParent.appendChild(div);

  }

  public get renderTarget(): RenderTexture {
    return this.currentRenderTexture;
  }

  public get size(): Vector2 {
    return this.currentRenderTexture.size;
  }

  public get aspectRatio(): number {
    return this.width / this.height;
  }

  public get width(): number {
    return this.currentRenderTexture.width;
  }

  public get height(): number {
    return this.currentRenderTexture.height;
  }

  //of the current pass: the passes into the higher mip levels have no msaa
  public get sampleCount(): number {
    return this.currentRenderTexture.getSampleCount(this.currentRenderTextureMipLevel);
  }

  public get mouseListener(): MouseListener {
    if (!this._mouseListener) {
      //pass document.body as the first argument to add the events to the body
      this._mouseListener = new MouseListener(this.canvas, this.canvas);
    }
    return this._mouseListener;
  }

  public update(time: number) {
    this.time.update(time);

    //work recorded between frames (a texture that finished loading generates its mipmaps, a video copies a frame)
    //leaves its pass open and its render target current. The frame starts on the screen, like after finish()
    this.endPasses();
    this.currentRenderTexture = this.renderTexture;
    this.currentRenderTextureMipLevel = 0;

    const canvasUpdate = this.canvasManager.update();

    this.renderTexture.setSize(this.canvas.width, this.canvas.height);

    this.renderBundleCacheManager.update();

    //only once the app used it: creating it adds the listeners
    this._mouseListener?.update();

    this.start();

    return canvasUpdate;
  }

  //copies the shader's uniforms into the uniform buffer and returns their offset, to pass to setBindGroups.
  //when the uniforms didn't change since the shader's last draw or dispatch in this command encoder, that copy is reused
  public setUniformData(shader: ShaderBase): number {
    const data = shader.uniformData;
    if (!data) return 0;
    if (!shader.uniformsChanged && shader.uniformEpoch === this.uniformEpoch) return shader.uniformOffset;

    const offset = this.uniformDataOffset;
    const end = offset + data.byteLength;
    const alignment = this.minUniformBufferOffsetAlignment;
    this.uniformDataOffset = Math.ceil(end / alignment) * alignment;

    if (end > this.uniformData.byteLength) {
      //finish() grows the buffer after the submit. until then this draw aliases the uniforms at offset 0
      if (offset <= this.uniformData.byteLength) {
        console.error('Renderer: the uniform buffer is full, it grows after this submit. size:', this.uniformData.byteLength, 'shader:', shader.name);
      }
      return 0;
    }

    this.uniformData.set(data, offset / 4);
    this.uniformUploadSize = end;
    shader.uniformOffset = offset;
    shader.uniformEpoch = this.uniformEpoch;
    shader.uniformsChanged = false;
    return offset;
  }

  private growUniformBuffer(requiredSize: number) {
    let size = this.uniformData.byteLength;
    while (size < requiredSize) size *= 2;
    LogGL.log('Renderer: growing the uniform buffer to', size);

    //destroying is fine: the frames that still use the old buffer have already been submitted
    this.uniformBuffer.destroy();
    this.uniformBuffer = this.createBuffer(size, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, false);
    this.uniformData = new Float32Array(size / 4);
    this.uniformDataOffset = 0;
    this.uniformUploadSize = 0;
    this.uniformEpoch++;
  }

  public start(): GPUCommandEncoder {
    if (this._commandEncoder) {
      return this._commandEncoder;
    }
    this._commandEncoder = this.device.createCommandEncoder();
    return this._commandEncoder;
  }

  public finish() {
    if (!this._commandEncoder) {
      return;
    }
    this.endPasses();

    if (this.uniformUploadSize > 0) {
      this.device.queue.writeBuffer(this.uniformBuffer, 0, this.uniformData.buffer as ArrayBuffer, this.uniformData.byteOffset, this.uniformUploadSize);
    }
    this.uniformUploadSize = 0;
    this.uniformEpoch++;

    const resolution = this.profiler.resolve(this._commandEncoder);

    this.device.queue.submit([this._commandEncoder.finish()]);

    //right after the submit the buffer can be replaced: the work that used it is submitted and nothing of the next
    //command encoder has bound it yet. An overflowing uniformDataOffset tells the size it needs
    if (this.uniformDataOffset > this.uniformData.byteLength) {
      this.growUniformBuffer(this.uniformDataOffset);
    }
    //the next command encoder can reuse the whole buffer: its writeBuffer is ordered after this submit
    this.uniformDataOffset = 0;

    if (resolution) {
      this.profiler.read(resolution.queryReadBuffer, resolution.queryLabels, resolution.bufferIndex, resolution.frameCounter);
    }

    this._commandEncoder = null;
    this.currentRenderTexture = this.renderTexture;
    this.currentRenderTextureMipLevel = 0;

    const callbacks = this.afterSubmitCallbacks;
    this.afterSubmitCallbacks = [];
    for (const callback of callbacks) callback();
  }

  //destroys a resource that draws or dispatches recorded earlier in this frame may still use: destroying it right away
  //would make the submit fail. Destroying after the submit is fine, the gpu keeps it alive until that work is done
  public destroyAfterSubmit(resource: { destroy(): void }) {
    if (this._commandEncoder) {
      this.afterSubmitCallbacks.push(() => resource.destroy());
    } else {
      resource.destroy();
    }
  }

  public get commandEncoder(): GPUCommandEncoder {
    if (!this._commandEncoder) {
      return this.start();
    }
    return this._commandEncoder;
  }

  //drawing outside a render pass (after a dispatch, a copy or finish) continues in the last render target without clearing
  private get renderPassEncoder(): GPURenderPassEncoder {
    if (!this._renderPassEncoder) {
      if (!this.warnedAboutResumedPass) {
        this.warnedAboutResumedPass = true;
        console.warn('Renderer: draw outside a render pass, continuing in the last render target without clearing. Call startRenderPass before drawing to silence this');
      }
      return this.startRenderPass(this.currentRenderTexture, { colorClear: false, depthClear: false }, this.currentRenderTextureMipLevel, 'resumed pass');
    }
    return this._renderPassEncoder;
  }

  //starts a new pass in the current render target and mip level, clearing it
  public clear(options: ClearOptions = {}) {
    this.startRenderPass(this.currentRenderTexture, options, this.currentRenderTextureMipLevel, 'clear');
  }

  public startRenderPass(renderTexture: RenderTexture | null = null, options: ClearOptions = {}, mipLevel: number = 0, label: string = 'Render Pass'): GPURenderPassEncoder {
    const colorClear = options.colorClear ?? true;
    const clearColor = options.clearColor ?? this.clearColor;
    const depthClear = options.depthClear ?? true;
    const depthClearValue = options.depthClearValue ?? DEFAULT_DEPTH_CLEAR_VALUE;
    const stencilClear = options.stencilClear ?? depthClear;
    const stencilClearValue = options.stencilClearValue ?? 0;

    if (this._renderPassEncoder && (!colorClear && !depthClear && !stencilClear) && (this.currentRenderTexture == (renderTexture || this.renderTexture)) && mipLevel == this.currentRenderTextureMipLevel) return this._renderPassEncoder;
    this.endPasses();
    this.start();
    this.currentRenderTexture = renderTexture || this.renderTexture;
    if (this.currentRenderTexture.resizeToCanvas) this.currentRenderTexture.setSize(this.canvas.width, this.canvas.height);
    this.currentRenderTexture.clear(colorClear, clearColor, depthClear, depthClearValue, stencilClear, stencilClearValue);

    this.currentRenderTextureMipLevel = mipLevel;
    const renderPassDescriptor = this.currentRenderTexture.getRenderPassDescriptor(mipLevel);

    //the descriptor is reused: always assign, so a pass without a query does not keep the writes of an earlier one
    renderPassDescriptor.timestampWrites = this.profiler.getTimestampWrites(label);

    this._renderPassEncoder = this.commandEncoder.beginRenderPass(renderPassDescriptor);
    this.renderPassState.reset();
    if (this.currentRenderTexture.stencilEnabled) this._renderPassEncoder.setStencilReference(1);

    return this._renderPassEncoder;
  }

  private endRenderPass() {
    if (!this._renderPassEncoder) return;
    this._renderPassEncoder.end();
    this._renderPassEncoder = null;
  }

  //dispatchCompute continues in the pass this returns: set a pipeline or bind groups on it yourself and the next
  //dispatchCompute doesn't know, so start a new pass after that
  public startComputePass(label: string = 'Compute Pass'): GPUComputePassEncoder {
    this.endComputePass();
    this.endRenderPass();
    this.start();
    const descriptor: GPUComputePassDescriptor = {};
    const timestampWrites = this.profiler.getTimestampWrites(label);
    if (timestampWrites) {
      descriptor.timestampWrites = timestampWrites;
    }

    this._computePassEncoder = this.commandEncoder.beginComputePass(descriptor);
    this.computePassState.reset();
    return this._computePassEncoder;
  }

  private endComputePass() {
    if (!this._computePassEncoder) return;
    this._computePassEncoder.end();
    this._computePassEncoder = null;
  }

  //executes a render bundle in the current render pass, in order with the other draws
  public drawBundle(bundle: GPURenderBundle) {
    this.renderPassEncoder.executeBundles([bundle]);
    this.renderPassState.reset();
  }

  public endPasses() {
    this.endRenderPass();
    this.endComputePass();
  }

  //the attachment state of a render bundle drawn into the current render target, see drawCachedBundle. only built again
  //when that changes
  private getAttachmentKey(rt: RenderTexture, depthStencilFormat: GPUTextureFormat | undefined, sampleCount: number): string {
    const cache = this.attachmentKeyCache;
    const mipLevel = this.currentRenderTextureMipLevel;
    if (cache.rt !== rt || cache.mipLevel !== mipLevel || cache.depthStencilFormat !== depthStencilFormat || cache.sampleCount !== sampleCount) {
      cache.rt = rt;
      cache.mipLevel = mipLevel;
      cache.depthStencilFormat = depthStencilFormat;
      cache.sampleCount = sampleCount;
      cache.key = `${rt.formats.join('_')}_${sampleCount}_${depthStencilFormat}_${mipLevel}`;
    }
    return cache.key;
  }

  //sets everything a draw needs on the render pass, skipping what is already set
  private prepareDraw(material: VertexFragmentShader, mesh: Mesh | null, indexBuffer: GPUBuffer | null, indexFormat: GPUIndexFormat): GPURenderPassEncoder {
    const uniformBufferOffset = this.setUniformData(material);
    const pass = this.renderPassEncoder;
    const state = this.renderPassState;

    if (mesh) {
      //before the pipeline: it depends on the layout
      material.setVertexBufferLayout(this.vertexBufferLayoutManager.getVertexBufferLayout(mesh, material));
      if (state.vertexBuffer !== mesh.vertexBuffer) {
        pass.setVertexBuffer(0, mesh.vertexBuffer);
        state.vertexBuffer = mesh.vertexBuffer;
      }
    } else {
      //no vertex buffer: the layout of a mesh the material was drawn with before would make the pipeline expect one
      material.clearVertexBufferLayout();
    }

    if (indexBuffer && (state.indexBuffer !== indexBuffer || state.indexFormat !== indexFormat)) {
      pass.setIndexBuffer(indexBuffer, indexFormat);
      state.indexBuffer = indexBuffer;
      state.indexFormat = indexFormat;
    }

    const pipeline = material.pipeline;
    if (state.pipeline !== pipeline) {
      pass.setPipeline(pipeline);
      state.pipeline = pipeline;
    }

    material.setBindGroups(pass, uniformBufferOffset, state.bindGroups);
    return pass;
  }

  /*
  records the draws of renderCallback into a render bundle once and executes that bundle in later frames. it is
  re-recorded when the render target, the material's pipeline state or bindings, or the offset of its uniforms change.
  only worth it for many draws that share one material (like a list of drawIndexedIndirect calls): a single draw is as
  fast to encode directly. renderCallback sets the vertex and index buffers and the pipeline, the bind groups are set
  key: tells apart the bundles drawn with this material, e.g. the shadow map cascade
   */
  public drawCachedBundle(key: string | number, material: VertexFragmentShader, renderCallback: (bundleEncoder: GPURenderBundleEncoder) => void) {
    const uniformBufferOffset = this.setUniformData(material);
    const pass = this.renderPassEncoder;
    const rt = this.currentRenderTexture;
    const depthStencilFormat = this.currentRenderTextureMipLevel == 0 ? rt.depthStencilAttachment?.format : undefined;
    const sampleCount = rt.getSampleCount(this.currentRenderTextureMipLevel);
    const attachmentKey = this.getAttachmentKey(rt, depthStencilFormat, sampleCount);
    const bindGroupIds = material.getBindGroupIds(this.bindGroupIds);

    let bundle = this.renderBundleCacheManager.get(material.id, key, attachmentKey, material.version, bindGroupIds, uniformBufferOffset);
    if (!bundle) {
      const bundleEncoder = this.device.createRenderBundleEncoder({
        colorFormats: rt.formats,
        depthStencilFormat,
        sampleCount,
      });
      material.setBindGroups(bundleEncoder, uniformBufferOffset);
      renderCallback(bundleEncoder);
      bundle = bundleEncoder.finish();
      bundle.label = 'bundle_' + material.name + '_' + key;
      this.renderBundleCacheManager.set(material.id, key, attachmentKey, material.version, bindGroupIds, uniformBufferOffset, bundle);
    }

    pass.executeBundles([bundle]);
    this.renderPassState.reset();
  }

  public draw(mesh: Mesh, material: VertexFragmentShader, options: Partial<DrawOptions> = {}) {
    const instanceCount = options.instanceCount || 1;
    const pass = this.prepareDraw(material, mesh, mesh.indexBuffer, mesh.indexFormat);
    if (mesh.indexBuffer) {
      pass.drawIndexed(mesh.indexCount, instanceCount);
    } else {
      pass.draw(mesh.vertexCount, instanceCount, 0, 0);
    }
  }

  public drawIndirect(mesh: Mesh, material: VertexFragmentShader, indirectBuffer: GPUBuffer, indirectOffset: GPUSize64 = 0) {
    const pass = this.prepareDraw(material, mesh, mesh.indexBuffer, mesh.indexFormat);
    if (mesh.indexBuffer) {
      pass.drawIndexedIndirect(indirectBuffer, indirectOffset);
    } else {
      pass.drawIndirect(indirectBuffer, indirectOffset);
    }
  }

  public drawVertices(vertexCount: number, material: VertexFragmentShader, indexBuffer: GPUBuffer | null = null, options: Partial<DrawVerticesOptions> = {}) {
    const instanceCount = options.instanceCount || 1;
    const firstVertex = options.firstVertex || 0;
    const firstInstance = options.firstInstance || 0;
    const firstIndex = options.firstIndex || 0;

    const pass = this.prepareDraw(material, null, indexBuffer, options.indexFormat ?? 'uint32');
    if (indexBuffer) {
      pass.drawIndexed(vertexCount, instanceCount, firstIndex, firstVertex, firstInstance);
    } else {
      pass.draw(vertexCount, instanceCount, firstVertex, firstInstance);
    }
  }

  public drawVerticesIndirect(
    material: VertexFragmentShader,
    indexBuffer: GPUBuffer | null = null,
    indirectBuffer: GPUBuffer,
    indirectOffset: GPUSize64 = 0,
    indexFormat: GPUIndexFormat = 'uint32'
  ) {
    const pass = this.prepareDraw(material, null, indexBuffer, indexFormat);
    if (indexBuffer) {
      pass.drawIndexedIndirect(indirectBuffer, indirectOffset);
    } else {
      pass.drawIndirect(indirectBuffer, indirectOffset);
    }
  }

  public blit(source: Texture | GPUTextureView | null = null, destination: RenderTexture | null = null, material: VertexFragmentShader | null = null, clear: boolean = true, mipLevel: number = 0) {
    material = material || this.copyTextureMaterial;

    if (source) {
      //a Texture is bound by reference, see ShaderBase.setTexture
      material.setTexture('texture', source);
    }
    this.startRenderPass(destination, { colorClear: clear, depthClear: clear }, mipLevel, 'blit ' + material.name);
    this.draw(this.blitMesh, material);
  }

  //destination texture needs GPUTextureUsage.RENDER_ATTACHMENT
  public blitMipToMip(source: Texture, sourceMipLevel: number, destination: RenderTexture, destinationMipLevel: number, material: VertexFragmentShader | null = null) {
    material = material || this.copyTextureMaterial;
    material.setBinding('texture', source.getMipView(sourceMipLevel));

    this.startRenderPass(destination, { colorClear: true, depthClear: true }, destinationMipLevel, 'blitMipToMip');
    this.draw(this.blitMesh, material);
  }

  public createMipsForRenderTexture(texture: RenderTexture, material: VertexFragmentShader | null = null) {
    for (let i = 1; i < texture.mipCount; i++) {
      this.blitMipToMip(texture, i - 1, texture, i, material);
    }
  }

  /*
  consecutive dispatches share one compute pass, which saves beginning and ending a pass for each of them. That is safe:
  every dispatch is a usage scope of its own, so it sees what the dispatches before it in the pass wrote. While the
  profiler records, every dispatch gets a pass of its own, so it is timed on its own.
  sets the pipeline and the bind groups, skipping what is already set on the pass
   */
  private prepareDispatch(shader: ComputeShader): GPUComputePassEncoder {
    const uniformBufferOffset = this.setUniformData(shader);
    const pass = this._computePassEncoder && !this.profiler.recording ? this._computePassEncoder : this.startComputePass(shader.name);
    const state = this.computePassState;

    const pipeline = shader.pipeline;
    if (state.pipeline !== pipeline) {
      pass.setPipeline(pipeline);
      state.pipeline = pipeline;
    }
    shader.setBindGroups(pass, uniformBufferOffset, state.bindGroups);
    return pass;
  }

  public dispatchCompute(shader: ComputeShader, groupCountX: number = 1, groupCountY = 1, groupCountZ = 1) {
    this.prepareDispatch(shader).dispatchWorkgroups(groupCountX, groupCountY, groupCountZ);
  }

  public dispatchComputeIndirect(shader: ComputeShader, indirectBuffer: GPUBuffer) {
    this.prepareDispatch(shader).dispatchWorkgroupsIndirect(indirectBuffer, 0);
  }

  public copyTextureToTexture(source: GPUTexture, destination: GPUTexture, sourceMipLevel: number = 0, destinationMipLevel: number = 0) {
    this.endPasses();
    const src: GPUImageCopyTexture = {
      texture: source,
      mipLevel: sourceMipLevel,
    };
    const dest: GPUImageCopyTexture = {
      texture: destination,
      mipLevel: destinationMipLevel,
    };
    //the size of the source mip level. the layers of an array don't shrink, the depth of a 3d texture does
    const width = Math.max(1, source.width >> sourceMipLevel);
    const height = Math.max(1, source.height >> sourceMipLevel);
    const depth = source.dimension == '3d' ? Math.max(1, source.depthOrArrayLayers >> sourceMipLevel) : source.depthOrArrayLayers;
    this.commandEncoder.copyTextureToTexture(src, dest, [width, height, depth]);
  }

  public setClearColor(r: number, g: number, b: number, a: number) {
    this.clearColor.setValues(r, g, b, a);
  }

  public createBuffer(size: GPUSize64, usage: GPUBufferUsageFlags, mappedAtCreation = false, label: string | null = null): GPUBuffer {
    return this.device.createBuffer({ size, usage, mappedAtCreation, label: label ? label : 'buffer_' + this.resourceID++ });
  }

  public writeBuffer(buffer: GPUBuffer, data: ArrayBuffer | ArrayBufferView) {
    if (data instanceof ArrayBuffer) {
      this.device.queue.writeBuffer(buffer, 0, data);
      return;
    }

    const sourceBuffer = data.buffer;
    if (!(sourceBuffer instanceof ArrayBuffer)) {
      throw new Error('Renderer.writeBuffer: SharedArrayBuffer-backed views are not supported');
    }

    this.device.queue.writeBuffer(buffer, 0, sourceBuffer, data.byteOffset, data.byteLength);
  }

  public copyBufferToBuffer(source: GPUBuffer, destination: GPUBuffer, sourceOffset: number = 0, destinationOffset = 0, size = source.size) {
    this.endPasses();
    this.commandEncoder.copyBufferToBuffer(source, sourceOffset, destination, destinationOffset, size);
  }

  //offset and size in bytes, multiples of 4
  public clearBuffer(buffer: GPUBuffer, offset = 0, size = buffer.size - offset) {
    this.endPasses();
    this.commandEncoder.clearBuffer(buffer, offset, size);
  }

  //texture needs GPUTextureUsage.RENDER_ATTACHMENT. Doesn't change the render target: when this is called during a pass
  //(e.g. setting the image of a texture), the draws after it continue in the current target, not in the last mip
  public generateMipmaps(texture: Texture, material?: VertexFragmentShader) {
    if (!this.mipmapGenerator) {
      this.mipmapGenerator = new WebGPUMipmapGenerator(this);
    }
    const renderTexture = this.currentRenderTexture;
    const mipLevel = this.currentRenderTextureMipLevel;
    const inRenderPass = this._renderPassEncoder !== null;

    this.mipmapGenerator.generateMipmaps(texture, material);

    if (inRenderPass) {
      this.startRenderPass(renderTexture, { colorClear: false, depthClear: false }, mipLevel, 'resumed pass');
    } else {
      this.endPasses();
      this.currentRenderTexture = renderTexture;
      this.currentRenderTextureMipLevel = mipLevel;
    }
  }

  //when the requested limit is higher than the supported limit: the supported limit is used
  private checkRequiredLimitsAgainstAdapterLimits(adapter: GPUAdapter, requiredLimits: Record<string, GPUSize64>) {
    const adapterLimits = adapter.limits;

    for (const [key, value] of Object.entries(requiredLimits)) {
      if (key in adapterLimits) {
        const limit = <number>adapterLimits[key as keyof GPUSupportedLimits];
        if (limit < value) {
          console.warn('Renderer: checkRequiredLimitsAgainstAdapterLimits: limit exceeded', key, value, limit);
          requiredLimits[key] = limit;
        }
      } else {
        console.warn('Renderer: checkRequiredLimitsAgainstAdapterLimits: key does not exist', key, value);
        // If the limit is not present in the adapter limits, remove it from required limits
        delete requiredLimits[key];
      }
    }
    return requiredLimits;
  }

  private checkRequiredFeaturesAgainstAdapterFeatures(adapter: GPUAdapter, requiredFeatures: GPUFeatureName[]): GPUFeatureName[] {
    const adapterFeatures = adapter.features;

    return requiredFeatures.filter(feature => {
      if (!adapterFeatures.has(feature)) {
        console.warn('Renderer: checkRequiredFeaturesAgainstAdapterFeatures: feature not supported', feature);
        return false;
      }
      return true;
    });
  }

  private createBlitTriangle(): Mesh {
    const triangle = new Mesh(this);
    triangle.setPositions(new Float32Array([-1, -1, 0, -1, 3, 0, 3, -1, 0]));
    triangle.setUV0(new Float32Array([0, 1, 0, -1, 2, 1]));
    triangle.setIndices([0, 2, 1]);
    return triangle;
  }

  //reads the buffer as written by the work recorded so far. with an open command encoder it resolves only after the
  //next finish(), so don't await it before the frame (or the init code that dispatched) calls finish(): call finish()
  //first, or don't await it there
  public async readBuffer<T extends Float32Array | Uint32Array | Int32Array | Uint8Array | Uint16Array>(
    buffer: GPUBuffer,
    arrayType: { new(buffer: ArrayBufferLike): T },
    logName: string | null = null,
  ): Promise<T> {
    const bufferRead = this.createBuffer(buffer.size, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);

    if (this._commandEncoder) {
      //the copy goes into the open command encoder, so it reads what the work recorded before it wrote (a separate
      //submit would run before that work). The staging buffer can only be mapped once that encoder is submitted
      this.endPasses();
      this._commandEncoder.copyBufferToBuffer(buffer, 0, bufferRead, 0, buffer.size);
      await new Promise<void>((resolve) => this.afterSubmitCallbacks.push(resolve));
    } else {
      const encoder = this.device.createCommandEncoder();
      encoder.copyBufferToBuffer(buffer, 0, bufferRead, 0, buffer.size);
      this.device.queue.submit([encoder.finish()]);
    }

    await bufferRead.mapAsync(GPUMapMode.READ, 0, bufferRead.size);
    const mappedRange = bufferRead.getMappedRange(0, bufferRead.size);
    const data = new arrayType(mappedRange.slice(0));
    bufferRead.unmap();
    bufferRead.destroy();

    if (logName !== null) {
      console.log(logName, 'read buffer', data);
    }
    return data;
  }

  public destruct() {
    //only the screen target is owned by the renderer, currentRenderTexture can be a render texture of the app
    this.renderTexture.destruct();
    this.blitMesh.destruct();
    this._mouseListener?.destruct();
    this.profiler.destruct();
    this.canvasManager.destruct();
    this.device.destroy();
  }
}
