import Renderer from '../render/Renderer';

//big enough for the structs that shaders bind in practice: a binding smaller than its struct fails validation
const DUMMY_BUFFER_SIZE = 4096;

export default class DummyBindings {
  // Keyed by sample type, view dimension and multisampling: one dummy per texture binding shape.
  private dummyTextureViews: Map<string, GPUTextureView> = new Map();
  // Keyed by format, view dimension and whether it is writable: one dummy per storage texture binding shape.
  private dummyStorageTextureViews: Map<string, GPUTextureView> = new Map();
  //by binding type. a resource bound as writable storage and as read-only (uniform, read-only storage) in the same
  //usage scope fails validation, so the writable bindings get dummies of their own. writable bindings can share one
  private dummyBuffers: Map<GPUBufferBindingType, GPUBuffer> = new Map();
  private dummySampler!: GPUSampler;
  private dummyComparisonSampler!: GPUSampler;
  private renderer: Renderer;

  constructor(renderer: Renderer) {
    this.renderer = renderer;
  }

  public getDummyBindData(entry: GPUBindGroupLayoutEntry, shaderName: string): GPUBuffer | GPUSampler | GPUTextureView {
    if (entry.sampler) {
      return entry.sampler.type === 'comparison' ? this.getDummyComparisonSampler() : this.getDummySampler();
    }

    if (entry.texture) {
      return this.getDummyTextureView(entry.texture);
    }

    if (entry.storageTexture) {
      return this.getDummyStorageTextureView(entry.storageTexture);
    }

    if (entry.buffer) {
      return this.getDummyBuffer(entry.buffer.type);
    }

    if (entry.externalTexture) {
      //there is no such thing as an empty external texture, it can only come from a video source
      console.warn(shaderName, 'DummyBindings: an external texture can not be substituted, call setExternalTexture before drawing', entry);
      return this.getDummyBuffer();
    }

    console.warn(shaderName, 'DummyBindings: unhandled: ', entry);
    return this.getDummyBuffer();
  }

  //a 1x1 texture with a format of the binding's sample type, and the layers its view dimension needs
  public getDummyTextureView(texture: GPUTextureBindingLayout): GPUTextureView {
    const sampleType: GPUTextureSampleType = texture.sampleType ?? 'float';
    const viewDimension: GPUTextureViewDimension = texture.viewDimension ?? '2d';
    const multisampled = texture.multisampled ?? false;
    const key = `${sampleType}_${viewDimension}_${multisampled}`;
    let view = this.dummyTextureViews.get(key);

    if (!view) {
      //r8unorm is filterable, so it also matches 'unfilterable-float'. the 8 bit integer formats can be multisampled, r32uint can't
      const format: GPUTextureFormat = sampleType === 'depth' ? 'depth24plus' : sampleType === 'uint' ? 'r8uint' : sampleType === 'sint' ? 'r8sint' : 'r8unorm';
      const layers = viewDimension === 'cube' || viewDimension === 'cube-array' ? 6 : 1;
      const dimension: GPUTextureDimension = viewDimension === '1d' ? '1d' : viewDimension === '3d' ? '3d' : '2d';
      view = this.renderer.device.createTexture({
        size: [1, 1, layers],
        format,
        //a multisampled texture has to be renderable
        usage: GPUTextureUsage.TEXTURE_BINDING | (multisampled ? GPUTextureUsage.RENDER_ATTACHMENT : 0),
        dimension,
        sampleCount: multisampled ? 4 : 1,
      }).createView({dimension: viewDimension});
      view.label = `dummyTextureView_${key}_${this.renderer.resourceID++}`;
      this.dummyTextureViews.set(key, view);
    }
    return view;
  }

  public getDummySampler(): GPUSampler {
    if (!this.dummySampler) {
      this.dummySampler = this.renderer.device.createSampler({
        magFilter: 'linear',
        minFilter: 'linear',
        mipmapFilter: 'linear',
        addressModeU: "repeat",
        addressModeV: "repeat",
        addressModeW: "repeat",
      });
    }
    return this.dummySampler;
  }

  public getDummyComparisonSampler(): GPUSampler {
    if (!this.dummyComparisonSampler) {
      this.dummyComparisonSampler = this.renderer.device.createSampler({
        magFilter: 'linear',
        minFilter: 'linear',
        compare: 'less',
      });
    }
    return this.dummyComparisonSampler;
  }

  public getDummyStorageTextureView(entry: GPUStorageTextureBindingLayout): GPUTextureView {
    const viewDimension: GPUTextureViewDimension = entry.viewDimension ?? '2d';
    const writable = (entry.access ?? 'write-only') !== 'read-only';
    const key = `${entry.format}_${viewDimension}_${writable}`;
    let view = this.dummyStorageTextureViews.get(key);

    if (!view) {
      // Storage textures are never cube, so the array dimensions map onto a single layer.
      const dimension: GPUTextureDimension = viewDimension === '1d' ? '1d' : viewDimension === '3d' ? '3d' : '2d';
      view = this.renderer.device.createTexture({
        size: [1, 1, 1],
        format: entry.format,
        usage: GPUTextureUsage.STORAGE_BINDING,
        dimension,
      }).createView({dimension: viewDimension});
      view.label = `dummyStorageTextureView_${key}_${this.renderer.resourceID++}`;
      this.dummyStorageTextureViews.set(key, view);
    }
    return view;
  }

  public getDummyBuffer(type: GPUBufferBindingType = 'storage'): GPUBuffer {
    let buffer = this.dummyBuffers.get(type);
    if (!buffer) {
      buffer = this.renderer.device.createBuffer({
        size: DUMMY_BUFFER_SIZE,
        usage: (type === 'uniform' ? GPUBufferUsage.UNIFORM : GPUBufferUsage.STORAGE) | GPUBufferUsage.COPY_DST,
        label: `dummyBuffer_${type}_${this.renderer.resourceID++}`,
      });
      this.dummyBuffers.set(type, buffer);
    }
    return buffer;
  }
}
