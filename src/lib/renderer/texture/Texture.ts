import Renderer from '../render/Renderer';
import Vector2 from '../math/Vector2';

export type TextureOptions ={
  format: GPUTextureFormat;
  useMips: boolean;
  usage: GPUTextureUsageFlags;
  dimension: GPUTextureDimension;
  viewDimension: GPUTextureViewDimension;
  createTexture:boolean;
}

export const defaultTextureOptions:TextureOptions = {
  format: 'rgba8unorm',
  useMips: false,
  usage: typeof GPUTextureUsage !== 'undefined' ? GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST : 0,
  dimension: '2d',
  viewDimension: '2d',
  createTexture:true,
};

export default class Texture{
  protected renderer: Renderer;
  public _gpuTexture!: GPUTexture;
  public view!: GPUTextureView;
  public id: number = 0;
  protected _options: TextureOptions;
  //by arrayLayer * 32 + mipLevel (a texture has at most 32 mips), a number so a lookup builds no string
  protected mipViews: Map<number, GPUTextureView> = new Map();

  constructor(renderer: Renderer, width: number = 1, height: number = 1, depth = 1, options: Partial<TextureOptions>={}) {
    this.id = renderer.instanceID++;
    this.renderer = renderer;

    this._options = {...defaultTextureOptions, ...options};

    //when creating a render texture, we don't want to create the texture here
    if(this._options.createTexture) {
      this.createTexture(width, height, depth);
    }
  }

  protected get options(): TextureOptions {
    return this._options;
  }

  public setSize(width: number, height: number, depth: number = 1){
    if(this.width === width && this.height === height && this.depth === depth) return;
    this.createTexture(width, height, depth);
  }

  protected createTexture( width: number, height: number, depth: number = 1){
    this.clearMipViews();
    //draws earlier in this frame can still use the old texture
    if (this._gpuTexture) this.renderer.destroyAfterSubmit(this._gpuTexture);
    this._gpuTexture = this.renderer.device.createTexture({
      size: [width, height, depth],
      format: this.options.format,
      usage: this.options.usage,
      dimension: this.options.dimension,
      mipLevelCount: this.getMipCount(width, height, depth),
    });

    this.view = this.gpuTexture.createView({
      dimension: this.options.viewDimension,
      baseMipLevel: 0,
      mipLevelCount: this.gpuTexture.mipLevelCount,
    });
    this.view.label = `Texture_`+ this.renderer.resourceID++;
  }

  public clearMipViews(){
    this.mipViews = new Map();
  }

  public getMipView(mipLevel: number, arrayLayer: number = 0){
    const id = arrayLayer * 32 + mipLevel;
    let mipView = this.mipViews.get(id);

    if(!mipView){
      mipView = this.gpuTexture.createView(Texture.getMipViewDescriptor(this.gpuTexture, mipLevel, arrayLayer));
      this.mipViews.set(id, mipView);
    }
    return mipView;
  }

  //one mip level of one layer. a 3d texture has no layers: its view is the whole 3d mip level
  public static getMipViewDescriptor(texture: GPUTexture, mipLevel: number, arrayLayer: number = 0): GPUTextureViewDescriptor {
    if (texture.dimension == '3d') {
      return {baseMipLevel: mipLevel, mipLevelCount: 1, dimension: '3d'};
    }
    return {baseMipLevel: mipLevel, mipLevelCount: 1, dimension: '2d', baseArrayLayer: arrayLayer, arrayLayerCount: 1};
  }

  //whether Renderer.generateMipmaps can render the mips of a texture of this format: it has to be renderable, and
  //filterable for the linear downsample. not: compressed, depth / stencil and integer formats
  public static canGenerateMipmaps(device: GPUDevice, format: GPUTextureFormat): boolean {
    if (/^(bc|etc2|eac|astc)/.test(format) || /^(depth|stencil)/.test(format) || /[us]int$/.test(format) || format == 'rgb9e5ufloat') {
      return false;
    }
    //renderable only with these features
    if (/(snorm|16unorm)$/.test(format)) return device.features.has('texture-formats-tier1');
    if (format == 'rg11b10ufloat') return device.features.has('rg11b10ufloat-renderable');
    if (format.includes('32float')) return device.features.has('float32-filterable');
    return true;
  }

  protected getMipCount( width: number, height: number, depth: number = -1){
    if(!this.options.useMips) return 1;
    let size = Math.max(width, height);
    // if(depth > 0) size = Math.max(size, depth);
    return Math.floor(Math.log2(size)) + 1;
  }

  public get gpuTexture(): GPUTexture {
    return this._gpuTexture;
  }

  public set gpuTexture(value: GPUTexture) {
    this._gpuTexture = value;
  }

  public get mipCount():number{
    return this.gpuTexture.mipLevelCount;
  }

  public get width(){
    if(!this.gpuTexture) return 0;
    return this.gpuTexture.width;
  }

  public get height(){
    if(!this.gpuTexture) return 0;
    return this.gpuTexture.height;
  }

  public get depth(){
    if(!this.gpuTexture) return 0;
    return this.gpuTexture.depthOrArrayLayers;
  }

  public get size(){
    return new Vector2(this.width, this.height);
  }

  public get aspectRatio(){
    return this.width / this.height;
  }

  public get format(): GPUTextureFormat{
    return this.gpuTexture.format;
  }

  //after the submit: draws earlier in this frame can still use the texture
  public destruct() {
    if (this._gpuTexture) this.renderer.destroyAfterSubmit(this._gpuTexture);
  }
}
