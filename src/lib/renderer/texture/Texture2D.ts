import Renderer from '../render/Renderer';
import Texture, {TextureOptions} from './Texture';
import {getFromTextureData, getImageSize} from 'monks-webgpu/utils/image/ImageUtils';
import LogGL from '../core/LogGL';

export default class Texture2D extends Texture {

  constructor(renderer: Renderer, width: number = 1, height: number = 1, options: Partial<TextureOptions> = {}) {
    super(renderer, width, height, 1, options);
  }

  //format defaults to the format of the texture. with useMips the mips are generated from the data, when the format
  //allows it (see Texture.canGenerateMipmaps). otherwise only mip 0 is set
  public setData(data: GPUAllowSharedBufferSource, width: number, height: number, format: GPUTextureFormat = this.options.format) {
    const generateMips = this.options.useMips && Texture.canGenerateMipmaps(this.renderer.device, format);
    //generateMipmaps renders into the mips
    const requiredUsage = generateMips ? GPUTextureUsage.RENDER_ATTACHMENT : 0;

    if (width != this.width || height != this.height || format != this.format || (this.gpuTexture.usage & requiredUsage) != requiredUsage) {
      this.options.format = format;
      this.options.usage |= requiredUsage;
      this.createTexture(width, height);
    }

    const formatInfo = this.getFormatBlockInfo(format);

    this.renderer.device.queue.writeTexture(
      {texture: this.gpuTexture},
      data,
      {bytesPerRow: Math.ceil(width / formatInfo.blockWidth) * formatInfo.bytesPerBlock},
      {width: width, height: height},
    );

    if (generateMips) {
      this.renderer.generateMipmaps(this);
    } else if (this.options.useMips) {
      LogGL.warn('Texture2D.setData: the mips of', format, 'can not be generated, only mip 0 is set');
    }
  }

  private getFormatBlockInfo(format: GPUTextureFormat): { bytesPerBlock: number; blockWidth: number } {
    // ASTC: always 16 bytes/block, block width encoded in name
    const astc = format.match(/^astc-(\d+)x\d+/);
    if (astc) return {bytesPerBlock: 16, blockWidth: parseInt(astc[1])};

    // BC / ETC2 / EAC: 4×4 blocks, 8 or 16 bytes per block
    if (/^(bc|etc2|eac)/.test(format)) {
      const is8 = /^(bc[14]-|etc2-rgb8|eac-r11)/.test(format);
      return {bytesPerBlock: is8 ? 8 : 16, blockWidth: 4};
    }

    // Depth / stencil
    const ds: Record<string, number> = {
      'stencil8': 1, 'depth16unorm': 2, 'depth24plus': 4,
      'depth32float': 4, 'depth24plus-stencil8': 4, 'depth32float-stencil8': 8,
    };
    if (format in ds) return {bytesPerBlock: ds[format], blockWidth: 1};

    // Packed formats
    if (/^(rgb9e5|rgb10a2|rg11b10)/.test(format))
      return {bytesPerBlock: 4, blockWidth: 1};

    // Standard uncompressed: channels × (bits / 8)
    const m = format.match(/^(r|rg|rgba|bgra)(\d+)/);
    if (m) {
      const ch: Record<string, number> = {r: 1, rg: 2, rgba: 4, bgra: 4};
      return {bytesPerBlock: ch[m[1]] * (parseInt(m[2]) / 8), blockWidth: 1};
    }

    console.warn('Unsupported format', format);
    return {bytesPerBlock: 4, blockWidth: 1};
  }

  //the pixels of an rgba8unorm texture, 4 bytes per pixel without row padding. finishes the current command encoder.
  //the texture needs GPUTextureUsage.COPY_SRC
  public getData(): Promise<Uint8Array> {
    return getFromTextureData(this.renderer, this.gpuTexture);
  }

  public setImage(image: GPUCopyExternalImageSource, flipY: boolean = false) {
    const {x: width, y: height} = getImageSize(image);
    //a video has no frame decoded yet
    if (width == 0 || height == 0) return;

    //copyExternalImageToTexture requires both of these on the destination
    const requiredUsage = GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT;

    if (!this.gpuTexture || this.gpuTexture.width != width || this.gpuTexture.height != height
      || (this.gpuTexture.usage & requiredUsage) != requiredUsage) {
      this.options.usage |= requiredUsage;
      this.createTexture(width, height);
    }

    this.renderer.device.queue.copyExternalImageToTexture(
      { source: image, flipY: flipY },
      { texture: this.gpuTexture },
      [width, height]
    );

    if (this.options.useMips) {
      this.renderer.generateMipmaps(this);
    }
  }
}