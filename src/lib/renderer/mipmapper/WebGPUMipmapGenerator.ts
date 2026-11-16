import Renderer from '../render/Renderer';
import LogGL from '../core/LogGL';
import Texture from '../texture/Texture';
import RenderTexture from '../texture/RenderTexture';
import VertexFragmentShader from '../material/VertexFragmentShader';
import blitShader from 'monks-webgpu/shaders/blit.wgsl?raw';

//renders into a mip level of one array layer of any texture, so the renderer can draw into it like into a RenderTexture
class MipRenderTarget extends RenderTexture {
  public texture!: Texture;
  public arrayLayer: number = 0;

  constructor(renderer: Renderer, format: GPUTextureFormat) {
    //a size of 0 creates no attachments of its own
    super(renderer, 0, 0, { formats: [format] });
  }

  public getRenderPassDescriptor(mipLevel: number): GPURenderPassDescriptor {
    return {
      colorAttachments: [{
        view: this.texture.getMipView(mipLevel, this.arrayLayer),
        loadOp: 'clear',
        storeOp: 'store',
      }],
    };
  }
}

export default class WebGPUMipmapGenerator {
  private renderer: Renderer;
  private material: VertexFragmentShader;
  private targetByFormat: Map<GPUTextureFormat, MipRenderTarget> = new Map();

  constructor(renderer: Renderer) {
    this.renderer = renderer;
    this.material = new VertexFragmentShader(renderer, 'mipmap', blitShader);
    //clamp-to-edge (the default), so the edges don't bleed into each other
    this.material.setSampler('textureSampler', renderer.device.createSampler({ minFilter: 'linear', magFilter: 'linear' }));
  }

  //renders every mip level from the level above it, for each array layer. the texture needs RENDER_ATTACHMENT usage
  //material: samples 'texture' to create the mip, by default a linear downsample
  public generateMipmaps(texture: Texture, material: VertexFragmentShader = this.material) {
    if (texture.gpuTexture.dimension != '2d') {
      throw new Error('Generating mipmaps for non-2d textures is currently unsupported!');
    }
    if (!(texture.gpuTexture.usage & GPUTextureUsage.RENDER_ATTACHMENT)) {
      LogGL.warn('generateMipmaps: texture was not created with RENDER_ATTACHMENT usage', texture);
      return;
    }
    if (!Texture.canGenerateMipmaps(this.renderer.device, texture.format)) {
      LogGL.warn('generateMipmaps: the mips of this format can not be rendered', texture.format);
      return;
    }

    let target = this.targetByFormat.get(texture.format);
    if (!target) {
      target = new MipRenderTarget(this.renderer, texture.format);
      this.targetByFormat.set(texture.format, target);
    }
    target.texture = texture;

    for (let arrayLayer = 0; arrayLayer < texture.depth; arrayLayer++) {
      target.arrayLayer = arrayLayer;
      for (let i = 1; i < texture.mipCount; i++) {
        this.renderer.blit(texture.getMipView(i - 1, arrayLayer), target, material, true, i);
      }
    }
  }
}
