import Renderer from '../render/Renderer';
import Color from '../math/Color';
import LogGL from '../core/LogGL';
import Texture, { TextureOptions } from './Texture';
import VertexFragmentShader from '../material/VertexFragmentShader';

export type RenderTextureOptions = TextureOptions & {
  //renders to the canvas
  toScreen: boolean;
  //msaa when above 1
  sampleCount: number;
  depth: boolean;
  //the format of the depth attachment with depth and without stencil. depth32float can be copied, so it is also
  //created with COPY_SRC
  depthFormat: GPUTextureFormat;
  stencil: boolean;
  //of the color attachments, one attachment per format
  formats: GPUTextureFormat[];
  //resized to the canvas when a pass into it starts
  scaleToCanvas: boolean;
  //the size passed to setSize is multiplied by this
  sizeMultiplier: number;
  //above 1: a 3d texture, rendered one slice at a time (see setSlice)
  depthOrArrayLength: number;
  //a single layer and mip view of a texture owned elsewhere, rendered into instead of creating color attachments.
  //formats should hold its format, sampleCount 1
  colorTarget: GPUTextureView | null;
  //with msaa: the multisampled color attachments are not stored after they are resolved, which saves a lot of bandwidth.
  //a later pass into this target that doesn't clear the color then starts from undefined samples
  discardMSAA: boolean;
  //the depth is not stored at the end of a pass, which saves bandwidth when nothing samples it or loads it in a later
  //pass. A later pass into this target that doesn't clear the depth then starts from undefined values
  discardDepth: boolean;
}

const defaultRenderTextureOptions: Partial<RenderTextureOptions> = {
  toScreen: false,
  sampleCount: 1,
  depth: false,
  depthFormat: 'depth24plus',
  stencil: false,
  formats: ['rgba8unorm'],
  scaleToCanvas: false,
  sizeMultiplier: 1.0,
  useMips: false,
  depthOrArrayLength: 1,
  colorTarget: null,
  discardMSAA: false,
  discardDepth: false,
  createTexture: false,
};

export default class RenderTexture extends Texture {
  //the textures rendered into, multisampled with msaa
  public attachments: GPUTexture[] = [];
  //mip level 0 of every attachment, the views rendered into. to sample an attachment use getView
  public views: GPUTextureView[] = [];
  public depthStencilAttachment: GPUTexture | null = null;
  private _depthSlice: number = 0;
  private _depthStencilView: GPUTextureView | null = null;
  private _renderPassDescriptor!: GPURenderPassDescriptor;
  //with msaa: by attachment index, the texture it is resolved into. it holds the mips. on screen attachment 0 is
  //resolved into the canvas texture instead, so it has none
  private resolveTextures: GPUTexture[] = [];
  //mip level 0 of every resolve texture, the resolve targets
  private resolveViews: GPUTextureView[] = [];
  //by attachment index, every mip of it as sampled in a shader: of the resolve texture with msaa
  private sampledViews: GPUTextureView[] = [];
  //with a colorTarget there is no texture of its own to take the size from
  private targetWidth: number = 0;
  private targetHeight: number = 0;
  //set by clear(), for the descriptors of the higher mip levels
  private colorLoadOp: GPULoadOp = "clear";
  private clearColor: GPUColor = [0, 0, 0, 0];
  //the views of every attachment at a mip level above 0, by attachmentIndex * 32 + mipLevel
  private attachmentMipViews: Map<number, GPUTextureView> = new Map();
  //by mip level above 0, the descriptors of the passes into it. the load op, clear color and slice are set on every use
  private mipRenderPassDescriptors: GPURenderPassDescriptor[] = [];
  //on screen: the canvas texture this.view was created from, it changes every frame
  private canvasTexture: GPUTexture | null = null;

  constructor(renderer: Renderer, width: number = 1, height: number = 1, options: Partial<RenderTextureOptions> = {}) {
    super(renderer, width, height, options.depthOrArrayLength, { ...defaultRenderTextureOptions, ...options });

    if (options.scaleToCanvas) {
      width = renderer.canvas.width;
      height = renderer.canvas.height;
    }
    this.setSize(width, height);
  }

  protected get options(): RenderTextureOptions {
    return <RenderTextureOptions>super.options;
  }

  public get formats(): GPUTextureFormat[] {
    return this.options.formats;
  }

  public get sampleCount(): number {
    return this.options.sampleCount;
  }

  //the passes into the higher mip levels render straight into the resolve textures, without msaa
  public getSampleCount(mipLevel: number): number {
    return mipLevel > 0 ? 1 : this.sampleCount;
  }

  public get resizeToCanvas(): boolean {
    return this.options.scaleToCanvas;
  }

  public getRenderPassDescriptor(mipLevel: number = 0): GPURenderPassDescriptor {
    if (mipLevel == 0) {
      if (this.options.toScreen) {
        this.updateCanvasView();
        const colorAttachment = (<GPURenderPassColorAttachment[]>this._renderPassDescriptor.colorAttachments)[0];
        if (this.options.sampleCount > 1) {
          colorAttachment.resolveTarget = this.view;
        } else {
          colorAttachment.view = this.view;
        }
      }
      return this._renderPassDescriptor;
    } else {
      return this.getRenderPassDescriptorForMipLevel(mipLevel);
    }
  }

  //a view is created only once per canvas texture, which stays the same during a frame
  private updateCanvasView() {
    const texture = this.renderer.context.getCurrentTexture();
    if (texture !== this.canvasTexture) {
      this.canvasTexture = texture;
      this.view = texture.createView();
    }
  }

  public get sizeMultiplier() {
    return this.options.sizeMultiplier;
  }

  public set sizeMultiplier(s: number) {
    this.options.sizeMultiplier = s;
  }

  //override of Texture class
  public get gpuTexture(): GPUTexture {
    if (this.rendersToCanvasTexture) return this.renderer.context.getCurrentTexture();
    return this.getGPUTexture(0);
  }

  //the texture that holds the result of an attachment, the resolve texture with msaa
  public getGPUTexture(attachmentIndex: number): GPUTexture {
    return this.resolveTextures[attachmentIndex] ?? this.attachments[attachmentIndex];
  }

  //an attachment as sampled in a shader: every mip, resolved with msaa. views[i] is multisampled with msaa
  public getView(attachmentIndex: number): GPUTextureView {
    return attachmentIndex == 0 ? this.view : this.sampledViews[attachmentIndex];
  }

  //the screen without msaa renders straight into the canvas texture, so it has no color attachment of its own
  private get rendersToCanvasTexture(): boolean {
    return this.options.toScreen && this.sampleCount == 1;
  }

  public get width(): number {
    return this.options.colorTarget || this.rendersToCanvasTexture ? this.targetWidth : super.width;
  }

  public get height(): number {
    return this.options.colorTarget || this.rendersToCanvasTexture ? this.targetHeight : super.height;
  }

  /** Returns the multisampled texture view when sampleCount > 1, null otherwise */
  public get msaaTextureView(): GPUTextureView | null {
    return this.sampleCount > 1 ? this.views[0] : null;
  }

  public get depthStencilView(): GPUTextureView | null {
    return this._depthStencilView;
  }

  public clear(colorClear: boolean, clearColor: Color, depthClear: boolean, depthClearValue: number, stencilClear: boolean = depthClear, stencilClearValue: number = 0) {
    //also used by the passes into the higher mip levels, which get their own descriptor
    this.colorLoadOp = colorClear ? "clear" : "load";
    this.clearColor = clearColor;

    if (this._renderPassDescriptor) {
      for (const colorAttachment of this._renderPassDescriptor.colorAttachments) {
        if (colorAttachment) {
          colorAttachment.loadOp = this.colorLoadOp;
          colorAttachment.clearValue = clearColor;
        }
      }
      const depthStencilAttachment = this._renderPassDescriptor.depthStencilAttachment;
      if (depthStencilAttachment && this.options.depth) {
        depthStencilAttachment.depthLoadOp = depthClear ? "clear" : "load";
        depthStencilAttachment.depthClearValue = depthClearValue;
      }
      if (depthStencilAttachment && this.options.stencil) {
        depthStencilAttachment.stencilLoadOp = stencilClear ? "clear" : "load";
        depthStencilAttachment.stencilClearValue = stencilClearValue;
      }
    }
  }

  public setSize(width: number, height: number) {
    width = Math.ceil(width * this.options.sizeMultiplier);
    height = Math.ceil(height * this.options.sizeMultiplier);

    if (this.width != width || this.height != height) {
      this.createAttachments(width, height, this.options.depthOrArrayLength);
      this.createRenderPassDescriptor();
      LogGL.log('RenderTexture: setSize', width, height);
    }
  }

  public setSlice(slice: number) {
    if (this.depth > 1) {
      if (slice < this.depth) {
        this._depthSlice = slice;
        //in place, the volume loops set a slice for every pass
        for (const colorAttachment of this._renderPassDescriptor.colorAttachments) {
          if (colorAttachment) colorAttachment.depthSlice = slice;
        }
      } else {
        LogGL.warn('RenderTexture: setSlice out of bounds', slice, this.depth);
      }
    }
  }

  //switches to another view of the same size and format, like another layer of the same mip
  public setColorTarget(view: GPUTextureView) {
    this.options.colorTarget = view;
    this.views[0] = view;
    this.view = view;
    this.mipRenderPassDescriptors = [];
    //in place, a cube is rendered by switching between its sides
    const colorAttachment = (<GPURenderPassColorAttachment[] | undefined>this._renderPassDescriptor?.colorAttachments)?.[0];
    if (colorAttachment) {
      colorAttachment.view = view;
    } else {
      this.createRenderPassDescriptor();
    }
  }

  //after the submit: draws earlier in this frame can still use the textures
  public destruct() {
    for (const attachment of this.attachments) {
      this.renderer.destroyAfterSubmit(attachment);
    }
    for (const resolveTexture of this.resolveTextures) {
      if (resolveTexture) this.renderer.destroyAfterSubmit(resolveTexture);
    }
    if (this.depthStencilAttachment) this.renderer.destroyAfterSubmit(this.depthStencilAttachment);
  }

  private createAttachments(width: number, height: number, depth: number) {
    //draws earlier in this frame can still use the old textures
    for (const attachment of this.attachments) {
      this.renderer.destroyAfterSubmit(attachment);
    }
    for (const resolveTexture of this.resolveTextures) {
      if (resolveTexture) this.renderer.destroyAfterSubmit(resolveTexture);
    }
    if (this.depthStencilAttachment) this.renderer.destroyAfterSubmit(this.depthStencilAttachment);
    this.depthStencilAttachment = null;
    this._depthStencilView = null;

    this.attachments = [];
    this.views = [];
    this.resolveTextures = [];
    this.resolveViews = [];
    this.sampledViews = [];
    this.clearMipViews();
    this.attachmentMipViews = new Map();
    this.mipRenderPassDescriptors = [];
    const colorTarget = this.options.colorTarget;
    const ownsAttachments = !colorTarget && !this.rendersToCanvasTexture;
    if (!ownsAttachments) {
      this.targetWidth = width;
      this.targetHeight = height;
    }
    if (colorTarget) {
      this.views.push(colorTarget);
      this.view = colorTarget;
    }
    if (this.rendersToCanvasTexture) {
      //replaced by the canvas texture of the frame in getRenderPassDescriptor
      this.updateCanvasView();
      this.views.push(this.view);
    }
    const multisampled = this.sampleCount > 1;
    const mipLevelCount = this.getMipCount(width, height);
    //copied from by getData
    const usage = this.options.usage | GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC;
    const formats = ownsAttachments ? this.options.formats : [];
    for (let i = 0; i < formats.length; i++) {
      const texture = this.renderer.device.createTexture({
        size: [width, height, depth],
        usage: usage,
        dimension: depth == 1 ? '2d' : '3d',
        format: formats[i],
        sampleCount: this.sampleCount,
        //a multisampled texture has a single mip, the mips are in the texture it is resolved into
        mipLevelCount: multisampled ? 1 : mipLevelCount,
      });
      this.attachments.push(texture);
      this.views.push(texture.createView({
        //needs to be one for color attachments otherwise webgpu gives error
        mipLevelCount: 1,
      }));

      if (multisampled && !(this.options.toScreen && i == 0)) {
        const resolveTexture = this.renderer.device.createTexture({
          size: [width, height],
          format: formats[i],
          usage: usage,
          sampleCount: 1,
          mipLevelCount: mipLevelCount,
        });
        this.resolveTextures[i] = resolveTexture;
        this.resolveViews[i] = resolveTexture.createView({
          mipLevelCount: 1,
        });
      }
      const sampledTexture = this.getGPUTexture(i);
      this.sampledViews[i] = sampledTexture.createView({
        mipLevelCount: sampledTexture.mipLevelCount,
      });
    }
    if (this.attachments.length > 0) {
      this.view = this.sampledViews[0];
    }

    if (this.options.depth || this.options.stencil) {
      const format: GPUTextureFormat = !this.options.stencil ? this.options.depthFormat : !this.options.depth ? 'stencil8' : 'depth24plus-stencil8';
      this.depthStencilAttachment = this.renderer.device.createTexture({
        size: [width, height],
        format,
        //TODO: GPUTextureUsage.TEXTURE_BINDING is only needed when the depth texture needs to be sampled in a shader
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | (format == 'depth32float' ? GPUTextureUsage.COPY_SRC : 0),
        sampleCount: this.sampleCount,
      });
      this._depthStencilView = this.depthStencilAttachment.createView({
        mipLevelCount: 1,
      });
    }
  }

  private getRenderPassDescriptorForMipLevel(mipLevel: number): GPURenderPassDescriptor {
    let renderPassDescriptor = this.mipRenderPassDescriptors[mipLevel];
    if (!renderPassDescriptor) {
      const colorAttachments: GPURenderPassColorAttachment[] = [];
      //every attachment, the pipeline has a target for each format
      const attachmentCount = Math.max(1, this.attachments.length);
      for (let i = 0; i < attachmentCount; i++) {
        colorAttachments.push({
          loadOp: this.colorLoadOp,
          storeOp: 'store',
          view: i == 0 ? this.getMipView(mipLevel) : this.getAttachmentMipView(i, mipLevel),
        });
      }
      //no depth attachment: the dimensions would not match
      renderPassDescriptor = { colorAttachments: colorAttachments };
      this.mipRenderPassDescriptors[mipLevel] = renderPassDescriptor;
    }

    //a 3d mip level is rendered one slice at a time, like mip level 0. it has fewer slices
    const depthSlice = this.options.depthOrArrayLength > 1 ? Math.min(this._depthSlice, Math.max(1, this.options.depthOrArrayLength >> mipLevel) - 1) : undefined;
    for (const colorAttachment of renderPassDescriptor.colorAttachments) {
      if (!colorAttachment) continue;
      colorAttachment.loadOp = this.colorLoadOp;
      colorAttachment.clearValue = this.clearColor;
      if (depthSlice !== undefined) colorAttachment.depthSlice = depthSlice;
    }
    return renderPassDescriptor;
  }

  private getAttachmentMipView(attachmentIndex: number, mipLevel: number): GPUTextureView {
    const id = attachmentIndex * 32 + mipLevel;
    let view = this.attachmentMipViews.get(id);
    if (!view) {
      const texture = this.getGPUTexture(attachmentIndex);
      view = texture.createView(Texture.getMipViewDescriptor(texture, mipLevel));
      this.attachmentMipViews.set(id, view);
    }
    return view;
  }


  private createRenderPassDescriptor() {
    const colorAttachments: GPURenderPassColorAttachment[] = [];

    for (let i = 0; i < this.views.length; i++) {
      const colorAttachment: GPURenderPassColorAttachment = {
        loadOp: "clear",
        storeOp: 'store',
        view: this.views[i],
      };
      if (this.options.depthOrArrayLength > 1) {
        colorAttachment.depthSlice = this._depthSlice;
      }
      //on screen attachment 0 is resolved into the canvas texture, see getRenderPassDescriptor
      if (this.resolveViews[i]) {
        colorAttachment.resolveTarget = this.resolveViews[i];
      }
      if (this.sampleCount > 1 && this.options.discardMSAA) {
        colorAttachment.storeOp = 'discard';
      }

      colorAttachments.push(colorAttachment);
    }

    this._renderPassDescriptor = {
      colorAttachments: colorAttachments,
    };

    if (this._depthStencilView) {
      this._renderPassDescriptor.depthStencilAttachment = {
        view: this._depthStencilView
      }
      if (this.options.depth) {
        this._renderPassDescriptor.depthStencilAttachment.depthLoadOp = "clear";
        this._renderPassDescriptor.depthStencilAttachment.depthStoreOp = this.options.discardDepth ? "discard" : "store";
      }
      if (this.options.stencil) {
        this._renderPassDescriptor.depthStencilAttachment.stencilLoadOp = "clear";
        this._renderPassDescriptor.depthStencilAttachment.stencilStoreOp = "store";
      }
    }
  }

  public generateMipmaps(material: VertexFragmentShader | null = null) {
    this.renderer.createMipsForRenderTexture(this, material);
  }

  public get stencilEnabled(): boolean {
    return this.options.stencil;
  }

  public get depthEnabled(): boolean {
    return this.options.depth;
  }
}
