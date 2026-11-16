import Renderer from '../render/Renderer';
import Matrix4x4 from '../math/Matrix4x4';
import Vector4 from '../math/Vector4';
import Vector3 from '../math/Vector3';
import Vector2 from '../math/Vector2';
import LogGL from '../core/LogGL';
import Texture from '../texture/Texture';
import UniformUtils from './shaderUtils/UniformUtils';
import BindGroupUtils from './shaderUtils/BindGroupUtils';
import DefineUtils from './shaderUtils/DefineUtils';
import WGSLUtils from './shaderUtils/WGSLUtils';
import Quaternion from '../math/Quaternion';
import {PipelineConstants} from './shaderUtils/OverrideUtils';

//the component type of a uniform: i32 and u32 uniforms are written as integers into uniformData
export type UniformType = 'f32' | 'i32' | 'u32';

export interface IUniform {
  name: string;
  startIndex: number;
  stride: number;
  type: UniformType;
}

export type ShaderBaseOptions = {
  includes?: string[];
  defines?: Record<string, string | boolean | number | undefined>;
}

export class BindGroupLayoutEntryInfo {
  public bindGroupLayoutEntry: GPUBindGroupLayoutEntry;
  public name: string;
  public groupIndex: number;

  public constructor(bindGroupLayoutEntry: GPUBindGroupLayoutEntry, name: string, groupIndex: number) {
    this.bindGroupLayoutEntry = bindGroupLayoutEntry;
    this.name = name;
    this.groupIndex = groupIndex;
  }
}

class BindGroup {
  public dirty: boolean = true;
  public entries: BindGroupLayoutEntryInfo[] = [];
  public id: number = -1;
  public bindGroup!: GPUBindGroup;
  public bindGroupLayout: GPUBindGroupLayout;
  public index: number;
  public currentBindData: (GPUBuffer | GPUSampler | GPUTextureView | GPUTexture | GPUExternalTexture)[] = [];
  public bindGroupEntries: GPUBindGroupEntry[] = [];
  //reused for every setBindGroup call: the encoder copies the values
  public dynamicOffsets: number[] = [0];
  public indexOfUniformEntry: number = -1;
  //the cache entry bindGroup came from
  public cached: CachedBindGroup | null = null;
  //the textures bound with setTexture. their view is looked up before every use, so a texture that is created
  //again (resized, loaded) stays bound
  public textureRefs: { binding: number, texture: Texture }[] = [];

  constructor(entries: BindGroupLayoutEntryInfo[], bindGroupLayout: GPUBindGroupLayout, index:number) {
    this.entries = entries;
    this.bindGroupLayout = bindGroupLayout;
    this.index = index;
  }
}

//the bind groups currently set on a pass encoder, so setBindGroups can skip the ones that are already set
export type BindGroupState = {
  bindGroups: (GPUBindGroup | undefined)[];
  offsets: number[];
}

//numbers for the bound resources and layouts, to hash the entries of a bind group. weak, so they can be collected
const resourceIds: WeakMap<object, number> = new WeakMap();
let nextResourceId = 1;

function getResourceId(resource: object): number {
  let id = resourceIds.get(resource);
  if (id === undefined) {
    id = nextResourceId++;
    resourceIds.set(resource, id);
  }
  return id;
}

class CachedBindGroup {
  public entries: GPUBindGroupEntry[] = [];
  public bindGroup: GPUBindGroup;
  public id: number = -1;
  public layout: GPUBindGroupLayout;
  //the frame it was last bound in, see updateBindGroup
  public lastUse: number = 0;

  constructor(renderer: Renderer, entries: GPUBindGroupEntry[], bindGroup: GPUBindGroup, layout: GPUBindGroupLayout) {
    this.entries = entries;
    this.bindGroup = bindGroup;
    this.layout = layout;
    this.id = renderer.instanceID++;
    this.lastUse = renderer.time.frameNumber;
  }
}

export default class ShaderBase {
  public renderer: Renderer;
  public name: string;
  public shaderCode: string;

  //bind groups
  private bindGroups: BindGroup[] = [];
  //by name, the first one when a name is used in more than one group
  private bindingsByName: Map<string, BindGroupLayoutEntryInfo> = new Map();
  //by the hash of their entries (see hashEntries), the groups of the shader share it
  private bindGroupCache: Map<number, CachedBindGroup[]> = new Map();
  private lastCacheEviction: number = -1;
  //this enables a comparison in getBindGroupByEntries. weak, so the buffers that are no longer bound can be collected
  private GPUBufferBindingByBuffer: WeakMap<GPUBuffer, GPUBufferBinding> = new WeakMap();
  private _pipelineLayout: GPUPipelineLayout | null = null;
  //the overrides, passed to every stage. created by the subclass, see setConstant
  protected constants!: PipelineConstants;

  //uniforms
  protected uniforms: IUniform[] = [];
  //the same, by name for the setters
  private uniformsByName: Map<string, IUniform> = new Map();
  private lengthMismatchWarned: Set<string> = new Set();
  public uniformData!: Float32Array;
  //views on the buffer of uniformData, to write the i32 and u32 uniforms
  private uniformDataInt!: Int32Array;
  private uniformDataUint!: Uint32Array;
  //managed by Renderer.setUniformData: where the uniforms were last copied to in the renderer's uniform buffer.
  //that copy can be reused until the uniforms change or the renderer uploads the buffer and starts over (a new epoch)
  public uniformsChanged: boolean = true;
  public uniformOffset: number = 0;
  public uniformEpoch: number = -1;
  public id: number = -1;
  public defines: Map<string, string | boolean | number | undefined> = new Map();

  public constructor(renderer: Renderer, name: string, shaderCode: string, options: ShaderBaseOptions = {}) {
    this.renderer = renderer;
    this.name = name;
    //what the device supports, like HAS_SUBGROUPS (see Renderer.shaderDefines). the defines of the shader win
    if (renderer.shaderDefines) {
      for (const [key, value] of renderer.shaderDefines) {
        this.defines.set(key, value);
      }
    }
    if (options.defines) {
      for (const [key, value] of Object.entries(options.defines)) {
        this.defines.set(key, value);
      }
    }
    this.id = renderer.instanceID++;

    //the includes go first, in order, on their own lines (so a directive at the end of one doesn't end up on a line with
    //the code after it), and everything is preprocessed in one go, so the #defines of an include apply to the shader.
    //the #defines of the shader come after the includes, so they are collected first to apply to the includes as well
    const includes = options.includes ?? [];
    let defines = this.defines;
    if (includes.length > 0) {
      defines = new Map();
      DefineUtils.preprocess(shaderCode, this.defines, defines);
    }
    //the directives (enable, requires, diagnostic) of the shader come after the includes, so they are moved to the top
    this.shaderCode = WGSLUtils.hoistDirectives(DefineUtils.preprocess([...includes, shaderCode].join('\n'), defines));

    this.initUniforms();
  }

  protected initBindGroups(){
    const bindGroupLayoutEntries = BindGroupUtils.getBindGroupLayoutEntries(this.shaderCode);
    const entriesByGroup: { [id: number]: GPUBindGroupLayoutEntry[] } = {};
    const entryInfosByGroup: { [id: number]: BindGroupLayoutEntryInfo[] } = {};
    const groupIndices: number[] = [];

    for (const bindGroupLayoutEntry of bindGroupLayoutEntries) {
      if(bindGroupLayoutEntry.name == 'uniforms'){
        (<GPUBufferBindingLayout>bindGroupLayoutEntry.bindGroupLayoutEntry.buffer).hasDynamicOffset = true;
      }

      if(groupIndices.indexOf(bindGroupLayoutEntry.groupIndex) == -1){
        groupIndices.push(bindGroupLayoutEntry.groupIndex);
        entriesByGroup[bindGroupLayoutEntry.groupIndex] = [];
        entryInfosByGroup[bindGroupLayoutEntry.groupIndex] = [];
      }
      entriesByGroup[bindGroupLayoutEntry.groupIndex].push(bindGroupLayoutEntry.bindGroupLayoutEntry);
      entryInfosByGroup[bindGroupLayoutEntry.groupIndex].push(bindGroupLayoutEntry);
    }

    //a group index the shader skips gets an empty bind group, so the pipeline layout has no holes
    const groupCount = groupIndices.length > 0 ? Math.max(...groupIndices) + 1 : 0;
    for (let groupIndex = 0; groupIndex < groupCount; groupIndex++) {
      if (groupIndices.indexOf(groupIndex) == -1) {
        groupIndices.push(groupIndex);
        entriesByGroup[groupIndex] = [];
        entryInfosByGroup[groupIndex] = [];
      }
    }

    for(const groupIndex of groupIndices){
      const bindGroupLayoutDescriptor: GPUBindGroupLayoutDescriptor = {
        label: this.name + '_' + groupIndex,
        entries: entriesByGroup[groupIndex],
      };
      const bindGroupLayout = this.renderer.device.createBindGroupLayout(bindGroupLayoutDescriptor);
      this.bindGroups[groupIndex] = new BindGroup(entryInfosByGroup[groupIndex], bindGroupLayout, groupIndex);
    }

    for (const bindGroup of this.bindGroups) {
      for (const entryInfo of bindGroup.entries) {
        if (!this.bindingsByName.has(entryInfo.name)) this.bindingsByName.set(entryInfo.name, entryInfo);
      }
    }
  }

  //the override constants: set by a subclass, with whether the ones without a default start at 0 (see PipelineConstants)
  public setConstant(name: string, value: boolean | number) {
    if (this.constants.has(name)) {
      if (this.constants.set(name, value)) {
        this.constantsChanged();
      }
    } else {
      console.warn('constant not found', this.name, name, this.constants.names, this.shaderCode);
    }
  }

  //undefined when the constant doesn't exist, or was not set and its default is an expression
  public getConstant(name: string): boolean | number | undefined {
    if (!this.constants.has(name)) {
      console.warn('constant not found', this.name, name, this.constants.names);
    }
    return this.constants.get(name);
  }

  //the pipeline has to be made again
  protected constantsChanged(): void {
  }

  public get bindGroupLayouts(): GPUBindGroupLayout[] {
    const layouts = [];
    for (const bindGroup of this.bindGroups) {
      layouts.push(bindGroup.bindGroupLayout);
    }
    return layouts;
  }

  //the bind group layouts are fixed once initBindGroups ran, so every pipeline of the shader shares one layout
  protected get pipelineLayout(): GPUPipelineLayout {
    if (!this._pipelineLayout) {
      this._pipelineLayout = this.renderer.device.createPipelineLayout({
        label: this.name + '_pipeline_layout',
        bindGroupLayouts: this.bindGroupLayouts,
      });
    }
    return this._pipelineLayout;
  }

  //the ids of the current bind groups, written into out, which is returned
  public getBindGroupIds(out: number[]): number[] {
    out.length = this.bindGroups.length;
    for (let i = 0; i < this.bindGroups.length; i++) {
      const bindGroup = this.bindGroups[i];
      this.updateBindGroup(bindGroup);
      out[i] = bindGroup.id;
    }
    return out;
  }

  //pass a state to skip the bind groups that are already set on the encoder
  public setBindGroups(passEncoder: GPUComputePassEncoder | GPURenderPassEncoder | GPURenderBundleEncoder, uniformBufferOffset:number, state?: BindGroupState) {
    for (const bindGroup of this.bindGroups) {
      this.updateBindGroup(bindGroup);

      const hasUniforms = bindGroup.indexOfUniformEntry >= 0;
      if (state && state.bindGroups[bindGroup.index] === bindGroup.bindGroup && (!hasUniforms || state.offsets[bindGroup.index] === uniformBufferOffset)) {
        continue;
      }

      if (hasUniforms) {
        bindGroup.dynamicOffsets[0] = uniformBufferOffset;
        passEncoder.setBindGroup(bindGroup.index, bindGroup.bindGroup, bindGroup.dynamicOffsets);
      } else {
        passEncoder.setBindGroup(bindGroup.index, bindGroup.bindGroup);
      }

      if (state) {
        state.bindGroups[bindGroup.index] = bindGroup.bindGroup;
        state.offsets[bindGroup.index] = uniformBufferOffset;
      }
    }
  }

  private bindAtIndex(bindingData: GPUBuffer | GPUSampler | GPUTextureView | GPUTexture | GPUExternalTexture, bindingIndex: number, groupIndex:number) {
    const bindGroupInfo = this.bindGroups[groupIndex];
    if(!bindGroupInfo) {
      LogGL.warn('bindGroup not found at group index', this.name, groupIndex, this.bindGroups);
      return;
    }

    if(bindGroupInfo.currentBindData[bindingIndex] == bindingData) return;

    if (bindingData instanceof GPUBuffer) {
      bindGroupInfo.bindGroupEntries[bindingIndex] = {
        binding: bindingIndex,
        resource: this.getBufferBindingByBuffer(bindingData),
      };
    } else {
      bindGroupInfo.bindGroupEntries[bindingIndex] = {
        binding:  bindingIndex,
        resource: bindingData,
      };
    }
    bindGroupInfo.currentBindData[bindingIndex] = bindingData;
    bindGroupInfo.dirty = true;
  }

  private getBindingByName(name: string): BindGroupLayoutEntryInfo | undefined {
    return this.bindingsByName.get(name);
  }

  public setBinding(name: string, bindingData: GPUBuffer | GPUSampler | GPUTextureView | GPUTexture | GPUExternalTexture) {
    const bindingInfo = this.getBindingByName(name);
    if (bindingInfo) {
      //replaces a texture bound with setTexture
      this.setTextureRef(bindingInfo, null);
      this.bindAtIndex(bindingData, bindingInfo.bindGroupLayoutEntry.binding, bindingInfo.groupIndex);
    }else{
      LogGL.warn('binding not found', this.name, name, bindingData, this.bindGroups);
    }
  }

  //remembers the texture bound at the binding, or forgets it with null
  private setTextureRef(bindingInfo: BindGroupLayoutEntryInfo, texture: Texture | null) {
    const bindGroup = this.bindGroups[bindingInfo.groupIndex];
    if (!bindGroup) return;
    const binding = bindingInfo.bindGroupLayoutEntry.binding;
    const refs = bindGroup.textureRefs;
    for (let i = 0; i < refs.length; i++) {
      if (refs[i].binding === binding) {
        if (texture) {
          refs[i].texture = texture;
        } else {
          refs.splice(i, 1);
        }
        return;
      }
    }
    if (texture) refs.push({binding, texture});
  }

  //this allows for lazier coding: it sets the uniform buffers, default samplers and textures when they are not yet set
  protected bindUnboundBindings(bindGroup: BindGroup) {
    for (const entryInfo of bindGroup.entries) {
      const entry = entryInfo.bindGroupLayoutEntry;

      if (entryInfo.name == 'uniforms') {
          this.bindAtIndex(this.renderer.uniformBuffer, entry.binding, bindGroup.index);
          (<GPUBufferBinding>bindGroup.bindGroupEntries[entry.binding].resource).size = this.uniformData.byteLength;
          bindGroup.indexOfUniformEntry = entry.binding;
      } else {
        if (bindGroup.currentBindData[entry.binding]) continue;

        if(!entry.sampler) {
          LogGL.warn(this.name, bindGroup.index, 'dummy data bound: ', entry);
        }
        this.bindAtIndex(this.renderer.dummyBindings.getDummyBindData(entry, this.name), entry.binding, bindGroup.index);
      }
    }
  }

  public setSampler(name: string, value: GPUSampler) {
    this.setBinding(name, value);
  }

  public setBuffer(name: string, value: GPUBuffer) {
    this.setBinding(name, value);
  }

  //a Texture is bound by reference: when it creates its GPU texture again (setSize, a loader that finished, a video
  //that changed size) the next draw binds the new view. A view or a GPUTexture is bound as it is
  public setTexture(name: string, value: Texture | GPUTextureView | GPUTexture) {
    if (!(value instanceof Texture)) {
      this.setBinding(name, value);
      return;
    }
    const bindingInfo = this.getBindingByName(name);
    if (!bindingInfo) {
      LogGL.warn('binding not found', this.name, name, value, this.bindGroups);
      return;
    }
    this.setTextureRef(bindingInfo, value);
    //a texture without a view yet (like a RenderTexture of size 0) gets it bound once it has one
    if (value.view) this.bindAtIndex(value.view, bindingInfo.bindGroupLayoutEntry.binding, bindingInfo.groupIndex);
  }

  //an external texture (texture_external in wgsl) expires at the end of the task it was imported in,
  //so this has to be called again every frame. see ExternalVideoTexture
  public setExternalTexture(name: string, value: GPUExternalTexture) {
    this.setBinding(name, value);
  }

  private getBufferBindingByBuffer(buffer: GPUBuffer): GPUBufferBinding {
    if (this.GPUBufferBindingByBuffer.has(buffer)) {
      return this.GPUBufferBindingByBuffer.get(buffer)!;
    } else {
      const bufferBinding = {buffer};
      this.GPUBufferBindingByBuffer.set(buffer, bufferBinding);
      return bufferBinding;
    }
  }

  private createBindGroup(bindGroup: BindGroup, entries: GPUBindGroupEntry[]) {
    return this.renderer.device.createBindGroup({
      label:   this.name + '_' + bindGroup.index,
      layout:  bindGroup.bindGroupLayout,
      entries: entries,
    })
  }

  private updateBindGroup(bindGroup: BindGroup) {
    //the renderer replaces its uniform buffer when it has to grow
    if (bindGroup.indexOfUniformEntry >= 0 && bindGroup.currentBindData[bindGroup.indexOfUniformEntry] !== this.renderer.uniformBuffer) {
      bindGroup.dirty = true;
    }

    //a texture bound with setTexture can have created a new view since
    const refs = bindGroup.textureRefs;
    for (let i = 0; i < refs.length; i++) {
      const view = refs[i].texture.view;
      if (view && view !== bindGroup.currentBindData[refs[i].binding]) {
        this.bindAtIndex(view, refs[i].binding, bindGroup.index);
      }
    }

    const frameNumber = this.renderer.time.frameNumber;
    if (bindGroup.dirty) {
      this.bindUnboundBindings(bindGroup);

      const hash = this.hashEntries(bindGroup);
      let cachedBindGroup = this.getBindGroupByEntries(bindGroup, hash);
      if (!cachedBindGroup) {
        //the entries are stored at their binding index: filter() leaves out the holes when the bindings are not
        //numbered contiguously (and makes the copy the cache keeps)
        const entries = bindGroup.bindGroupEntries.filter(entry => entry);
        cachedBindGroup = new CachedBindGroup(this.renderer, entries, this.createBindGroup(bindGroup, entries), bindGroup.bindGroupLayout);

        if (this.lastCacheEviction !== frameNumber) {
          this.lastCacheEviction = frameNumber;
          this.evictBindGroups(frameNumber - 10);
        }
        let bucket = this.bindGroupCache.get(hash);
        if (!bucket) {
          bucket = [];
          this.bindGroupCache.set(hash, bucket);
        }
        bucket.push(cachedBindGroup);
      }
      bindGroup.cached = cachedBindGroup;
      bindGroup.id = cachedBindGroup.id;
      bindGroup.bindGroup = cachedBindGroup.bindGroup;
      bindGroup.dirty = false;
    }
    //also when it stays bound without changes
    if (bindGroup.cached) bindGroup.cached.lastUse = frameNumber;
  }

  //removes the bind groups that weren't bound since frameThreshold. by last use, so the ones that are switched between
  //(like ping-pong textures) stay. in place, a removed bucket is the only garbage
  private evictBindGroups(frameThreshold: number) {
    for (const [hash, bucket] of this.bindGroupCache) {
      let kept = 0;
      for (const cachedBindGroup of bucket) {
        if (cachedBindGroup.lastUse > frameThreshold) bucket[kept++] = cachedBindGroup;
      }
      bucket.length = kept;
      if (kept === 0) this.bindGroupCache.delete(hash);
    }
  }

  //of the layout and the entries, so equal entries hash the same
  private hashEntries(bindGroup: BindGroup): number {
    let hash = getResourceId(bindGroup.bindGroupLayout);
    const entries = bindGroup.bindGroupEntries;
    for (let i = 0; i < entries.length; i++) {
      const entry = entries[i];
      if (!entry) continue;
      hash = (Math.imul(hash, 31) + entry.binding) | 0;
      hash = (Math.imul(hash, 31) + getResourceId(entry.resource)) | 0;
    }
    return hash;
  }

  //compares the entries as they are stored, at their binding index with holes, so a cache hit allocates nothing.
  //the layout too: the groups of a shader share the cache
  private getBindGroupByEntries(bindGroup: BindGroup, hash: number): CachedBindGroup | null {
    const bucket = this.bindGroupCache.get(hash);
    if (!bucket) return null;
    const entries = bindGroup.bindGroupEntries;
    for (const cachedBindGroup of bucket) {
      if (cachedBindGroup.layout !== bindGroup.bindGroupLayout) continue;
      const cachedEntries = cachedBindGroup.entries;
      let cachedIndex = 0;
      let match = true;
      for (let i = 0; i < entries.length; i++) {
        const entry = entries[i];
        if (!entry) continue;
        const cachedEntry = cachedEntries[cachedIndex++];
        if (!cachedEntry || cachedEntry.binding !== entry.binding || cachedEntry.resource !== entry.resource) {
          match = false;
          break;
        }
      }
      if (match && cachedIndex === cachedEntries.length) {
        return cachedBindGroup;
      }
    }
    return null;
  }

  //lets go of the bind groups and the resources bound to them, so destroyed textures and buffers can be collected
  public destruct(): void {
    this.bindGroupCache.clear();
    for (const bindGroup of this.bindGroups) {
      bindGroup.textureRefs = [];
      bindGroup.currentBindData = [];
      bindGroup.bindGroupEntries = [];
      bindGroup.cached = null;
      bindGroup.dirty = true;
    }
  }

  //--------------------------------------------------------------------------------------------------------------------
  //uniforms

  private initUniforms() {
    const uniforms = UniformUtils.getUniformsFromShader(this.shaderCode);
    if (uniforms.length > 0) {
      this.uniforms = this.orderUniforms(uniforms);
      for (const uniform of this.uniforms) {
        if (!this.uniformsByName.has(uniform.name)) this.uniformsByName.set(uniform.name, uniform);
      }
      const orderedStructString = UniformUtils.getUniformStructString(this.uniforms);
      this.shaderCode = UniformUtils.replaceUniformsWithOrderedUniforms(this.shaderCode, orderedStructString);
    }
  }

  public setBool(name: string, value: boolean) {
    this.setUniformComponents(name, 1, value ? 1 : 0);
  }

  public setFloat(name: string, value: number) {
    this.setUniformComponents(name, 1, value);
  }

  public setVector2(name: string, value: Vector2) {
    this.setUniformComponents(name, 2, value.x, value.y);
  }

  public setVector3(name: string, value: Vector3) {
    this.setUniformComponents(name, 3, value.x, value.y, value.z);
  }

  public setVector4(name: string, value: Vector4 | Quaternion) {
    this.setUniformComponents(name, 4, value.x, value.y, value.z, value.w);
  }

  //setUniform without an array, the setters above are called for every draw. Like setUniform it converts the values to
  //the type of the uniform, and drops the components the uniform doesn't have
  private setUniformComponents(name: string, count: number, x: number, y: number = 0, z: number = 0, w: number = 0) {
    const uniform = this.getUniform(name);
    if (!uniform) return;
    if (uniform.stride != count) {
      this.warnLengthMismatch(uniform, count);
      count = Math.min(count, uniform.stride);
    }
    const data = this.getUniformDataView(uniform);
    const index = uniform.startIndex;
    data[index] = x;
    if (count > 1) data[index + 1] = y;
    if (count > 2) data[index + 2] = z;
    if (count > 3) data[index + 3] = w;
    this.uniformsChanged = true;
  }

  public setMatrix(name: string, value: Matrix4x4) {
    this.setUniform(name, value.m);
  }

  //the values are converted to the type of the uniform, so setFloat works for an i32 or u32 uniform as well.
  //values past the length of the uniform are dropped, so they don't overwrite the uniform after it
  public setUniform(name: string, value: number[] | Float32Array, warning:boolean = true) {
    const uniform = this.getUniform(name, warning);
    if(uniform){
      if(uniform.stride != value.length){
        this.warnLengthMismatch(uniform, value.length);
        if (value.length > uniform.stride) value = value.slice(0, uniform.stride);
      }
      //set() converts number[] to the type of the view
      this.getUniformDataView(uniform).set(value, uniform.startIndex);
      this.uniformsChanged = true;
    }
  }

  //once per uniform, the setters are called for every draw
  private warnLengthMismatch(uniform: IUniform, length: number) {
    if (this.lengthMismatchWarned.has(uniform.name)) return;
    this.lengthMismatchWarned.add(uniform.name);
    console.warn('setUniform: value length does not match uniform length', this.name, uniform.name, uniform.stride, length);
  }

  public getUniformValue(name: string): number[] {
    const uniform = this.getUniform(name);
    if (uniform) {
      const data = [];
      const view = this.getUniformDataView(uniform);
      for (let i = 0; i < uniform.stride; i++) {
        data.push(view[uniform.startIndex + i]);
      }
      return data;
    }
    return [];
  }

  private getUniformDataView(uniform: IUniform): Float32Array | Int32Array | Uint32Array {
    return uniform.type == 'i32' ? this.uniformDataInt : uniform.type == 'u32' ? this.uniformDataUint : this.uniformData;
  }

  //without a warning when it doesn't
  public hasUniform(name: string): boolean {
    return this.uniformsByName.has(name);
  }

  private getUniform(name: string, warning:boolean = true): IUniform | null {
    const uniform = this.uniformsByName.get(name);
    if (uniform) return uniform;
    if(warning)LogGL.warn('uniform not found on shader', this.name, name, this.uniforms);
    return null;
  }

  private orderUniforms(uniforms: IUniform[]): IUniform[] {
    const paddedAndSortedUniforms: IUniform[] = [];
    let paddingIndex = 0;

    uniforms.sort((a, b) => {
      return a.stride - b.stride;
    });

    let componentCount = 0;
    do {
      const uniform: IUniform | undefined = uniforms.pop();
      if (uniform) {
        paddedAndSortedUniforms.push(uniform);
        uniform.startIndex = componentCount;
        componentCount += uniform.stride;
      }

      //fills the rest of the 16 bytes with the biggest uniforms that fit, and padding when none do, so the next vector
      //(or matrix) starts aligned and the size ends up a multiple of 4 components
      let slotsLeft = (4 - componentCount % 4) % 4;
      for (let i = uniforms.length - 1; i >= 0 && slotsLeft > 0; i--) {
        const uniform = uniforms[i];
        if (uniform.stride <= slotsLeft) {
          slotsLeft -= uniform.stride;
          paddedAndSortedUniforms.push(uniform);
          uniform.startIndex = componentCount;
          componentCount += uniform.stride;
          uniforms.splice(i, 1);
        }
      }
      for (let i = 0; i < slotsLeft; i++) {
        paddedAndSortedUniforms.push({name: `padding${paddingIndex++}`, startIndex: componentCount, stride: 1, type: 'f32'});
        componentCount++;
      }
    } while (uniforms.length > 0);

    this.uniformData = new Float32Array(componentCount);
    this.uniformDataInt = new Int32Array(this.uniformData.buffer);
    this.uniformDataUint = new Uint32Array(this.uniformData.buffer);
    return paddedAndSortedUniforms;
  }

}
