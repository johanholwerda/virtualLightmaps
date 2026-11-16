import { BindGroupLayoutEntryInfo } from '../ShaderBase';
import LogGL from '../../core/LogGL';
import WGSLUtils from './WGSLUtils';

export default class BindGroupUtils {

  //type and storageType come without whitespace, e.g. "texture_storage_2d<rgba8unorm,write>" and "storage,read"
  public static createEntry(binding: number, visibility: number, type: string, storageType: string): GPUBindGroupLayoutEntry {
    const entry: GPUBindGroupLayoutEntry = {
      binding,
      visibility,
    };

    if (type.startsWith('sampler')) {
      entry.sampler = {type: type === 'sampler_comparison' ? 'comparison' : 'filtering'};
    } else if (type === 'texture_external') {
      //has to be checked before the generic texture_ branch below
      entry.externalTexture = {};
    } else if (type.startsWith('texture_storage')) {
      // Parses e.g. "texture_storage_2d_array<rgba8unorm,write>" into dim="2d_array", format="rgba8unorm", access="write"
      const match = type.match(/^texture_storage_(\w+)<(\w+),(\w+)>$/);
      if (!match) {
        LogGL.warn('BindGroupUtils: can not parse storage texture type', type);
        return entry;
      }
      const [, dim, format, access] = match;
      entry.storageTexture = {
        access: access === 'read_write' ? 'read-write' : access === 'read' ? 'read-only' : 'write-only',
        format: <GPUTextureFormat>format,
        viewDimension: this.getViewDimension(dim),
      };
      if (entry.storageTexture.access !== 'read-only') {
        //not supported in vertex shader
        entry.visibility = visibility & ~GPUShaderStage.VERTEX;
      }
    } else if (type.startsWith('texture_depth')) {
      // e.g. "texture_depth_2d", "texture_depth_cube_array", "texture_depth_multisampled_2d"
      entry.texture = {
        sampleType: 'depth',
        viewDimension: this.getViewDimension(type.slice('texture_depth_'.length)),
        multisampled: type.includes('multisampled'),
      };
    } else if (type.startsWith('texture_')) {
      // Parses e.g. "texture_2d<f32>" into dim="2d", format="f32"
      // Also handles "texture_cube<f32>", "texture_2d_array<f32>", "texture_multisampled_2d<f32>"
      const match = type.match(/^texture_(\w+)<(\w+)>$/);
      if (!match) {
        LogGL.warn('BindGroupUtils: can not parse texture type', type);
        return entry;
      }
      const [, dim, format] = match;
      const multisampled = type.includes('multisampled');
      entry.texture = {
        //a multisampled float texture can't be filtered, so 'float' is not allowed for it
        sampleType: format === 'f32' ? (multisampled ? 'unfilterable-float' : 'float') : format === 'i32' ? 'sint' : 'uint',
        viewDimension: this.getViewDimension(dim),
        multisampled,
      };
    } else if (storageType.includes('uniform')) {
      //before the array check below: var<uniform> can hold an array too
      entry.buffer = {
        type: 'uniform',
        hasDynamicOffset: type.includes('dynamic'),
        minBindingSize: 0,
      };
    } else if (type.startsWith('array') || storageType.includes('storage')) {
      //the access mode of <storage> defaults to read
      const isReadOnly = !storageType.includes('write');
      entry.buffer = {
        type: isReadOnly ? 'read-only-storage' : 'storage',
        hasDynamicOffset: type.includes('dynamic'),
        minBindingSize: 0,
      };
      if (!isReadOnly) {
        // Remove vertex visibility for storage buffers that aren't read-only
        entry.visibility = visibility & ~GPUShaderStage.VERTEX;
      }
    } else {
      LogGL.warn('BindGroupUtils: unknown binding type', type, storageType);
    }

    return entry;
  }

  //dim is the part of the type name that describes the dimension, e.g. "2d", "2d_array", "cube_array" or "multisampled_2d"
  private static getViewDimension(dim: string = '2d'): GPUTextureViewDimension {
    const isArray = dim.endsWith('_array');
    const base = dim.replace('_array', '').replace('multisampled_', '');
    let viewDimension: GPUTextureViewDimension;
    switch (base) {
      case '1d': viewDimension = '1d'; break;
      case '3d': viewDimension = '3d'; break;
      case 'cube': viewDimension = 'cube'; break;
      default: viewDimension = '2d'; break; // Default fallback
    }
    return isArray ? `${viewDimension}-array` as GPUTextureViewDimension : viewDimension;
  }

  //the functions that sample (or gather) one of their texture parameters, directly or by passing it on to
  //another such function. a texture that is passed to one of them is sampled as well
  private static getSamplingFunctions(code: string): Set<string> {
    const functions = WGSLUtils.getFunctions(code);
    const samplingFunctions = new Set<string>();
    let changed = true;
    while (changed) {
      changed = false;
      for (const fn of functions) {
        if (samplingFunctions.has(fn.name)) continue;
        for (const param of WGSLUtils.splitList(fn.params)) {
          const [, name, type] = param.text.match(/([A-Za-z_]\w*)\s*:\s*(\S+)/) || [];
          if (type?.startsWith('texture') && this.isTextureSampled(fn.body, name, samplingFunctions)) {
            samplingFunctions.add(fn.name);
            changed = true;
            break;
          }
        }
      }
    }
    return samplingFunctions;
  }

  private static isTextureSampled(code: string, name: string, samplingFunctions: Set<string>): boolean {
    // the texture is the first argument, or the second after the component of textureGather(component, t, s, coords).
    // the second argument of the other sampling functions is a sampler, so it can't be the texture
    let pattern = `\\b(?:textureSample|textureGather)\\w*\\s*\\(\\s*(?:[^,()]*,\\s*)?${name}\\b`;
    if (samplingFunctions.size > 0) {
      // or name is one of the arguments of a call to one of the sampling functions
      pattern += `|\\b(?:${Array.from(samplingFunctions).join('|')})\\s*\\([^;{}]*\\b${name}\\b`;
    }
    return new RegExp(pattern).test(code);
  }

  public static getBindGroupLayoutEntries(shaderCode: string): BindGroupLayoutEntryInfo[] {
    const entries: BindGroupLayoutEntryInfo[] = [];

    shaderCode = WGSLUtils.maskComments(shaderCode);
    const constants = WGSLUtils.getIntegerConstants(shaderCode);
    const aliases = WGSLUtils.getAliases(shaderCode);
    const samplingFunctions = this.getSamplingFunctions(shaderCode);

    // Determine shader stages by looking at the function declarations
    const visibility = (/@vertex/.test(shaderCode) ? GPUShaderStage.VERTEX : 0) |
      (/@fragment/.test(shaderCode) ? GPUShaderStage.FRAGMENT : 0) |
      (/@compute/.test(shaderCode) ? GPUShaderStage.COMPUTE : 0);

    // Parses a WGSL var declaration with its attributes, which can be on other lines than the var and can share
    // a line with other declarations. Capture groups:
    //   1: attributes  — e.g. "@group(0) @binding(1) "
    //   2: storageType — optional address space and access mode, e.g. "uniform", "storage, read"
    //   3: name        — the variable name
    //   4: type        — the full type up to the semicolon, e.g. "texture_2d<f32>", "array<vec4<f32>>"
    // Examples:
    //   "@group(0) @binding(0) var<uniform> uniforms : Uniforms;"   → ("uniform", "uniforms", "Uniforms")
    //   "@group(0) @binding(1)\nvar<storage, read> data: array<T>;" → ("storage, read", "data", "array<T>")
    //   "@group(0) @binding(2) var textureSampler: sampler;"         → (undefined, "textureSampler", "sampler")
    const declarationRegex = /((?:@\w+(?:\s*\([^()]*\))?\s*)+)\bvar\b\s*(?:<([^>]*)>)?\s*([A-Za-z_]\w*)\s*:\s*([^;]+);/g;

    let match;
    while ((match = declarationRegex.exec(shaderCode)) !== null) {
      const [, attributes, storageTypeWithWhitespace, name, typeWithWhitespace] = match;

      const group = WGSLUtils.getAttributeValue(attributes, 'group', constants);
      const binding = WGSLUtils.getAttributeValue(attributes, 'binding', constants);
      if (group === undefined || binding === undefined) continue;

      const storageType = WGSLUtils.removeWhitespace(storageTypeWithWhitespace ?? '');
      const type = WGSLUtils.resolveAlias(WGSLUtils.removeWhitespace(typeWithWhitespace), aliases);

      const entry = this.createEntry(binding, visibility, type, storageType);

      // A float texture that is not sampled (directly or in a function it is passed to) is marked unfilterable-float.
      // every float format can be bound to that, also the ones that can't be filtered, like r32float
      if (entry.texture && entry.texture.sampleType === 'float' && !this.isTextureSampled(shaderCode, name, samplingFunctions)) {
        entry.texture.sampleType = 'unfilterable-float';
      }

      entries.push(new BindGroupLayoutEntryInfo(entry, name, group));
    }

    return entries;
  }

}
