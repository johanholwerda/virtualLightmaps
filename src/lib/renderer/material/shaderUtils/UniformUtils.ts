import { IUniform, UniformType } from '../ShaderBase';
import LogGL from '../../core/LogGL';
import WGSLUtils from './WGSLUtils';

export default class UniformUtils {

  //replaces the struct Uniforms { ... } of the shader with the reordered version. the struct is looked up in the code
  //without comments, so a struct Uniforms in a comment (or a } in a comment inside the struct) is not mistaken for it
  public static replaceUniformsWithOrderedUniforms(shaderCode: string, orderedUniforms: string): string {
    const struct = WGSLUtils.findStruct(WGSLUtils.maskComments(shaderCode), 'Uniforms');
    if (!struct) return shaderCode;

    return shaderCode.slice(0, struct.start) + orderedUniforms + shaderCode.slice(struct.close + 1);
  }

  public static getUniformStructString(uniforms: IUniform[]): string {
    const members = uniforms.map(uniform => `  ${uniform.name} : ${this.getTypeFromStride(uniform.stride, uniform.type)},\n`);
    return `struct Uniforms {\n${members.join('')}}\n`;
  }

  public static getTypeFromStride(stride: number, type: UniformType = 'f32'): string {
    switch (stride) {
      case 16:
        return `mat4x4<${type}>`;
      case 4:
        return `vec4<${type}>`;
      case 3:
        return `vec3<${type}>`;
      case 2:
        return `vec2<${type}>`;
      default:
        return type;
    }
  }

  //"vec4<f32>", "vec4f", "u32", "mat4x4f", ... (without whitespace) → the component type and count
  private static parseType(type: string): { type: UniformType, stride: number } | undefined {
    const scalarOrVector = WGSLUtils.parseScalarOrVector(type);
    if (scalarOrVector) return {type: scalarOrVector.scalar, stride: scalarOrVector.components};

    // "mat4x4<f32>" or "mat4x4f". other matrices pad their columns, which orderUniforms doesn't do
    if (/^mat4x4(?:<f32>|f)$/.test(type)) return {type: 'f32', stride: 16};

    return undefined;
  }

  public static getUniformsFromShader(shaderCode: string): IUniform[] {
    shaderCode = WGSLUtils.maskComments(shaderCode);
    const struct = WGSLUtils.findStruct(shaderCode, 'Uniforms');

    if (!struct) {
      return [];
    }

    const result: IUniform[] = [];
    // e.g. "viewProjection : mat4x4<f32>,\n\t@align(16) time : f32," → ["viewProjection : mat4x4<f32>", "\n\t@align(16) time : f32"]
    for (const member of WGSLUtils.splitList(shaderCode.slice(struct.open + 1, struct.close))) {
      // Parses a single uniform field into name and type, skipping attributes like @align(16) before it.
      //   "viewProjection : mat4x4<f32>" → ("viewProjection", "mat4x4<f32>")
      //   "@size(16) time : f32"         → ("time", "f32")
      const matched = member.text.match(/([A-Za-z_]\w*)\s*:\s*([^]+)$/);
      if (!matched) {
        continue;
      }

      const name = matched[1];
      const typeString = WGSLUtils.removeWhitespace(matched[2]);
      let parsed = this.parseType(typeString);
      if (!parsed) {
        LogGL.warn('UniformUtils: unsupported uniform type, it is replaced by f32:', name, typeString, '(supported: f32, i32, u32, their vectors and mat4x4<f32>)');
        parsed = {type: 'f32', stride: 1};
      }

      //startIndex is set by ShaderBase.orderUniforms
      result.push({name, type: parsed.type, stride: parsed.stride, startIndex: 0});
    }

    return result;
  }
}
