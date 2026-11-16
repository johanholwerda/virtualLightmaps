import LogGL from '../../core/LogGL';
import WGSLUtils from './WGSLUtils';

export interface VertexInput {
  name: string;
  //the @location(n) of the input
  location: number;
  stride: number;
  format: GPUVertexFormat;
}

export default class VertexInputUtils {

  //the inputs of vs_main: parameters with @location(n), and the @location(n) members of the structs it takes.
  //without a vs_main, the members of struct VertexInput
  public static getVertexInputs(shaderCode: string): VertexInput[] {
    const vertexInputs: VertexInput[] = [];

    // Comments are masked (replaced by spaces) for the parsing. Every index in it is the same as in shaderCode,
    // so the original text of an input can be looked up to find annotations like /* unorm */ next to its type
    const maskedCode = WGSLUtils.maskComments(shaderCode);
    const constants = WGSLUtils.getIntegerConstants(maskedCode);

    //[start, end) ranges in the code that contain @location(n) declarations, separated by commas
    const lists: { start: number, end: number }[] = [];
    const addStruct = (name: string) => {
      const struct = WGSLUtils.findStruct(maskedCode, name);
      if (struct) lists.push({start: struct.open + 1, end: struct.close});
    };

    const vsMain = WGSLUtils.findFunction(maskedCode, 'vs_main');
    if (vsMain) {
      // e.g. "fn vs_main(@location(0) pos: vec3f, input: VertexInput, @builtin(vertex_index) index: u32)"
      lists.push({start: vsMain.paramsStart, end: vsMain.paramsStart + vsMain.params.length});
      for (const param of WGSLUtils.splitList(vsMain.params)) {
        const [, type] = param.text.match(/^[^@]*?[A-Za-z_]\w*\s*:\s*([A-Za-z_]\w*)\s*$/) || [];
        if (type) addStruct(type);
      }
    } else {
      addStruct('VertexInput');
    }

    for (const list of lists) {
      for (const item of WGSLUtils.splitList(maskedCode.slice(list.start, list.end))) {
        const location = WGSLUtils.getAttributeValue(item.text, 'location', constants);
        if (location === undefined) continue;

        // Parses the name and type after the attributes. Examples:
        //   "@location(0) position : vec3<f32>"          → ("position", "vec3<f32>")
        //   "@location( 1 ) @interpolate(flat) id: u32"  → ("id", "u32")
        // Capture groups: 1=attribute name, 2=type
        const match = item.text.match(/([A-Za-z_]\w*)\s*:\s*([^]+)$/);
        if (!match) continue;

        // The type as written in the original code, so including comments like "vec4<f32> /* unorm */"
        const typeStart = list.start + item.start + item.text.length - match[2].length;
        const typeWithComments = shaderCode.slice(typeStart, list.start + item.start + item.text.length);
        const {format, components} = this.wgslTypeToVertexFormat(typeWithComments);

        vertexInputs.push({
          name: match[1],
          location,
          stride: components,
          format,
        });
      }
    }

    return vertexInputs;
  }

  //a WGSL type → the vertex format for it, and its number of components. A comment after the type picks a smaller
  //format for a vec2 or vec4:
  //  f32: "vec4<f32> /* unorm */" → unorm8x4, "vec2f /* snorm */" → snorm8x2, "vec2f /* unorm16 */" → unorm16x2,
  //       "vec4f /* snorm16 */" → snorm16x4, "vec2f /* float16 */" → float16x2
  //  u32: "vec4u /* uint8 */" → uint8x4, "vec2u /* uint16 */" → uint16x2
  //  i32: "vec4i /* sint8 */" → sint8x4, "vec2i /* sint16 */" → sint16x2
  private static wgslTypeToVertexFormat(wgslType: string): { format: GPUVertexFormat, components: number } {
    // Capture groups: 1=kind (unorm/snorm/float/uint/sint), 2=bits (8/16), optional for unorm and snorm (8)
    const annotation = wgslType.match(/\/\*\s*(unorm|snorm|float|uint|sint)(8|16)?\s*\*\//);

    // Strip comments (e.g. /* unorm */) and whitespace to get the raw WGSL type for matching
    const cleanType = WGSLUtils.removeWhitespace(wgslType.replace(/\/\*[\s\S]*?\*\/|\/\/.*/g, ''));

    // e.g. "vec3<f32>" or "vec3f" → components=3, dataType="f32"
    const parsed = WGSLUtils.parseScalarOrVector(cleanType);
    if (!parsed) {
      LogGL.warn('VertexInputUtils: unsupported vertex input type, using float32:', wgslType);
      return {format: 'float32', components: 1};
    }
    const {scalar: dataType, components} = parsed;

    if (annotation) {
      const kind = annotation[1];
      const bits = annotation[2] ?? (kind === 'unorm' || kind === 'snorm' ? '8' : '');
      // the normalized and float16 formats are read as f32, uint as u32 and sint as i32. the formats smaller than
      // 32 bits only exist with 2 or 4 components, and float only with 16 bits
      const readAs = kind === 'uint' ? 'u32' : kind === 'sint' ? 'i32' : 'f32';
      if (bits && !(kind === 'float' && bits !== '16') && dataType === readAs && (components === 2 || components === 4)) {
        return {format: `${kind}${bits}x${components}` as GPUVertexFormat, components};
      }
      LogGL.warn('VertexInputUtils: unsupported annotation, ignored for', wgslType, '(unorm, snorm, unorm16, snorm16 and float16 need a vec2 or vec4 of f32, uint8 and uint16 of u32, sint8 and sint16 of i32)');
    }

    const format = dataType === 'f32' ? 'float32' : dataType === 'u32' ? 'uint32' : 'sint32';
    return {format: (components === 1 ? format : `${format}x${components}`) as GPUVertexFormat, components};
  }

}
