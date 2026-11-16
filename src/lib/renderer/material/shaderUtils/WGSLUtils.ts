import LogGL from '../../core/LogGL';

export type WGSLFunction = {
  name: string;
  //the parameter list, without the parentheses, and its index in the code
  params: string;
  paramsStart: number;
  //the body, including the braces
  body: string;
}

export type WGSLListItem = {
  text: string;
  //index of text in the string that was split
  start: number;
}

export type WGSLScalarType = 'f32' | 'i32' | 'u32';

//the suffix of a vector type like vec3f
const SHORTHAND_SCALAR_TYPES: Record<string, WGSLScalarType> = {f: 'f32', i: 'i32', u: 'u32'};

//helpers shared by the shaderUtils parsers. they expect code that went through maskComments
export default class WGSLUtils {
  private static lastCode: string = '';
  private static lastMasked: string = '';

  //replaces the comments with spaces. newlines are kept, so every index in the result is the same as in the code:
  //a match in the masked code can be looked up (or replaced) in the original.
  //block comments nest in WGSL: all of "/* a /* b */ c */" is a comment
  public static maskComments(code: string): string {
    //the parsers of one shader all start with the same code
    if (code === this.lastCode) return this.lastMasked;

    const chars = code.split('');
    let i = 0;
    while (i < code.length) {
      if (code[i] === '/' && code[i + 1] === '/') {
        while (i < code.length && code[i] !== '\n' && code[i] !== '\r') chars[i++] = ' ';
      } else if (code[i] === '/' && code[i + 1] === '*') {
        let depth = 0;
        while (i < code.length) {
          if (code[i] === '/' && code[i + 1] === '*') {
            depth++;
            chars[i] = chars[i + 1] = ' ';
            i += 2;
          } else if (code[i] === '*' && code[i + 1] === '/') {
            depth--;
            chars[i] = chars[i + 1] = ' ';
            i += 2;
            if (depth === 0) break;
          } else {
            if (code[i] !== '\n' && code[i] !== '\r') chars[i] = ' ';
            i++;
          }
        }
      } else {
        i++;
      }
    }

    this.lastCode = code;
    this.lastMasked = chars.join('');
    return this.lastMasked;
  }

  //"enable subgroups;", "requires subgroup_id;" and "diagnostic(off, derivative_uniformity);" must come before every
  //declaration. The includes go in front of the shader, so a directive of the shader (or of a later include) ends up
  //after the code of the includes, which doesn't compile. This moves the directives to the start of the first line:
  //on that line, so the line numbers of compile errors stay the same. A directive that is there more than once (two
  //includes that both enable f16) is kept once. Directives in comments are not moved
  public static hoistDirectives(code: string): string {
    const masked = this.maskComments(code);
    const directives: string[] = [];
    const seen = new Set<string>();
    const chars = code.split('');
    // one or more directives at the start of a line: "enable f16, subgroups;" or "diagnostic(off, x.y);"
    const directive = /^\s*(?:(?:enable|requires)\s+[A-Za-z_][\w\s,]*;|diagnostic\s*\([^()]*\)\s*;)/;

    let depth = 0;
    let lineStart = 0;
    while (lineStart < masked.length) {
      let lineEnd = masked.indexOf('\n', lineStart);
      if (lineEnd < 0) lineEnd = masked.length;

      //only outside of braces: a directive is never inside a function or a struct
      if (depth === 0) {
        let index = lineStart;
        let match;
        while ((match = directive.exec(masked.slice(index, lineEnd))) !== null) {
          const text = match[0].trim();
          const key = this.removeWhitespace(text);
          if (!seen.has(key)) {
            seen.add(key);
            directives.push(text);
          }
          for (let i = index; i < index + match[0].length; i++) {
            if (chars[i] !== '\n' && chars[i] !== '\r') chars[i] = ' ';
          }
          index += match[0].length;
        }
      }

      for (let i = lineStart; i < lineEnd; i++) {
        if (masked[i] === '{') depth++;
        else if (masked[i] === '}') depth--;
      }
      lineStart = lineEnd + 1;
    }

    if (directives.length === 0) return code;
    return directives.join(' ') + ' ' + chars.join('');
  }

  public static removeWhitespace(text: string): string {
    return text.replace(/\s+/g, '');
  }

  //"f32", "vec3<u32>", "vec4f", ... (without whitespace) → the scalar type and the number of components.
  //undefined for other types
  public static parseScalarOrVector(type: string): { scalar: WGSLScalarType, components: number } | undefined {
    const scalar = type.match(/^(f32|i32|u32)$/);
    if (scalar) return {scalar: scalar[1] as WGSLScalarType, components: 1};

    // "vec3<f32>" or "vec3f"
    const vector = type.match(/^vec([234])(?:<(f32|i32|u32)>|([fiu]))$/);
    if (vector) return {scalar: (vector[2] ?? SHORTHAND_SCALAR_TYPES[vector[3]]) as WGSLScalarType, components: parseInt(vector[1])};

    return undefined;
  }

  //the index of the bracket that closes the one at openIndex, or -1
  public static findClosing(code: string, openIndex: number): number {
    const open = code[openIndex];
    const close = open === '(' ? ')' : open === '[' ? ']' : '}';
    let depth = 0;
    for (let i = openIndex; i < code.length; i++) {
      if (code[i] === open) {
        depth++;
      } else if (code[i] === close) {
        depth--;
        if (depth === 0) return i;
      }
    }
    return -1;
  }

  //splits on the commas that are not inside (), <> or [], e.g. struct members or function parameters:
  //"a : array<f32, 4>, @location(0) b : vec3f" → ["a : array<f32, 4>", " @location(0) b : vec3f"]
  public static splitList(text: string): WGSLListItem[] {
    const items: WGSLListItem[] = [];
    let depth = 0;
    let start = 0;
    for (let i = 0; i <= text.length; i++) {
      const char = text[i];
      if (char === '(' || char === '<' || char === '[') depth++;
      else if (char === ')' || char === '>' || char === ']') depth--;
      else if ((char === ',' && depth === 0) || i === text.length) {
        if (text.slice(start, i).trim()) items.push({text: text.slice(start, i), start});
        start = i + 1;
      }
    }
    return items;
  }

  public static getFunctions(code: string): WGSLFunction[] {
    const functions: WGSLFunction[] = [];
    // "fn name(" — the parameters and the body are found by matching the brackets, so attributes like
    // @location(0) in the parameters or the return type don't end them early
    const regex = /\bfn\s+([A-Za-z_]\w*)\s*\(/g;
    let match;
    while ((match = regex.exec(code)) !== null) {
      const paramsOpen = match.index + match[0].length - 1;
      const paramsClose = this.findClosing(code, paramsOpen);
      if (paramsClose < 0) continue;
      const bodyOpen = code.indexOf('{', paramsClose);
      const bodyClose = bodyOpen < 0 ? -1 : this.findClosing(code, bodyOpen);
      if (bodyClose < 0) continue;
      functions.push({
        name: match[1],
        params: code.slice(paramsOpen + 1, paramsClose),
        paramsStart: paramsOpen + 1,
        body: code.slice(bodyOpen, bodyClose + 1),
      });
      regex.lastIndex = bodyClose;
    }
    return functions;
  }

  public static findFunction(code: string, name: string): WGSLFunction | undefined {
    return this.getFunctions(code).find(fn => fn.name === name);
  }

  //"struct Name { ... }": the index of "struct", of the opening brace and of the closing brace, or undefined
  public static findStruct(code: string, name: string): { start: number, open: number, close: number } | undefined {
    const match = new RegExp(`\\bstruct\\s+${name}\\s*\\{`).exec(code);
    if (!match) return undefined;
    const open = match.index + match[0].length - 1;
    const close = this.findClosing(code, open);
    return close < 0 ? undefined : {start: match.index, open, close};
  }

  //"alias Name = type;" → Name → type, without whitespace
  public static getAliases(code: string): Map<string, string> {
    const aliases = new Map<string, string>();
    const regex = /\balias\s+([A-Za-z_]\w*)\s*=\s*([^;]+);/g;
    let match;
    while ((match = regex.exec(code)) !== null) {
      aliases.set(match[1], this.removeWhitespace(match[2]));
    }
    return aliases;
  }

  public static resolveAlias(type: string, aliases: Map<string, string>): string {
    //an alias can refer to another alias
    for (let i = 0; i < 8 && aliases.has(type); i++) {
      type = aliases.get(type)!;
    }
    return type;
  }

  //"const NAME = 3;" (with or without a type) → NAME → 3. only integer literals and other integer consts,
  //which is what is used for @group(...) and @binding(...) in practice
  public static getIntegerConstants(code: string): Map<string, number> {
    const constants = new Map<string, number>();
    const regex = /\bconst\s+([A-Za-z_]\w*)\s*(?::\s*\w+\s*)?=\s*([^;]+);/g;
    let match;
    while ((match = regex.exec(code)) !== null) {
      const value = this.parseInteger(match[2], constants);
      if (value !== undefined) constants.set(match[1], value);
    }
    return constants;
  }

  //an integer literal (decimal or hex, with an optional i/u suffix) or the name of an integer const
  public static parseInteger(expression: string, constants?: Map<string, number>): number | undefined {
    expression = expression.trim();
    const literal = expression.match(/^(0[xX][0-9a-fA-F]+|\d+)[iu]?$/);
    if (literal) return parseInt(literal[1]);
    return constants?.get(expression);
  }

  //the value of an attribute like @group(1), @binding(2u), @id(ID) or @location(0,). undefined when the attributes
  //don't have it, or (with a warning) when its value is not an integer literal or integer const
  public static getAttributeValue(attributes: string, name: string, constants: Map<string, number>): number | undefined {
    const match = attributes.match(new RegExp(`@${name}\\s*\\(([^()]*)\\)`));
    if (!match) return undefined;
    const expression = match[1].replace(/,\s*$/, '');
    const value = this.parseInteger(expression, constants);
    if (value === undefined) LogGL.warn('WGSLUtils: can not evaluate', `@${name}(${expression})`, '(supported: integer literals and integer consts)');
    return value;
  }
}
