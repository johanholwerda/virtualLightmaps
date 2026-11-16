import LogGL from '../../core/LogGL';
import WGSLUtils from './WGSLUtils';

export default class DefineUtils {

  //outDefines: receives the external defines plus the #defines of the code that were active
  public static preprocess(shaderCode: string, externalDefines: Map<string, string | boolean | number | undefined>, outDefines?: Map<string, string | boolean | number | undefined>): string {
    // Merge external defines into a working copy
    const defines = outDefines ?? new Map<string, string | boolean | number | undefined>();
    for (const [name, value] of externalDefines) defines.set(name, value);

    const lines = shaderCode.split('\n');
    // The directives are read from the lines without comments: "#endif // FEATURE" is an #endif, the value of
    // "#define X 0.5 // note" is 0.5, and a directive inside a block comment is not one.
    // maskComments keeps the newlines, so maskedLines[i] is lines[i] with its comments replaced by spaces
    const maskedLines = WGSLUtils.maskComments(shaderCode).split('\n');
    const output: string[] = [];

    // Stack tracks nested #ifdef/#ifndef blocks. Each entry: active: whether the condition of this block (or of its
    // #else) is true, parentActive: whether the code around the block is active, elseAllowed: whether #else is still valid
    const stack: { active: boolean; parentActive: boolean; elseAllowed: boolean }[] = [];

    //a define passed as false or undefined is not defined, so {FEATURE: condition} works for #ifdef FEATURE
    const isDefined = (name: string): boolean => {
      const value = defines.get(name);
      return value !== false && value !== undefined;
    };

    const isActive = (): boolean => {
      if (stack.length === 0) return true;
      const top = stack[stack.length - 1];
      return top.active && top.parentActive;
    };

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];
      const trimmed = maskedLines[i].trim();
      //a directive or an inactive line stays as an empty line, so the line numbers of compile errors stay the same
      output[i] = '';

      // #define NAME or #define NAME value
      const defineMatch = trimmed.match(/^#define\s+(\w+)(?:\s+(.+))?$/);
      if (defineMatch) {
        if (isActive()) {
          const name = defineMatch[1];
          const value = defineMatch[2]?.trim();
          defines.set(name, value !== undefined ? value : true);
        }
        continue;
      }

      // #ifdef NAME or #ifndef NAME
      const ifdefMatch = trimmed.match(/^#if(n?)def\s+(\w+)$/);
      if (ifdefMatch) {
        const defined = isDefined(ifdefMatch[2]);
        stack.push({ active: ifdefMatch[1] ? !defined : defined, parentActive: isActive(), elseAllowed: true });
        continue;
      }

      // #else
      if (trimmed === '#else') {
        if (stack.length === 0) {
          LogGL.warn('DefineUtils: #else without matching #ifdef/#ifndef');
          continue;
        }
        const top = stack[stack.length - 1];
        if (!top.elseAllowed) {
          LogGL.warn('DefineUtils: duplicate #else');
          continue;
        }
        top.active = !top.active;
        top.elseAllowed = false;
        continue;
      }

      // #endif
      if (trimmed === '#endif') {
        if (stack.length === 0) {
          LogGL.warn('DefineUtils: #endif without matching #ifdef/#ifndef');
          continue;
        }
        stack.pop();
        continue;
      }

      // WGSL doesn't use #, so this is a directive that is not supported (like #if) or written wrong (like
      // "#ifdef A B"). It is passed on like a regular line, so the shader doesn't compile instead of taking a wrong branch
      if (trimmed.startsWith('#')) {
        LogGL.warn('DefineUtils: unsupported directive:', trimmed, '(supported: #define NAME [value], #ifdef NAME, #ifndef NAME, #else, #endif)');
      }

      // Regular line — only emit if all conditions in the stack are active
      if (isActive()) {
        output[i] = line;
      }
    }

    if (stack.length > 0) {
      LogGL.warn(`DefineUtils: ${stack.length} unclosed #ifdef/#ifndef block(s)`);
    }

    return this.substituteValues(output.join('\n'), defines);
  }

  // Value substitution: replaces the names of the defines that have a string or number value (not just boolean flags).
  // A value is expanded as well, so "#define A (B * 2)" also uses the value of B, whichever of the two was defined
  // first. A define is not expanded inside its own value, so "#define A A + 1" doesn't recurse.
  private static substituteValues(code: string, defines: Map<string, string | boolean | number | undefined>): string {
    const values = new Map<string, string>();
    for (const [name, value] of defines) {
      if (typeof value === 'string' || typeof value === 'number') values.set(name, String(value));
    }
    if (values.size === 0) return code;

    // Whole-word occurrences only (not inside other identifiers), all names in one pass
    const regex = new RegExp(`\\b(?:${Array.from(values.keys()).join('|')})\\b`, 'g');
    const expanding = new Set<string>();
    const expand = (text: string): string => text.replace(regex, name => {
      if (expanding.has(name)) return name;
      expanding.add(name);
      const expanded = expand(values.get(name)!);
      expanding.delete(name);
      return expanded;
    });
    return expand(code);
  }
}
