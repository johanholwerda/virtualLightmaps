import WGSLUtils from './WGSLUtils';

export type OverrideType = 'bool' | 'f32' | 'u32' | 'i32' | 'f16';

export type OverrideInfo = {
  name: string;
  //what the pipeline knows the override by: the @id when it has one ("7" for @id(7)), the name otherwise
  key: string;
  //the declared type, or the type of the literal it is initialized with. undefined for an inferred type
  //that depends on an expression
  type: OverrideType | undefined;
  //the default, when it is a literal. without a default the zero value of the type (what PipelineConstants passes with
  //defaultToZero). undefined when the default is an expression
  value: boolean | number | undefined;
  //whether there is a default, so the pipeline doesn't need a value for it
  hasDefault: boolean;
}

export default class OverrideUtils {

  public static getOverrides(shaderCode: string): OverrideInfo[] {
    const overrides: OverrideInfo[] = [];
    shaderCode = WGSLUtils.maskComments(shaderCode);
    const constants = WGSLUtils.getIntegerConstants(shaderCode);

    // Matches WGSL override declarations. The type is optional when there is an initializer, which can be any expression.
    //   "override USE_ALBEDO_MAP: bool = false;"  → (undefined, "USE_ALBEDO_MAP", "bool", "false")
    //   "override WORKGROUP_SIZE_X: u32;"         → (undefined, "WORKGROUP_SIZE_X", "u32", undefined)
    //   "override SCALE = BASE * 2.0;"            → (undefined, "SCALE", undefined, "BASE * 2.0")
    //   "@id(7) override COUNT: u32 = 1u;"        → ("@id(7) ", "COUNT", "u32", "1u")
    // Capture groups: 1=attributes, 2=name, 3=type (optional), 4=initializer (optional)
    const regex = /((?:@\w+(?:\s*\([^()]*\))?\s*)*)\boverride\s+([A-Za-z_]\w*)\s*(?::\s*(\w+)\s*)?(?:=\s*([^;]+))?;/g;

    let match;
    while ((match = regex.exec(shaderCode)) !== null) {
      const [, attributes, name, declaredType, initializer] = match;

      const id = WGSLUtils.getAttributeValue(attributes, 'id', constants);

      const literal = initializer !== undefined ? this.parseLiteral(initializer.trim()) : undefined;
      const type = <OverrideType | undefined>declaredType ?? literal?.type;

      // no default: the zero value of the type, see PipelineConstants
      const value = initializer === undefined ? (type === 'bool' ? false : 0) : literal?.value;

      overrides.push({
        name,
        key: id !== undefined ? String(id) : name,
        type,
        value,
        hasDefault: initializer !== undefined,
      });
    }

    return overrides;
  }

  //"true", "1.5", "1e+3", "2u", "0x2Au", "-3i", ... → the value and the type it implies. undefined for an expression
  private static parseLiteral(literal: string): { value: boolean | number, type: OverrideType } | undefined {
    if (literal === 'true' || literal === 'false') {
      return {value: literal === 'true', type: 'bool'};
    }

    const hex = literal.match(/^(-?)0[xX]([0-9a-fA-F]+)([iu]?)$/);
    if (hex) {
      return {value: parseInt(hex[1] + hex[2], 16), type: hex[3] === 'u' ? 'u32' : 'i32'};
    }

    const decimal = literal.match(/^-?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?([iufh]?)$/);
    if (decimal) {
      const suffix = decimal[1];
      const isFloat = suffix === 'f' || suffix === 'h' || /[.eE]/.test(literal);
      const type: OverrideType = suffix === 'u' ? 'u32' : suffix === 'h' ? 'f16' : isFloat ? 'f32' : 'i32';
      return {value: parseFloat(literal), type};
    }

    return undefined;
  }
}

//the values of the overrides of one shader module. one set is passed to every stage: the pipeline accepts
//overrides that a stage doesn't use, so there is no need to know which stage uses which
export class PipelineConstants {
  private overridesByName: Map<string, OverrideInfo> = new Map();
  //what the pipeline is created with, keyed by OverrideInfo.key: the values that were set (and with defaultToZero the
  //overrides without a default). the others are left out, so the shader computes them (e.g. "override SCALE = BASE * 2.0;")
  //(a bool is passed as 1 / 0)
  public readonly values: Record<string, number> = {};

  //defaultToZero: overrides without a default start at 0 / false, so the pipeline can be created before they are set.
  //without it the pipeline fails until they are set, so a forgotten count is a validation error instead of 0
  public constructor(shaderCode: string, defaultToZero: boolean) {
    for (const override of OverrideUtils.getOverrides(shaderCode)) {
      this.overridesByName.set(override.name, override);
      if (!override.hasDefault && defaultToZero) {
        this.values[override.key] = Number(override.value);
      }
    }
  }

  public has(name: string): boolean {
    return this.overridesByName.has(name);
  }

  //the value that was set, or else the default. undefined when the default is an expression, or when there is no default
  //and it was not set (without defaultToZero), so that setting it to 0 / false is a change
  public get(name: string): boolean | number | undefined {
    const override = this.overridesByName.get(name);
    if (!override) return undefined;
    if (!Object.hasOwn(this.values, override.key)) return override.hasDefault ? override.value : undefined;
    const value = this.values[override.key];
    return override.type === 'bool' ? value !== 0 : value;
  }

  //returns whether the value changed. setting the value it already has (also the default when it was not set) is no change
  public set(name: string, value: boolean | number): boolean {
    const override = this.overridesByName.get(name);
    if (!override || Number(this.get(name)) === Number(value)) return false;
    if (override.hasDefault && Number(override.value) === Number(value)) {
      //back to the default: left out again, so the values (and the key) are the same as before it was set
      delete this.values[override.key];
    } else {
      this.values[override.key] = Number(value);
    }
    return true;
  }

  //identifies the values, e.g. to cache pipelines by: the same values give the same key, whatever order they were set in
  public get key(): string {
    return Object.keys(this.values).sort().map(key => `${key}=${this.values[key]}`).join(',');
  }

  public get names(): string[] {
    return Array.from(this.overridesByName.keys());
  }
}
