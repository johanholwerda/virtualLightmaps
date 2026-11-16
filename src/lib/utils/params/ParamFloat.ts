import ParamBase, {PARAM_FLOAT, ParamOptions, ParamType} from './ParamBase';
import ParamGroup from '../params/ParamGroup';
import {clamp} from "../../renderer/utils/MathUtils";

function calcDefaultPrecision(max: number) {
  max = Math.abs(max);
  if (max > 0) {
    return clamp(Math.ceil(-Math.log10(max)) + 2, 1, 5);
  } else {
    return 0;
  }
}

export default class ParamFloat extends ParamBase<number> {
  constructor(parent: ParamGroup, name: string, value: number, min: number = 0, max: number = 1, options: Partial<ParamOptions<number>> = {}) {
    super(parent, name, value, {
      min:       min > value ? min - 1 : min,
      max:       max < value ? value * 2 : max,
      precision: calcDefaultPrecision(max),
      ...options,
    });
  }

  public get type(): ParamType {
    return PARAM_FLOAT;
  }

  public setShaderUniform(value: number): void {
    this.materials.forEach((m) => m.setFloat(this.name, value));
  }
}
