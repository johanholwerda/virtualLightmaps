import ParamBase, {PARAM_INT, ParamOptions, ParamType} from './ParamBase';
import ParamGroup from '../params/ParamGroup';

export default class ParamInt extends ParamBase<number> {
  constructor(parent: ParamGroup, name: string, value: number, min: number = 0, max: number = 1, options: Partial<ParamOptions<number>> = {}) {
    super(parent, name, value, {
      min:  min > value ? min - 1 : min,
      max:  max < value ? value * 2 : max,
      step: 1,
      ...options,
    });
  }

  public get type(): ParamType {
    return PARAM_INT;
  }

  // Blending presets lerps through fractional values; an int param must stay whole,
  // because these values index arrays and drive pass counts.
  public setValue(value: number, callCallback: boolean = true): void {
    super.setValue(Math.round(value), callCallback);
  }

  public setShaderUniform(value: number): void {
    this.materials.forEach((m) => m.setFloat(this.name, value));
  }
}
