import ParamBase, {PARAM_SELECT, ParamOptions, ParamSelectOptionsType, ParamType} from './ParamBase';
import ParamGroup from './ParamGroup';

export default class ParamSelect extends ParamBase<number> {
  constructor(parent: ParamGroup, name: string, value: number, selectOptions: ParamSelectOptionsType, options: Partial<ParamOptions<number>> = {}) {
    super(parent, name, value, {
      selectOptions,
      ...options,
    });
  }

  public get type(): ParamType {
    return PARAM_SELECT;
  }

  public setOptions(selectOptions: ParamSelectOptionsType): void {
    this.options.selectOptions = selectOptions;
  }

  public setShaderUniform(value: number): void {
    this.materials.forEach((m) => m.setFloat(this.name, value));
  }
}
