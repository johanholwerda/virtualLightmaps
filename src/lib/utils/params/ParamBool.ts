import ParamBase, {PARAM_BOOL, ParamType} from './ParamBase';

export default class ParamBool extends ParamBase<boolean> {
  public get type(): ParamType {
    return PARAM_BOOL;
  }

  public setShaderUniform(value: any): void {
    this.materials.forEach((m) => m.setFloat(this.name, value === true ? 1 : 0));
  }
}
