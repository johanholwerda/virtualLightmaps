import Vector3 from '../../renderer/math/Vector3';
import ParamBase, {PARAM_COLOR, ParamType} from './ParamBase';

export default class ParamColor extends ParamBase<Vector3> {
  public get type(): ParamType {
    return PARAM_COLOR;
  }

  public setValue(value: Vector3, callCallback: boolean = true): void {
    super.setValue(value.clone(), callCallback);
  }

  public setShaderUniform(value: Vector3): void {
    this.materials.forEach((m) => m.setVector3(this.name, value));
  }
}
