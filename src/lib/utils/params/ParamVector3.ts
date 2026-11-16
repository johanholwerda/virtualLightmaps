import Vector3 from '../../renderer/math/Vector3';
import ParamBase, { PARAM_VECTOR3, ParamType } from './ParamBase';

export default class ParamVector3 extends ParamBase<Vector3> {
  public get type(): ParamType {
    return PARAM_VECTOR3;
  }

  public setValue(value: Vector3, callCallback: boolean = true): void {
    // store a clone to avoid external mutation side-effects
    super.setValue(value.clone(), callCallback);
  }

  public setShaderUniform(value: Vector3): void {
    this.materials.forEach((m) => m.setVector3(this.name, value));
  }
}
