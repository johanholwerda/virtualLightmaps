import ParamBase, {PARAM_BUTTON, ParamType} from './ParamBase';

export default class ParamButton extends ParamBase<any> {
  public get type(): ParamType {
    return PARAM_BUTTON;
  }
}
