import ParamBase, {PARAM_TEXT, ParamType} from './ParamBase';

export default class ParamText extends ParamBase<string> {
  public get type(): ParamType {
    return PARAM_TEXT;
  }
}
