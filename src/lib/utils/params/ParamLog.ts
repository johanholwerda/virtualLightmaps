import ParamBase, {PARAM_LOG, ParamType} from "./ParamBase";
import ParamGroup from "./ParamGroup";

export default class ParamLog extends ParamBase<string> {
  constructor(parent: ParamGroup, value: string) {
    super(parent, 'log', value);
  }

  //a log shows a result, it is no input: a renderer that waits for a param change to restart (see changeCount) ignores it
  public setValue(value: string, callCallback: boolean = true): void {
    const changeCount = ParamBase.changeCount;
    super.setValue(value, callCallback);
    ParamBase.changeCount = changeCount;
  }

  public get type(): ParamType {
    return PARAM_LOG;
  }
}