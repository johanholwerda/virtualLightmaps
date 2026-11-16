import ParamGroup from "./ParamGroup";
import ParamBase, {
  PARAM_BOOL, PARAM_BUTTON, PARAM_COLOR, PARAM_FLOAT, PARAM_INT, PARAM_LOG,
  PARAM_SELECT, PARAM_TEXT, PARAM_VECTOR3
} from "./ParamBase";
import Vector3 from "../../renderer/math/Vector3";
import Color from "../../renderer/math/Color";


export function vec3ToHex(vec3: Vector3): string {
  return new Color(vec3.x, vec3.y, vec3.z, 1).getHex();
}

export function hexToVec3(hex: string) {
  return new Color().setHex(hex).vector3;
}

export function paramSerialize(param: ParamBase) {
  return {
    name:    param.name,
    type:    param.type,
    value:   param.type === PARAM_COLOR ? vec3ToHex(param.value) : param.value,
    options: param.options,
  };
}

export function paramGroupSerialize(paramGroup: ParamGroup): any {
  return {
    name:      paramGroup.name,
    collapsed: paramGroup.collapsed,
    params:    paramGroup.params.map(param => paramSerialize(param)),
    children:  paramGroup.children.map(child => paramGroupSerialize(child)),
  };
}

export function paramGroupDeserialize(data: any) {
  const paramGroup = new ParamGroup(data.name || '');
  paramGroup.collapsed = !!data.collapsed;
  if (data.params) {
    for (const paramData of data.params) {
      const opts = paramData.options || {};
      switch (paramData.type) {
        case PARAM_INT:
          paramGroup.addParamInt(paramData.name, paramData.value, opts.min, opts.max, opts);
          break;
        case PARAM_FLOAT:
          paramGroup.addParamFloat(paramData.name, paramData.value, opts.min, opts.max, opts);
          break;
        case PARAM_BOOL:
          paramGroup.addParamBool(paramData.name, paramData.value, opts);
          break;
        case PARAM_BUTTON:
          paramGroup.addButton(paramData.name, opts);
          break;
        case PARAM_TEXT:
          paramGroup.addParamText(paramData.name, paramData.value, opts);
          break;
        case PARAM_SELECT:
          paramGroup.addParamSelect(paramData.name, paramData.value, opts.selectOptions || [], opts);
          break;
        case PARAM_LOG:
          paramGroup.addLog(paramData.value || '');
          break;
        case PARAM_COLOR:
          paramGroup.addParamColor(paramData.name, typeof paramData.value === 'string' ? hexToVec3(paramData.value) : new Vector3(paramData.value.x, paramData.value.y, paramData.value.z), opts);
          break;
        case PARAM_VECTOR3:
          paramGroup.addParamVector3(paramData.name, new Vector3(paramData.value.x, paramData.value.y, paramData.value.z), opts);
          break;
        default:
          break;
      }
    }
  }
  if (data.children) {
    for (const childData of data.children) {
      paramGroup.children.push(paramGroupDeserialize(childData));
    }
  }
  return paramGroup;
}

export function addCallBackToAllParams(paramGroup: ParamGroup, callback: (param: ParamBase) => ((value: any) => void)) {
  paramGroup.params.forEach(param => param.addCallback(callback(param)));
  paramGroup.children.forEach(child => addCallBackToAllParams(child, callback));
}

export function changeValueOfParamInGroup(paramGroup: ParamGroup, id: number, value: any) {
  paramGroup.params.forEach(param => {
    if (param.id === id) {
      const v = param.type === PARAM_COLOR ? hexToVec3(value) :
                param.type === PARAM_VECTOR3 ? new Vector3(value.x, value.y, value.z) :
                value;
      param.setValue(v);
    }
  });
  paramGroup.children.forEach(child => changeValueOfParamInGroup(child, id, value));

}