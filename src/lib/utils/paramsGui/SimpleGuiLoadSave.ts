import ParamGroup from "../params/ParamGroup";
import {guiCreateButtonField} from "./SimpleGui";
import {hexToVec3, paramGroupSerialize} from "../params/ParamGroupUtils";
import {PARAM_COLOR, PARAM_VECTOR3, ParamType} from "../params/ParamBase";
import Vector3 from "../../renderer/math/Vector3";


export function addSaveLoadButtons(rootGroup: ParamGroup, gui: HTMLElement | null) {
  if (!gui) return;

  const buttonContainer = document.createElement('div');
  const buttonGroup = new ParamGroup();
  const save = buttonGroup.addButton('Save JSON', { callback: () => saveJson(rootGroup) });
  const load = buttonGroup.addButton('Load JSON', { callback: () => loadJson(rootGroup) });
  guiCreateButtonField(buttonContainer, save);
  guiCreateButtonField(buttonContainer, load);
  gui.appendChild(buttonContainer);
}

function saveJson(rootGroup: ParamGroup) {
  const data = paramGroupSerialize(rootGroup);
  const jsonString = JSON.stringify(data, null, 2);
  const blob = new Blob([jsonString], {type: 'application/json'});
  const url = URL.createObjectURL(blob);

  const a = document.createElement('a');
  a.href = url;
  a.download = 'params.json';
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

function copyValues(data: any, target: ParamGroup) {
  if (!data || !target) return;

  if (Array.isArray(data.children)) {
    data.children.forEach((childData: any) => {
      const targetChild = target.children.find((c) => c.name === childData.name);
      if (targetChild) {
        copyValues(childData, targetChild);
      }
    });
  }

  if (Array.isArray(data.params)) {
    data.params.forEach((param: {
      value: any;
      type: ParamType;
      name: string;
    }) => {
      const targetParam = target.params.find(a => a.name === param.name);
      if (!targetParam) return;
      const v = param.type === PARAM_COLOR
        ? (typeof param.value === 'string' ? hexToVec3(param.value) : (param.value instanceof Vector3 ? param.value : new Vector3(param.value.x, param.value.y, param.value.z)))
        : param.type === PARAM_VECTOR3
        ? (param.value instanceof Vector3 ? param.value : new Vector3(param.value.x, param.value.y, param.value.z))
        : param.value;
      targetParam.setValue(v);
    });
  }
}

function loadJson(rootGroup: ParamGroup) {
  const input = document.createElement('input');
  input.type = 'file';
  input.accept = '.json';
  input.onchange = (event) => {
    const file = (event.target as HTMLInputElement).files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const result = e.target?.result;
        if (!result) {
          console.error('No file content read');
          return;
        }

        const data = JSON.parse(result as string);
        copyValues(data, rootGroup);
      } catch (error) {
        console.error('Error details:', error);
        console.error('File content that failed to parse:', e.target?.result);
      }
    };

    reader.onerror = (error) => {
      console.error('FileReader error:', error);
    };

    reader.readAsText(file);
  };
  input.click();
}
