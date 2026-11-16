import type Vector3 from '../../renderer/math/Vector3';
import ParamButton from './ParamButton';
import ParamBool from './ParamBool';
import ParamColor from './ParamColor';
import ParamFloat from './ParamFloat';
import ParamInt from './ParamInt';
import ParamBase, { type ParamOptions, type ParamSelectOptionsType } from './ParamBase';
import ParamText from './ParamText';
import ParamSelect from "./ParamSelect";
import ParamLog from "./ParamLog";
import ParamVector3 from "./ParamVector3";

export default class ParamGroup {
  private static _id: number = 0;

  public params: ParamBase[] = [];
  public name: string = '';
  public children: ParamGroup[] = [];
  private _collapsed: boolean = false;
  public id: number = 0;
  private collapsedCallbacks: ((collapsed: boolean) => void)[] = [];

  constructor(name: string = '') {
    this.id = ParamGroup._id++;
    this.name = name;
  }

  public addCollapsedCallback(callback: (collapsed: boolean) => void) {
    this.collapsedCallbacks.push(callback);
  }

  public set collapsed(value: boolean) {
    if (value === this._collapsed) return;
    this._collapsed = value;
    this.collapsedCallbacks.forEach(cb => cb(value));
  }

  public get collapsed(): boolean {
    return this._collapsed;
  }

  public addGroup(name: string): ParamGroup {
    const child: ParamGroup = new ParamGroup(name);
    this.children.push(child);
    return child;
  }

  //the materials in the options get the uniform with the name of the param, the callback is called on every change
  public addParamFloat(name: string, value: number, min: number = 0.0, max: number = 1.0, options: Partial<ParamOptions<number>> = {}): ParamFloat {
    return this.add(new ParamFloat(this, name, value, min, max, options));
  }

  public addParamSelect(name: string, value: number, selectOptions: ParamSelectOptionsType, options: Partial<ParamOptions<number>> = {}): ParamSelect {
    return this.add(new ParamSelect(this, name, value, ParamGroup.toSelectObject(selectOptions), options));
  }

  public addParamInt(name: string, value: number, min: number = 0, max: number = 1, options: Partial<ParamOptions<number>> = {}): ParamInt {
    return this.add(new ParamInt(this, name, value, min, max, options));
  }

  public addParamBool(name: string, value: boolean, options: Partial<ParamOptions<boolean>> = {}): ParamBool {
    return this.add(new ParamBool(this, name, value, options));
  }

  public addParamColor(name: string, value: Vector3, options: Partial<ParamOptions<Vector3>> = {}): ParamColor {
    return this.add(new ParamColor(this, name, value, options));
  }

  public addParamVector3(name: string, value: Vector3, options: Partial<ParamOptions<Vector3>> = {}): ParamVector3 {
    return this.add(new ParamVector3(this, name, value, options));
  }

  public addParamText(name: string, value: string, options: Partial<ParamOptions<string>> = {}): ParamText {
    return this.add(new ParamText(this, name, value, options));
  }

  public addButton(name: string, options: Partial<ParamOptions<void>> = {}): ParamButton {
    return this.add(new ParamButton(this, name, true, options));
  }

  public addLog(value: string) {
    return this.add(new ParamLog(this, value));
  }

  private add<T extends ParamBase>(param: T): T {
    this.params.push(param);
    return param;
  }

  //names in order get their index as value, an object of name: value (like a deserialized one) is kept
  private static toSelectObject(selectOptions: ParamSelectOptionsType): ParamSelectOptionsType {
    if (!Array.isArray(selectOptions)) return selectOptions;
    const selectObject: { [k: string]: number } = {};
    selectOptions.forEach((value, index) => selectObject[value] = index);
    return selectObject;
  }

  public getParameters(): ParamBase[] {
    // add es2019 to the lib in tsconfig.json to use flatMap
    return [...this.params, ...this.children.flatMap((child: ParamGroup) => child.getParameters())];
  }
}
