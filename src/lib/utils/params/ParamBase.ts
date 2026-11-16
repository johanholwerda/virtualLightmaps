import ParamGroup from './ParamGroup';
import type Vector3 from "../../renderer/math/Vector3";

export type ParamCallback<T = any> = (value: T) => void;
export type ParamSelectOptionsType = { [k: string]: number } | string[];

export interface ParamMaterial {
  setFloat(name: string, value: number): void;

  setVector3(name: string, value: Vector3): void;

  hasUniform(name: string): boolean;
}

// Turns the field's label into a clickable trigger.
export type ParamAction = {
  title?: string;
  onClick: () => void;
};

export type ParamOptions<T = any> = {
  min: number;
  max: number;
  callback?: ParamCallback<T>; // called when the value changes
  materials?: ParamMaterial[]; // the uniform with the name of the param is set on these (the ones without it are skipped)
  step?: number;
  precision?: number;
  selectOptions?: ParamSelectOptionsType;
  typeIn?: boolean;
  action?: ParamAction;
  compact?: boolean; // narrow label, tighter row
};

export const PARAM_BOOL = 0;
export const PARAM_INT = 1;
export const PARAM_FLOAT = 2;
export const PARAM_COLOR = 3;
export const PARAM_TEXT = 4;
export const PARAM_BUTTON = 5;
export const PARAM_SELECT = 6;
export const PARAM_LOG = 7;
export const PARAM_VECTOR3 = 8;

export type ParamType = typeof PARAM_BOOL | typeof PARAM_INT | typeof PARAM_FLOAT | typeof PARAM_TEXT | typeof PARAM_BUTTON | typeof PARAM_COLOR | typeof PARAM_SELECT | typeof PARAM_LOG | typeof PARAM_VECTOR3;

export default class ParamBase<T = any> {
  private static _id: number = 0;
  //grows on every change of any param, so a renderer that stops rendering a converged image knows when to start again
  public static changeCount: number = 0;

  public name: string;
  public options: ParamOptions;
  public id: number;
  public parent: ParamGroup; //needed for preset manager

  protected materials: ParamMaterial[];
  protected callbacks: ((value: T) => void)[] = [];
  protected _value!: T;

  constructor(parent: ParamGroup, name: string, value: T, options: Partial<ParamOptions<T>> = {}) {
    this.id = ParamBase._id++;
    //callback and materials are no settings of the ui (the options get serialized), so they stay out of the options
    const { callback = null, materials = [], ...settings } = options;
    this.options = { min: 0, max: 1, ...settings };

    this.name = name;
    this.parent = parent;

    this.materials = materials.filter((m) => m.hasUniform(this.name));
    if (callback !== null) {
      this.addCallback(callback);
    }

    this.setValue(value, false);
  }

  public get type(): ParamType {
    return PARAM_BOOL;
  }

  // override
  public get value(): T {
    return this._value;
  }

  public addCallback(callback: (value: T) => void) {
    this.callbacks.push(callback);
  }

  public triggerCallbacks() {
    ParamBase.changeCount++;
    this.callbacks.forEach((callback) => callback(this.value));
  }

  public setValue(value: T, callCallback: boolean = true): void {
    if (value !== this._value) {
      this._value = value;
      ParamBase.changeCount++;

      this.setShaderUniform(value);

      if (callCallback) {
        this.triggerCallbacks();
      }
    }
  }

  public setFromUI(value: T): void {
    // LogGL.log ("setFromUI:",  this.name, value);
    this.setValue(value, true);
  }

  public setShaderUniform(value: T): void {
  }
}
