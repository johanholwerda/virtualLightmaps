import ParamGroup from '../params/ParamGroup';
import type ParamSelect from '../params/ParamSelect';
import { PARAM_BOOL, PARAM_BUTTON, PARAM_COLOR, PARAM_FLOAT, PARAM_INT, PARAM_LOG, PARAM_SELECT, PARAM_TEXT, PARAM_VECTOR3 } from '../params/ParamBase';
import {
  guiCreateButtonField,
  guiCreateCheckboxField,
  guiCreateColorField,
  guiCreateContainer,
  guiCreateGroup,
  guiCreateLogField,
  guiCreateNumberField,
  guiCreateSelectField,
  guiCreateSliderField,
  guiCreateTextField,
  guiCreateVector3Field,
  GUI_MIN_WIDTH
} from "./SimpleGui";
import { guiCreateProfilerToggle, guiCreateStats, guiCreateUncappedToggle } from "./SimpleGuiStats";
import IDestructible from "../../renderer/core/IDestructible";
import { addSaveLoadButtons } from "./SimpleGuiLoadSave";

export type ParamGroupUI = { div: HTMLElement, children: ParamGroupUI[] };

export default class ParamsGui implements IDestructible {
  public div: ParamGroupUI | undefined;
  private gui: HTMLDivElement | undefined;
  private rootGroup: ParamGroup | undefined;
  private addStats: boolean = true;
  private minWidth: number = GUI_MIN_WIDTH;

  constructor(rootGroup: ParamGroup, collapseChildren = false, addLoadSaveButtons = false, addStats = true, minWidth = GUI_MIN_WIDTH) {
    if (typeof window === 'undefined') return;

    this.rootGroup = rootGroup;
    this.addStats = addStats;
    this.minWidth = minWidth;

    this.div = this.readGroup(undefined, rootGroup, guiCreateContainer(), collapseChildren);

    if (addLoadSaveButtons) addSaveLoadButtons(this.rootGroup, this.gui || null);
  }

  destruct() {
    if (this.gui && this.gui.parentElement) {
      this.gui.parentElement.removeChild(this.gui);
    }
    this.gui = undefined;
  }

  private readGroup(parentGroup: ParamGroup | undefined, group: ParamGroup, container: HTMLElement, collapseChildren: boolean): ParamGroupUI {
    if (!group.name) {
      throw new Error('group name not initialized');
    }

    const gui = guiCreateGroup(container, group, this.minWidth);
    const groupUI = { div: gui, children: new Array<ParamGroupUI>() };

    if (!parentGroup) {
      this.gui = gui.parentElement as HTMLDivElement;
      // Stats belong to the main gui only, so secondary panels don't repeat them.
      if (this.addStats && !document.querySelector('.gui-stats-row')) {
        const stats = guiCreateStats(gui);
        guiCreateProfilerToggle(stats);
        guiCreateUncappedToggle(stats);
      }
    }

    for (let p of group.params) {
      switch (p.type) {
        case PARAM_FLOAT:
        case PARAM_INT:
          if (p.options.typeIn) {
            guiCreateNumberField(gui, p);
          } else {
            guiCreateSliderField(gui, p);
          }
          break;
        case PARAM_COLOR:
          guiCreateColorField(gui, p);
          break;
        case PARAM_BOOL:
          guiCreateCheckboxField(gui, p);
          break;
        case PARAM_BUTTON:
          guiCreateButtonField(gui, p);
          break;
        case PARAM_TEXT:
          guiCreateTextField(gui, p);
          break;
        case PARAM_SELECT:
          guiCreateSelectField(gui, p as ParamSelect);
          break;
        case PARAM_LOG:
          guiCreateLogField(gui, p);
          break;
        case PARAM_VECTOR3:
          guiCreateVector3Field(gui, p);
          break;
        default:
          // default case if none of the above cases match
          break;
      }
    }

    for (let child of group.children) {
      child.collapsed = child.collapsed || collapseChildren;
      groupUI.children.push(this.readGroup(group, child, gui, collapseChildren));
    }

    return groupUI;
  }
}