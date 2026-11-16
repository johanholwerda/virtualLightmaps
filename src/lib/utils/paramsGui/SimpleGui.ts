import ParamButton from '../params/ParamButton';
import ParamInt from '../params/ParamInt';
import ParamFloat from '../params/ParamFloat';
import ParamBool from '../params/ParamBool';
import Vector3 from "../../renderer/math/Vector3";
import Color from "../../renderer/math/Color";
import ParamColor from "../params/ParamColor";
import ParamGroup from "../params/ParamGroup";
import ParamText from "../params/ParamText";
import { isNumber } from "../../renderer/utils/MathUtils";
import ParamBase, { PARAM_FLOAT, PARAM_INT } from "../params/ParamBase";
import ParamSelect from "../params/ParamSelect";

const inputEvents = ['input', 'change'];
const suiCSS = `
.gui {
  position: fixed;
  right: 0;
  top: 0;
  z-index: 999;
  display: flex;
  flex-direction: row;
  color: white;
  font: 12px sans-serif;
  user-select: none;
  align-items: flex-start;
}
.gui-group {
  overflow: hidden;
}
.gui-group::-webkit-scrollbar {
  display: none;
}
.gui-group[open] {
  margin-bottom: 6px;
}
.gui > .gui-group {
  max-height: calc(100vh - 8px);
  overflow-y: auto;
  margin-right: 8px;
  background-color: rgba(51, 51, 51, 0.7);
  border-radius: 0 0 8px 8px;
  min-width: 280px;
  position:relative;
}
.gui-group-header {
  padding: 9px 8px;
  cursor: pointer;
  background-color: #000;
  font-weight: bold;
}
.gui-group-header:hover {
  opacity: 0.8;
}
.gui-group .gui-group .gui-group .gui-group-header {
  margin-left: 8px;
  background: #222;
}
.gui-mover {
  position: absolute;
  bottom: 0;
  top: 0;
  width: 7px;
  background: #555;
  opacity: 0;
}
.gui-mover:hover {
  cursor: ew-resize;
  opacity: 0.5;
}
.gui-form-field {
  display: flex;
  flex-direction: row;
  align-items: center;
  min-height: 32px;
  margin: 0 8px;
}
.gui-stats-row {
  display: flex;
  flex-direction: row;
  align-items: center;
  gap: 8px;
  margin: 4px 8px 0;
}
.gui-canvas {
  flex: 1 1 0;
  min-width: 0;
  height: 26px !important;
}
.gui-stats-row input[type="checkbox"] {
  flex: 0 0 auto;
}
.gui-stats-toggle {
  flex: 0 0 auto;
  margin-right: -6px;
  font-size: 11px;
  cursor: pointer;
  user-select: none;
}
.gui label {
  width: 40%;
  min-width: 120px;
  flex-shrink: 0;
  text-overflow: ellipsis;
  overflow: hidden;
  color: lightgray;
  white-space: nowrap;
}
.gui output {
  min-width: 50px;
  max-width: 50px;
  text-align: right;
  color: lightgray;
  cursor: pointer;
}
.gui pre {
  padding: 4px;
  padding-left: 16px;
  background-color: #6662;
  margin: 8px;
}
.gui input {
  flex-grow: 1;
  height: 24px;
}
.gui input[type="checkbox"] {
  height: 16px;
}
.gui button, .gui select {
  background-color: #666;
  border: none;
  border-radius: 4px;
  color: white;
  text-align: center;
  width: 100%;
  padding: 6px 16px;
  cursor: pointer;
}
.gui button:hover {
  background-color: #555;
}
.gui input:not([type='text']) {
  cursor: pointer;
}
/* Hide number input spinners */
.gui-vector3-input::-webkit-outer-spin-button,
.gui-vector3-input::-webkit-inner-spin-button {
  -webkit-appearance: none;
  margin: 0;
}
.gui-vector3-input[type=number] {
  -moz-appearance: textfield;
}
/* Compact row: small label and field on one line */
.gui-form-field.gui-form-field-compact {
  min-height: 26px;
  gap: 6px;
}
.gui .gui-form-field-compact label {
  width: auto;
  min-width: 26px;
  text-align: center;
}
.gui .gui-form-field-compact input {
  min-width: 0;
}
/* Label doubling as a button */
.gui .gui-label-action {
  cursor: pointer;
  color: white;
  background-color: #666;
  border-radius: 4px;
  padding: 5px 6px;
  line-height: 1;
}
.gui .gui-label-action:hover {
  background-color: #555;
}
.gui .gui-form-field-active .gui-label-action {
  background-color: #0af;
}
.gui .gui-form-field-active input[type='text'] {
  box-shadow: inset 0 0 0 2px #0af;
}
.gui label.draggable, .gui .gui-vector3-label.draggable {
  cursor: ew-resize;
  color: #fff;
}
.gui label.draggable:active, .gui .gui-vector3-label.draggable:active {
  color: #0af;
}
`;


function copyToClipboard(text: string) {
  console.log(text);
  navigator.clipboard.writeText(text);
}

function guiCreateFormField(param: ParamBase | undefined, field: HTMLElement, output: HTMLElement | undefined = undefined) {
  const container = document.createElement('div');
  container.className = 'gui-form-field';
  if (param) {
    const label = document.createElement('label');
    label.innerText = param.name;
    label.title = param.name;
    label.setAttribute('for', `input-${param.id}`);
    container.appendChild(label);
  }
  container.appendChild(field);
  if (output) {
    container.appendChild(output);
  }
  return { container, label: container.querySelector('label') as HTMLLabelElement };
}

function addLabelDragging(label: HTMLElement, param: ParamBase, updateFn: (delta: number) => void) {
  label.classList.add('draggable');
  let startX = 0;
  let isDragging = false;

  const onMouseMove = (e: MouseEvent) => {
    if (!isDragging) return;
    const delta = e.clientX - startX;
    startX = e.clientX;
    updateFn(delta);
  };

  const onMouseUp = () => {
    isDragging = false;
    document.removeEventListener('mousemove', onMouseMove);
    document.removeEventListener('mouseup', onMouseUp);
    document.body.style.cursor = '';
  };

  label.addEventListener('mousedown', (e) => {
    startX = e.clientX;
    isDragging = true;
    document.addEventListener('mousemove', onMouseMove);
    document.addEventListener('mouseup', onMouseUp);
    document.body.style.cursor = 'ew-resize';
    e.preventDefault();
  });
}

export function guiCreateButtonField(parent: HTMLElement, button: ParamButton) {
  const buttonElement = document.createElement('button');
  buttonElement.innerText = button.name;
  buttonElement.addEventListener('click', () => button.triggerCallbacks());

  parent.appendChild(guiCreateFormField(undefined, buttonElement).container);
}

export function guiCreateSliderField(parent: HTMLElement, param: ParamFloat | ParamInt) {
  const sliderElement = document.createElement('input');
  const o = param.options;
  sliderElement.id = `input-${param.id}`;
  sliderElement.type = 'range';
  sliderElement.min = o.min.toString();
  sliderElement.max = o.max.toString();
  sliderElement.step = o.step !== undefined ? `${o.step}` : 'any';

  const outputElement = document.createElement('output');

  const setValue = (v: number) => {
    sliderElement.value = `${v}`;
    sliderElement.title = `${v.toFixed(o.precision || 0)}`;
    outputElement.innerText = `${v.toFixed(o.precision || 0)}`;
  };

  setValue(param.value);

  outputElement.addEventListener('click', () => copyToClipboard(outputElement.innerText));
  outputElement.addEventListener('dblclick', () => {
    let valueFromPrompt = prompt(`Enter value [${o.min}-${o.max}]`, sliderElement.value);
    if (valueFromPrompt !== null && valueFromPrompt !== '') {
      const value = parseFloat(valueFromPrompt);
      if (isNumber(valueFromPrompt)) {
        param.setValue(param.type === PARAM_INT ? Math.round(value) : value, true);
      }
    }
  });

  inputEvents.forEach(event => {
    sliderElement.addEventListener(event, (e) => {
      const parsed = parseFloat((e.target as HTMLInputElement).value);
      if (!isNaN(parsed)) {
        param.setValue(param.type === PARAM_INT ? Math.round(parsed) : parsed, true);
      }
    });
  });
  param.addCallback(setValue);

  parent.appendChild(guiCreateFormField(param, sliderElement, outputElement).container);
}

export function guiCreateNumberField(parent: HTMLElement, param: ParamFloat | ParamInt) {
  const inputElement = document.createElement('input');
  const o = param.options;
  inputElement.id = `input-${param.id}`;
  inputElement.type = 'number';
  inputElement.step = o.step !== undefined ? `${o.step}` : 'any';
  inputElement.classList.add('gui-vector3-input'); // reuse same styling for now
  inputElement.style.flex = '1';

  const setValue = (v: number) => {
    inputElement.value = `${v}`;
  };

  setValue(param.value);

  inputElement.addEventListener('change', (e) => {
    const parsed = parseFloat((e.target as HTMLInputElement).value);
    if (!isNaN(parsed)) {
      param.setValue(param.type === PARAM_INT ? Math.round(parsed) : parsed, true);
    }
  });

  param.addCallback(setValue);

  const { container, label } = guiCreateFormField(param, inputElement);
  if (label) {
    addLabelDragging(label, param, (delta) => {
      const step = o.step !== undefined ? o.step : (param.type === PARAM_FLOAT ? 0.01 : 1);
      const nextVal = param.value + delta * step;
      param.setFromUI(param.type === PARAM_INT ? Math.round(nextVal) : nextVal);
    });
  }
  parent.appendChild(container);
}

export function guiCreateColorField(parent: HTMLElement, param: ParamColor) {
  const colorElement = document.createElement('input');
  colorElement.id = `input-${param.id}`;
  colorElement.type = 'color';
  colorElement.value = new Color(param.value.x, param.value.y, param.value.z, 1).getHex();

  const outputElement = document.createElement('output');
  outputElement.addEventListener('click', () => copyToClipboard(outputElement.innerText));
  outputElement.addEventListener('dblclick', () => {
    const result = prompt('Enter value', outputElement.innerText);
    if (result) {
      param.setValue(new Color().setHex(result).vector3, true);
    }
  });
  const setValue = (v: Vector3) => {
    const hex = colorElement.value = new Color(v.x, v.y, v.z, 1).getHex();
    colorElement.value = `${hex}`;
    outputElement.innerText = `${hex}`;
  };

  setValue(param.value);

  inputEvents.forEach(event => {
    colorElement.addEventListener(event, (e) => {
      param.setValue(new Color().setHex(colorElement.value).vector3, true);
    });
  });
  param.addCallback(setValue);

  parent.appendChild(guiCreateFormField(param, colorElement, outputElement).container);
}

export function guiCreateCheckboxField(parent: HTMLElement, param: ParamBool) {
  const checkboxElement = document.createElement('input');
  checkboxElement.id = `input-${param.id}`;
  checkboxElement.type = 'checkbox';
  checkboxElement.checked = param.value;
  checkboxElement.addEventListener('change', (e) => {
    param.setValue((e.target as HTMLInputElement).checked, true);
  });
  param.addCallback((v: boolean) => checkboxElement.checked = v);

  parent.appendChild(guiCreateFormField(param, checkboxElement).container);
}

export function guiCreateSelectField(parent: HTMLElement, param: ParamSelect) {
  const selectElement = document.createElement('select');
  const o = param.options;

  selectElement.id = `input-${param.id}`;

  // add options from o.selectOptions to selectElement
  for (const [text, value] of Object.entries(o.selectOptions || {})) {
    const option = document.createElement('option');
    option.value = `${value}`;
    option.text = text;
    selectElement.appendChild(option);
  }

  const outputElement = document.createElement('output');

  const setValue = (v: number) => {
    selectElement.value = `${v}`;
    const formatted = o.precision !== undefined ? v.toFixed(o.precision) : `${v}`;
    selectElement.title = formatted;
    outputElement.innerText = formatted;
  };

  setValue(param.value);

  outputElement.addEventListener('click', () => copyToClipboard(outputElement.innerText));
  inputEvents.forEach(event => {
    selectElement.addEventListener(event, (e) => {
      param.setValue(parseFloat((e.target as HTMLInputElement).value), true);
    });
  });
  param.addCallback(setValue);

  parent.appendChild(guiCreateFormField(param, selectElement, outputElement).container);
}

export function guiCreateTextField(parent: HTMLElement, param: ParamText) {
  const textElement = document.createElement('input');
  textElement.id = `input-${param.id}`;
  textElement.type = 'text';
  textElement.value = param.value;
  inputEvents.forEach(event => {
    textElement.addEventListener(event, (e) => {
      param.setValue((e.target as HTMLInputElement).value, true);
    });
  });
  param.addCallback((v: string) => textElement.value = v);

  const { container, label } = guiCreateFormField(param, textElement);
  if (param.options.compact) {
    container.classList.add('gui-form-field-compact');
  }

  const action = param.options.action;
  if (action && label) {
    // Clicking the label triggers the action instead of focusing the input.
    label.removeAttribute('for');
    label.classList.add('gui-label-action');
    label.title = action.title || param.name;
    label.addEventListener('click', () => action.onClick());
  }

  parent.appendChild(container);
}

export function guiCreateLogField(parent: HTMLElement, param: ParamBase) {
  const logElement = document.createElement('pre');
  logElement.innerText = param.value.toString();
  param.addCallback((v: any) => logElement.innerText = v.toString());

  parent.appendChild(logElement);
}

export function guiCreateVector3Field(parent: HTMLElement, param: ParamBase) {
  const container = document.createElement('div');
  container.style.display = 'flex';
  container.style.gap = '2px';
  container.style.flexGrow = '1';  // Take available space from form field

  const makeNumberInput = (label: string, idSuffix: string, initial: number) => {
    const wrapper = document.createElement('div');
    wrapper.style.display = 'flex';
    wrapper.style.alignItems = 'center';
    wrapper.style.gap = '2px';
    wrapper.style.flex = '1';
    wrapper.style.minWidth = '0';  // Important: allow shrinking

    const labelEl = document.createElement('span');
    labelEl.innerText = label;
    labelEl.style.color = '#999';
    labelEl.style.fontSize = '10px';
    labelEl.style.flexShrink = '0';

    const inp = document.createElement('input');
    inp.type = 'number';
    inp.step = 'any';
    inp.value = `${initial}`;
    inp.id = `input-${param.id}-${idSuffix}`;
    inp.style.flex = '1';
    inp.style.minWidth = '0';
    inp.style.width = '0';  // Force flex to control width
    inp.style.padding = '2px';
    inp.style.fontSize = '11px';
    inp.classList.add('gui-vector3-input');  // Add class for CSS targeting

    wrapper.appendChild(labelEl);
    wrapper.appendChild(inp);
    return { wrapper, input: inp, label: labelEl };
  };

  const { wrapper: xWrapper, input: xInput, label: xLabel } = makeNumberInput('x', 'x', param.value.x);
  const { wrapper: yWrapper, input: yInput, label: yLabel } = makeNumberInput('y', 'y', param.value.y);
  const { wrapper: zWrapper, input: zInput, label: zLabel } = makeNumberInput('z', 'z', param.value.z);

  addLabelDragging(xLabel, param, (delta) => {
    const step = param.options.step !== undefined ? param.options.step : 0.01;
    const v = param.value.clone();
    v.x += delta * step;
    param.setFromUI(v as any);
  });
  xLabel.classList.add('gui-vector3-label');

  addLabelDragging(yLabel, param, (delta) => {
    const step = param.options.step !== undefined ? param.options.step : 0.01;
    const v = param.value.clone();
    v.y += delta * step;
    param.setFromUI(v as any);
  });
  yLabel.classList.add('gui-vector3-label');

  addLabelDragging(zLabel, param, (delta) => {
    const step = param.options.step !== undefined ? param.options.step : 0.01;
    const v = param.value.clone();
    v.z += delta * step;
    param.setFromUI(v as any);
  });
  zLabel.classList.add('gui-vector3-label');

  const precision = param.options.precision !== undefined ? param.options.precision : 4;
  const setInputs = (v: Vector3) => {
    xInput.value = `${parseFloat(v.x.toFixed(precision))}`;
    yInput.value = `${parseFloat(v.y.toFixed(precision))}`;
    zInput.value = `${parseFloat(v.z.toFixed(precision))}`;
  };

  const onChange = () => {
    const x = parseFloat(xInput.value);
    const y = parseFloat(yInput.value);
    const z = parseFloat(zInput.value);
    if (!isNaN(x) && !isNaN(y) && !isNaN(z)) {
      param.setFromUI(new Vector3(x, y, z) as any);
    }
  };

  // Only listen to 'change' event (fires on blur or Enter key)
  xInput.addEventListener('change', onChange);
  yInput.addEventListener('change', onChange);
  zInput.addEventListener('change', onChange);

  param.addCallback((v: any) => setInputs(v as Vector3));

  container.appendChild(xWrapper);
  container.appendChild(yWrapper);
  container.appendChild(zWrapper);

  parent.appendChild(guiCreateFormField(param, container).container);
}

export const GUI_MIN_WIDTH = 280;

export function guiCreateGroup(parent: HTMLElement, paramGroup: ParamGroup, minWidth: number = GUI_MIN_WIDTH) {
  const groupContainer = document.createElement('details');

  groupContainer.className = `gui-group`;
  groupContainer.open = !paramGroup.collapsed;

  parent.appendChild(groupContainer);

  const header = document.createElement('summary');
  header.className = 'gui-group-header';
  groupContainer.appendChild(header);
  header.innerText = paramGroup.name;
  const body = document.createElement('div');
  body.className = 'gui-group-body';
  groupContainer.appendChild(body);

  // When the user toggles the UI, update the model
  groupContainer.addEventListener('toggle', () => {
    paramGroup.collapsed = !groupContainer.open;
  });

  // When the model changes programmatically, update the UI
  paramGroup.addCollapsedCallback((collapsed: boolean) => {
    groupContainer.open = !collapsed;
  });

  if (parent?.classList.contains('gui')) {
    if (minWidth !== GUI_MIN_WIDTH) {
      // width too, otherwise the panel keeps the max-content width of its fields
      groupContainer.style.minWidth = `${minWidth}px`;
      groupContainer.style.width = `${minWidth}px`;
    }

    const guiMover = document.createElement("div");
    guiMover.className = "gui-mover";
    groupContainer.appendChild(guiMover);

    let startMouseX: number = 0;
    let startWidth: number = 0;

    const handleResize = (event: MouseEvent) => {
      if (groupContainer) {
        const deltaX = startMouseX - event.x;
        groupContainer.style.width = Math.max(startWidth + deltaX, minWidth) + "px";
      }
    };

    const onMouseUp = () => {
      document.removeEventListener("mousemove", handleResize, false);
      document.removeEventListener("mouseup", onMouseUp, false);
    };

    guiMover.addEventListener("mousedown", event => {
      startMouseX = event.x;
      startWidth = groupContainer.getBoundingClientRect().width;
      document.addEventListener("mousemove", handleResize, false);
      document.addEventListener("mouseup", onMouseUp, false);
    }, false);

  }

  return body;
}

let container: HTMLDivElement | null = null;

export function guiCreateContainer() {
  if (container) {
    return container;
  }
  const style = document.createElement('style');
  style.innerHTML = suiCSS;
  document.head.appendChild(style);

  container = document.createElement('div');
  container.className = 'gui';

  document.body.appendChild(container);

  return container;
}