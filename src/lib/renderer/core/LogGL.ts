let debugDiv: HTMLDivElement | undefined;

function getDebugDiv() {
  if (typeof document !== "undefined" && !debugDiv) {
    debugDiv = document.createElement('div');
    const style = debugDiv.style;
    style.position = 'fixed';
    style.left = '0';
    style.bottom = '0';
    style.color = '#fff';
    style.backgroundColor = '#22222288';
    style.overflowX = 'hidden';
    style.overflowY = 'auto';
    style.display = 'block';
    style.boxSizing = 'border-box';
    style.width = '100vw';
    style.height = '30vh';
    style.padding = '16px';
    style.margin = '0';
    style.font = '12px monospace';
    style.zIndex = '998';

    document.body.appendChild(debugDiv);
  }
  return debugDiv;
}

function clearDiv() {
  const div = getDebugDiv();
  if (div) {
    div.innerHTML = '';
  }
}

function logToDiv(html: string) {
  const div = getDebugDiv();
  if (div) {
    //appending, innerHTML += would parse the whole log again
    div.insertAdjacentHTML('beforeend', html);

    div.scrollTop = div.scrollHeight;
  }

}

function stringify(obj: any) {
  let cache: any[] = [];
  return JSON.stringify(obj, function (key, value) {
    if (typeof value === 'object' && value !== null) {
      if (cache.indexOf(value) !== -1) {
        // Circular reference found, discard key
        return;
      }
      // Store value in our collection
      cache.push(value);
    }
    return value;
  });
}

function logToDivWithColor(color: string, ...messages: any[]) {
  if (typeof document === 'undefined') {
    console.log(...messages);
    return;
  }

  const styled = messages.length > 1 && messages[0] && messages[0].toString().indexOf('%c') >= 0;
  messages[0] = messages[0].toString().replace(/\t/gi, '&nbsp;&nbsp;');
  messages[0] = messages[0].toString().replace(/ /gi, '&nbsp;');


  let str = `<span ${color ? `style="color: ${color}; font-weight: bold;"` : ''}>`;
  if (styled) {
    let i = 0;
    str += messages[0].replace(/%c/gi, () => {
      i++;
      return `</span><span style="${messages[i]}">`;
    });
  } else {
    str += `${messages.map((a) => (a && a !== a.toString() ? stringify(a) : a)).join(' ')}`;
  }
  logToDiv((str + `</span><br />`).replace(/\n/gi, '<br />'));
}

export default class LogGL {
  private static _enabled: boolean = false;
  private static _logToScreen: boolean = false;
  private static _screenClearFlag: boolean = true;

  public static get ENABLED(): boolean {
    return LogGL._enabled;
  }

  public static set ENABLED(v: boolean) {
    LogGL._enabled = v;

    if (LogGL.ENABLED) {
      if (LogGL.LOG_TO_SCREEN) {
        LogGL.log = LogGL.screenLog;
        LogGL.warn = LogGL.screenWarn;
        LogGL.error = LogGL.screenError;
      } else {
        LogGL.log = console.log;
        LogGL.warn = console.warn;
        LogGL.error = LogGL.screenError;
      }
    } else {
      LogGL.log = (...messages: any[]) => {
      };
      LogGL.warn = (...messages: any[]) => {
      };
      LogGL.error = LogGL.LOG_TO_SCREEN ? LogGL.screenError : console.error;
    }
  }

  public static get LOG_TO_SCREEN(): boolean {
    return LogGL._logToScreen;
  }

  public static set LOG_TO_SCREEN(v: boolean) {
    LogGL._logToScreen = v;
    LogGL.ENABLED = LogGL.ENABLED;
  }

  public static log(...messages: any[]) {
  }

  //errors are logged also before ENABLED is set, like after setting it to false
  public static error(...messages: any[]) {
    console.error(...messages);
  }

  public static warn(...messages: any[]) {
  }

  public static screen(...messages: any[]) {
    // clear div for new frame
    if (LogGL._screenClearFlag === true) {
      clearDiv();
      LogGL._screenClearFlag = false;
      setTimeout(() => {
        LogGL._screenClearFlag = true;
      }, 0);
    }
    LogGL.screenLog('', ...messages);
    const div = getDebugDiv();
    if (div) {
      div.style.height = '30px';
    }
  }

  private static screenLog(...messages: any[]) {
    logToDivWithColor('', ...messages);
  }

  private static screenWarn(...messages: any[]) {
    logToDivWithColor('#fa4', ...messages);
  }

  //also to the console, for the stack trace
  private static screenError(...messages: any[]) {
    console.error(...messages);
    logToDivWithColor('#f44', ...messages);
    LogGL.LOG_TO_SCREEN = true;
  }
}
