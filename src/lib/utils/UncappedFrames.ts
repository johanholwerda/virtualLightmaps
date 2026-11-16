/*
 Lets requestAnimationFrame run without vsync, to see how many fps an example could reach. install() patches
 requestAnimationFrame for the whole page, so neither the renderer nor the examples change; setEnabled() switches
 between the native rAF and the uncapped loop (the checkbox next to the fps counter, or ?uncapped in the url).

 When uncapped, a frame starts as soon as the gpu has finished the frame before the previous one, so one frame can be
 in flight while the cpu encodes the next (like a normal rAF loop), but frames never pile up. All callbacks requested
 during a frame run together in the next one, like real rAF, so the stats panel counts frames and not callbacks.
 The gpu stays at full load, so it doesn't clock down the way it does when vsync leaves it idle.

 The device is captured by wrapping GPUAdapter.requestDevice, so install() has to run before the renderer is created.
 Until a device exists the native rAF is used. The canvas still only shows a frame per vsync; the others are rendered
 and dropped. Uncapped frames keep running in a hidden tab, where rAF stops.
 */

type FrameCallback = (time: number) => void;

export default class UncappedFrames {
  private static installed = false;
  private static enabled = false;
  private static device: GPUDevice | null = null;

  private static nativeRequest: (callback: FrameRequestCallback) => number;
  private static nativeCancel: (id: number) => void;

  private static nextId = 1;
  private static pending = new Map<number, FrameCallback>();
  //callbacks that were handed to the native rAF when uncapped was turned off, so their (negative) ids still cancel
  private static movedToNative = new Map<number, number>();
  private static scheduled = false;
  private static previousFrameDone: Promise<unknown> = Promise.resolve();
  private static channel: MessageChannel;

  public static install(enabled: boolean) {
    if (UncappedFrames.installed) return;
    UncappedFrames.installed = true;
    UncappedFrames.enabled = enabled;

    UncappedFrames.nativeRequest = window.requestAnimationFrame.bind(window);
    UncappedFrames.nativeCancel = window.cancelAnimationFrame.bind(window);

    const requestDevice = GPUAdapter.prototype.requestDevice;
    GPUAdapter.prototype.requestDevice = async function (this: GPUAdapter, descriptor?: GPUDeviceDescriptor) {
      const device = await requestDevice.call(this, descriptor);
      UncappedFrames.device = device;
      return device;
    };

    //a macrotask, so input, timers and rendering still get a turn between frames
    UncappedFrames.channel = new MessageChannel();
    UncappedFrames.channel.port1.onmessage = () => UncappedFrames.runFrame();

    window.requestAnimationFrame = (callback: FrameRequestCallback) => UncappedFrames.request(callback);
    window.cancelAnimationFrame = (id: number) => UncappedFrames.cancel(id);
  }

  public static isInstalled(): boolean {
    return UncappedFrames.installed;
  }

  public static isEnabled(): boolean {
    return UncappedFrames.enabled;
  }

  public static setEnabled(enabled: boolean) {
    if (!UncappedFrames.installed || enabled === UncappedFrames.enabled) return;
    UncappedFrames.enabled = enabled;
    if (!enabled) {
      //the loops waiting for an uncapped frame continue on the native rAF
      for (const [id, callback] of UncappedFrames.pending) {
        UncappedFrames.movedToNative.set(id, UncappedFrames.nativeRequest((time) => {
          UncappedFrames.movedToNative.delete(id);
          callback(time);
        }));
      }
      UncappedFrames.pending.clear();
    }
    console.log(`UncappedFrames: requestAnimationFrame is ${enabled ? 'not ' : ''}tied to vsync`);
  }

  private static request(callback: FrameRequestCallback): number {
    if (!UncappedFrames.enabled || !UncappedFrames.device) return UncappedFrames.nativeRequest(callback);
    const id = UncappedFrames.nextId++;
    UncappedFrames.pending.set(id, callback);
    UncappedFrames.schedule();
    return -id; //negative, so cancel can tell it from a native id
  }

  private static cancel(id: number) {
    if (id >= 0) {
      UncappedFrames.nativeCancel(id);
      return;
    }
    UncappedFrames.pending.delete(-id);
    const nativeId = UncappedFrames.movedToNative.get(-id);
    if (nativeId !== undefined) {
      UncappedFrames.nativeCancel(nativeId);
      UncappedFrames.movedToNative.delete(-id);
    }
  }

  private static schedule() {
    if (UncappedFrames.scheduled) return;
    UncappedFrames.scheduled = true;
    const post = () => UncappedFrames.channel.port2.postMessage(0);
    UncappedFrames.previousFrameDone.then(post, post);
  }

  private static runFrame() {
    UncappedFrames.scheduled = false;
    const callbacks = UncappedFrames.pending;
    UncappedFrames.pending = new Map();

    //this frame waits for the one before the previous one; requests made by the callbacks wait for the previous one
    const waitFor = UncappedFrames.previousFrameDone;
    UncappedFrames.scheduled = true;

    const time = performance.now();
    for (const callback of callbacks.values()) {
      try {
        callback(time);
      } catch (e) {
        console.error(e);
      }
    }

    //this frame's work is submitted now
    UncappedFrames.previousFrameDone = UncappedFrames.device ? UncappedFrames.device.queue.onSubmittedWorkDone() : Promise.resolve();
    UncappedFrames.scheduled = false;
    if (UncappedFrames.pending.size > 0) {
      UncappedFrames.scheduled = true;
      const post = () => UncappedFrames.channel.port2.postMessage(0);
      waitFor.then(post, post);
    }
  }
}
