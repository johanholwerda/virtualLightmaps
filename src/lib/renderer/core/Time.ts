import {clamp, lerp} from '../utils/MathUtils';

export default class Time {
  public frameNumber: number = 0;
  public time: number = 0;
  public fps: number = 60;
  public dt: number = 1 / this.fps;
  public avgFps: number = 60;
  public minFps: number = 1e10;
  public maxFps: number = 0;

  public update(time: number) {
    time *= 0.001;
    //the real frame time on screens up to 1000 hz. the minimum keeps two updates with the same time from dividing by 0,
    //the maximum keeps a stall (a hidden tab, a breakpoint) from becoming one huge step
    this.dt = clamp(time - this.time, 1 / 1000, 1 / 30);

    this.time = time;
    this.fps = 1 / this.dt;
    this.minFps = Math.min(this.minFps, this.fps);
    this.maxFps = Math.max(this.maxFps, this.fps);

    this.avgFps = lerp(this.fps, this.avgFps, 0.95);
    this.frameNumber++;
  }
}
