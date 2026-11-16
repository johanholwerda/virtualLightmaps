import Vector3 from "monks-webgpu/renderer/math/Vector3";
import ParamGroup from "monks-webgpu/utils/params/ParamGroup";
import ParamBool from "monks-webgpu/utils/params/ParamBool";
import ParamFloat from "monks-webgpu/utils/params/ParamFloat";
import ParamColor from "monks-webgpu/utils/params/ParamColor";
import ParamBase, { type ParamMaterial } from "monks-webgpu/utils/params/ParamBase";
import Scene from "./Scene";

//the sun at its highest, the sunDirection the scene was made for (see the cubes in Scene.createBuilding)
const NOON = new Vector3(.75, 1, .5).normalize();
//on the horizon, at right angles to NOON. The sun goes around the axis between the two
const SUNRISE = new Vector3(.5, 0, -.75).normalize();
//the angle of the sun on its circle at the start and the end of the day, just above the horizon
const DAY_START_ANGLE = .12;
//the part of the loop the sun is up, the rest is a short night back to the morning
const DAY_FRACTION = 50 / 60;

const NIGHT_SKY = new Vector3(.02, .03, .08);
//the sky around sunrise and sunset, the warm light of the sky low near the sun
const GOLDEN_SKY = new Vector3(1, .65, .5);
//how much more air the sunlight goes through per channel, see getSunTint
const EXTINCTION = new Vector3(.04, .12, .3);

const smoothstep = (edge0:number, edge1:number, x:number) => {
  const t = Math.min(Math.max((x - edge0) / (edge1 - edge0), 0), 1);
  return t * t * (3 - 2 * t);
};

//the air the light travels through, 1 straight up. Clamped at the horizon
const getAirMass = (elevation:number) => 1 / (Math.max(elevation, 0) + .05);

/*
Moves the sun from the morning through the afternoon to the evening, then quickly under the ground back to the morning.
Low suns are dimmer and redder, the sky goes from a golden color at the horizon to its full blue at noon, and is dark at
night. Off, it is noon: the sun and the sky the scene has without the cycle, so the test images stay comparable.
The caches follow the light with their own delay, groundTruth restarts every frame
 */
export default class DayCycle {
  private scene: Scene;
  //the materials of the path tracers, which have directionalColor
  private materials: ParamMaterial[];
  private directionalColor: ParamColor;
  public enabled!: ParamBool;
  private duration!: ParamFloat;
  //the sky color at noon
  private daySky: Vector3;
  //seconds into the loop, 0 is the morning
  private time = 0;
  private lastFrameTime: number | null = null;
  private sunDirection = new Vector3();
  private skyColor = new Vector3();
  private sunColor = new Vector3();

  constructor(scene:Scene, materials:ParamMaterial[], directionalColor:ParamColor) {
    this.scene = scene;
    this.materials = materials.filter(material => material.hasUniform('directionalColor'));
    this.directionalColor = directionalColor;
    this.daySky = scene.skyColor.clone();
  }

  public init(paramGroup:ParamGroup) {
    //off by default: the noon sun the scene starts with
    this.enabled = paramGroup.addParamBool('dayCycle', false, { callback: () => {
      this.lastFrameTime = null;
      if(!this.enabled.value) this.apply(Math.PI / 2);
    } });
    //seconds for a whole day and night
    this.duration = paramGroup.addParamFloat('dayCycleDuration', 60, 10, 300);
  }

  //the sky color at noon, from the skyColor and skyBrightness params. With the cycle off it shows right away,
  //with the cycle on the next update blends it in
  public setDaySky(color:Vector3) {
    this.daySky.copy(color);
    if(!this.enabled.value) this.apply(Math.PI / 2);
  }

  public update() {
    if(!this.enabled.value) return;
    const now = performance.now() / 1000;
    //not more than a frame of .1 s, so a hidden tab doesn't jump
    if(this.lastFrameTime !== null) this.time += Math.min(now - this.lastFrameTime, .1);
    this.lastFrameTime = now;

    const duration = this.duration.value;
    this.time %= duration;
    this.apply(this.getSunAngle(this.time / duration));
    //groundTruth converges for a fixed light only, see GroundTruth.update
    ParamBase.changeCount++;
  }

  //the angle of the sun on its circle, 0 is on the horizon in the morning, PI/2 is noon. t is the part of the loop
  private getSunAngle(t:number): number {
    const dayStart = DAY_START_ANGLE, dayEnd = Math.PI - DAY_START_ANGLE;
    if(t < DAY_FRACTION) return dayStart + (dayEnd - dayStart) * t / DAY_FRACTION;

    //the night: on to the next morning, eased so it leaves and joins the day at the speed of the day
    const s = (t - DAY_FRACTION) / (1 - DAY_FRACTION);
    const p0 = dayEnd, p1 = 2 * Math.PI + dayStart;
    const tangent = (dayEnd - dayStart) / DAY_FRACTION * (1 - DAY_FRACTION);
    const s2 = s * s, s3 = s2 * s;
    return (2 * s3 - 3 * s2 + 1) * p0 + (s3 - 2 * s2 + s) * tangent + (-2 * s3 + 3 * s2) * p1 + (s3 - s2) * tangent;
  }

  private apply(angle:number) {
    const direction = this.sunDirection.copy(SUNRISE).multiplyScalar(Math.cos(angle));
    direction.x += NOON.x * Math.sin(angle);
    direction.y += NOON.y * Math.sin(angle);
    direction.z += NOON.z * Math.sin(angle);
    const elevation = direction.y;

    const sky = this.skyColor.copy(NIGHT_SKY)
      .lerp(GOLDEN_SKY, smoothstep(-.2, .03, elevation))
      .lerp(this.daySky, smoothstep(.03, .45, elevation));
    this.scene.setSun(direction, sky);

    const sunColor = this.getSunTint(elevation).multiply(this.directionalColor.value);
    for(const material of this.materials) material.setVector3('directionalColor', sunColor);
  }

  //1 at noon, so the directionalColor param is the color of the noon sun. Lower the light goes through more air, which
  //takes out more blue than red. Under the horizon the ground is in the way
  private getSunTint(elevation:number): Vector3 {
    const extraAir = getAirMass(elevation) - getAirMass(NOON.y);
    const fade = smoothstep(-.01, .04, elevation);
    return this.sunColor.setValues(
      Math.exp(-EXTINCTION.x * extraAir) * fade,
      Math.exp(-EXTINCTION.y * extraAir) * fade,
      Math.exp(-EXTINCTION.z * extraAir) * fade
    );
  }
}
