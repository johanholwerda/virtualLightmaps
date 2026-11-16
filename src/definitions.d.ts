declare module '*.txt?raw' {
  const exp: string;
  export default exp;
}

declare module '*.glsl?raw' {
  const exp: string;
  export default exp;
}

/* declare module '*.wgsl?raw' {
  const exp: string;
  export default exp;
} */

declare module "*.wgsl?raw" {
  const content: string;
  export default content;
}

declare module '*?url' {
  const exp: string;
  export default exp;
}

declare module '*.png' {
  const exp: string;
  export default exp;
}

declare module '*.webp' {
  const exp: string;
  export default exp;
}


declare module '*.jpg' {
  const exp: string;
  export default exp;
}

declare module '*.avif' {
  const exp: string;
  export default exp;
}


declare module '*.wad' {
  const exp: string;
  export default exp;
}

declare module '*.mp3' {
  const exp: string;
  export default exp;
}

declare module '*.wav' {
  const exp: string;
  export default exp;
}

declare module '*.json' {
  const exp: Array<any> | { [key: string]: any };
  export default exp;
}

declare module '*.txt' {
  const exp: string;
  export default exp;
}

declare module 'file-loader!*' {
  const exp: string;
  export default exp;
}

declare module '*.glsl' {
  const exp: string;
  export default exp;
}

declare module '*.wgsl' {
  const exp: string;
  export default exp;
}

declare module '*.fs' {
  const exp: string;
  export default exp;
}

declare module '*.vs' {
  const exp: string;
  export default exp;
}

declare module '*.dds' {
  const exp: string;
  export default exp;
}

declare module '*.obj' {
  const exp: string;
  export default exp;
}

declare module '*.mp4' {
  const exp: string;
  export default exp;
}

declare module '*.hdr' {
  const exp: string;
  export default exp;
}

declare module '*.mtl' {
  const exp: string;
  export default exp;
}

declare module 'single-example';

interface Window {
  AudioContext: typeof AudioContext;
  webkitAudioContext: typeof AudioContext;
}

interface ImportMeta {
  glob?: any;
}
declare module '*.svg?raw' {
  const content: string;
  export default content;
}
