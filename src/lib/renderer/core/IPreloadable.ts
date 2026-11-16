export default interface IPreloadable {
  load(callback: () => any, onProgress?: (progress: number) => void): void;
}
