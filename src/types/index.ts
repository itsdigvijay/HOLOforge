export type ManagerStatus = 'idle' | 'ready' | 'running' | 'error';

export interface TrackingFrame {
  timestampMs: number;
}

export interface GestureResult {
  id: string;
  confidence: number;
}

export interface PowerDefinition {
  id: string;
  label: string;
}

export interface EffectContext {
  canvas: HTMLCanvasElement;
}
