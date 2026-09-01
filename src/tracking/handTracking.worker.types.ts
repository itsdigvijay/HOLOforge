export type MediaPipeDelegate = 'GPU' | 'CPU';

export interface MediaPipeTrackingConfig {
  minHandDetectionConfidence: number;
  minHandPresenceConfidence: number;
  minTrackingConfidence: number;
}

export interface WorkerInitMessage {
  type: 'init';
  wasmPath: string;
  modelPath: string;
  config: MediaPipeTrackingConfig;
}

export interface WorkerFrameMessage {
  type: 'frame';
  frame: ImageBitmap | VideoFrame;
  timestamp: number;
  runToken: number;
}

export interface WorkerDisposeMessage {
  type: 'dispose';
}

export interface WorkerConfigureMessage {
  type: 'configure';
  revision: number;
  config: MediaPipeTrackingConfig;
}

export type HandTrackingWorkerRequest =
  | WorkerInitMessage
  | WorkerFrameMessage
  | WorkerConfigureMessage
  | WorkerDisposeMessage;

export interface WorkerReadyMessage {
  type: 'ready';
  delegate: MediaPipeDelegate;
}

export interface WorkerResultMessage {
  type: 'result';
  runToken: number;
  timestamp: number;
  inferenceTimeMs: number;
  handCount: number;
  landmarks: Float32Array;
  handedness: Array<{ categoryName: string; score: number }>;
  motionBlurScore: number;
}

export interface WorkerConfiguredMessage {
  type: 'configured';
  revision: number;
}

export interface WorkerErrorMessage {
  type: 'error';
  message: string;
  runToken?: number;
  revision?: number;
}

export type HandTrackingWorkerResponse =
  | WorkerReadyMessage
  | WorkerResultMessage
  | WorkerConfiguredMessage
  | WorkerErrorMessage;
