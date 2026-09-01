import type { MediaPipeDelegate } from './handTracking.worker.types';

export interface FaceWorkerInitMessage {
  type: 'init';
  wasmPath: string;
  modelPath: string;
  outputLandmarks: boolean;
}

export interface FaceWorkerFrameMessage {
  type: 'frame';
  frame: ImageBitmap | VideoFrame;
  timestamp: number;
  runToken: number;
}

export interface FaceWorkerDisposeMessage {
  type: 'dispose';
}

export interface FaceWorkerLandmarkOutputMessage {
  type: 'setLandmarkOutput';
  enabled: boolean;
}

export type FaceTrackingWorkerRequest =
  | FaceWorkerInitMessage
  | FaceWorkerFrameMessage
  | FaceWorkerLandmarkOutputMessage
  | FaceWorkerDisposeMessage;

export interface FaceWorkerReadyMessage {
  type: 'ready';
  delegate: MediaPipeDelegate;
}

export interface FaceWorkerResultMessage {
  type: 'result';
  runToken: number;
  timestamp: number;
  inferenceTimeMs: number;
  faceDetected: boolean;
  /** Seven x/y/z points used to derive the public tracking anchors. */
  keyLandmarks: Float32Array;
  landmarkCount: number;
  /** Complete x/y/z mesh, only populated while visualization is enabled. */
  landmarks: Float32Array;
  transformationMatrix: Float32Array;
  confidence: number;
}

export interface FaceWorkerErrorMessage {
  type: 'error';
  message: string;
  runToken?: number;
}

export type FaceTrackingWorkerResponse =
  | FaceWorkerReadyMessage
  | FaceWorkerResultMessage
  | FaceWorkerErrorMessage;
