import type { MediaPipeDelegate } from './handTracking.worker.types';

export interface PoseWorkerInitMessage {
  type: 'init';
  wasmPath: string;
  modelPath: string;
  outputLandmarks: boolean;
}

export interface PoseWorkerFrameMessage {
  type: 'frame';
  frame: ImageBitmap | VideoFrame;
  timestamp: number;
  runToken: number;
}

export interface PoseWorkerLandmarkOutputMessage {
  type: 'setLandmarkOutput';
  enabled: boolean;
}

export interface PoseWorkerDisposeMessage {
  type: 'dispose';
}

export type PoseTrackingWorkerRequest =
  | PoseWorkerInitMessage
  | PoseWorkerFrameMessage
  | PoseWorkerLandmarkOutputMessage
  | PoseWorkerDisposeMessage;

export interface PoseWorkerReadyMessage {
  type: 'ready';
  delegate: MediaPipeDelegate;
}

export interface PoseWorkerResultMessage {
  type: 'result';
  runToken: number;
  timestamp: number;
  inferenceTimeMs: number;
  poseDetected: boolean;
  /** Compact normalized x/y/z/visibility/presence tracking anchors. */
  keyLandmarks: Float32Array;
  /** Matching x/y/z anchors in MediaPipe world coordinates. */
  keyWorldLandmarks: Float32Array;
  /** Complete normalized pose skeleton, populated only for Developer Mode. */
  landmarks: Float32Array;
  landmarkCount: number;
}

export interface PoseWorkerErrorMessage {
  type: 'error';
  message: string;
  runToken?: number;
}

export type PoseTrackingWorkerResponse =
  | PoseWorkerReadyMessage
  | PoseWorkerResultMessage
  | PoseWorkerErrorMessage;
