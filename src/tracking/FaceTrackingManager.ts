import type { MediaPipeDelegate } from './handTracking.worker.types';
import type {
  FaceTrackingWorkerResponse,
  FaceWorkerResultMessage,
} from './faceTracking.worker.types';
import {
  subscribeToVideoFrames,
  type VideoFrameCapture,
  type VideoFrameSubscription,
} from './VideoFrameSource';
export type FaceTrackingStatus = 'idle' | 'loading' | 'online' | 'error';

export interface FacePoint {
  /** Normalized coordinates in the unmirrored camera image. */
  x: number;
  y: number;
  z: number;
}

export interface HeadRotation {
  /** Rotation around the horizontal axis, in degrees. */
  pitch: number;
  /** Rotation around the vertical axis, in degrees. */
  yaw: number;
  /** Rotation around the camera axis, in degrees. */
  roll: number;
}

export interface FacingDirection {
  horizontal: 'left' | 'center' | 'right';
  vertical: 'up' | 'center' | 'down';
  label: string;
}

export interface FaceTrackingSnapshot {
  status: FaceTrackingStatus;
  faceDetected: boolean;
  /** Interleaved x/y/z values for the complete MediaPipe mesh when requested. */
  landmarks: Float32Array;
  landmarkCount: number;
  leftEyeCenter: FacePoint | null;
  rightEyeCenter: FacePoint | null;
  leftTemple: FacePoint | null;
  rightTemple: FacePoint | null;
  forehead: FacePoint | null;
  faceCenter: FacePoint | null;
  headRotation: HeadRotation | null;
  facingDirection: FacingDirection | null;
  /** MediaPipe does not always expose per-face confidence for this task. */
  confidence: number | null;
  fps: number;
  inferenceTimeMs: number;
  trackingLatencyMs: number;
  targetInferenceFps: number;
  delegate: MediaPipeDelegate | null;
  usingVideoFrameCallback: boolean;
  timestamp: number;
}

const WASM_PATH = '/mediapipe/wasm';
const MODEL_PATH = '/models/face_landmarker.task';
const LANDMARK_STRIDE = 3;
const LEFT_IRIS_CENTER = 0;
const RIGHT_IRIS_CENTER = 1;
const LEFT_TEMPLE = 2;
const RIGHT_TEMPLE = 3;
const FOREHEAD = 4;
const CHIN = 5;
const NOSE_TIP = 6;
const EMPTY_LANDMARKS = new Float32Array(0);

const EMPTY_SNAPSHOT: FaceTrackingSnapshot = {
  status: 'idle',
  faceDetected: false,
  landmarks: EMPTY_LANDMARKS,
  landmarkCount: 0,
  leftEyeCenter: null,
  rightEyeCenter: null,
  leftTemple: null,
  rightTemple: null,
  forehead: null,
  faceCenter: null,
  headRotation: null,
  facingDirection: null,
  confidence: null,
  fps: 0,
  inferenceTimeMs: 0,
  trackingLatencyMs: 0,
  targetInferenceFps: 15,
  delegate: null,
  usingVideoFrameCallback: false,
  timestamp: 0,
};

const toDegrees = (radians: number) => (radians * 180) / Math.PI;
const clamp = (value: number, min: number, max: number) =>
  Math.min(max, Math.max(min, value));

function extractPointFromLandmark(
  landmarks: Float32Array,
  index: number,
): FacePoint {
  const offset = index * LANDMARK_STRIDE;
  return {
    x: landmarks[offset] ?? 0,
    y: landmarks[offset + 1] ?? 0,
    z: landmarks[offset + 2] ?? 0,
  };
}

function rotationFromMatrix(matrix: Float32Array): HeadRotation | null {
  if (matrix.length < 16) return null;
  const m00 = matrix[0];
  const m10 = matrix[4];
  const m20 = matrix[8];
  const m21 = matrix[9];
  const m22 = matrix[10];
  const horizontalScale = Math.hypot(m00, m10);
  return {
    pitch: toDegrees(Math.atan2(m21, m22)),
    yaw: toDegrees(Math.atan2(-m20, horizontalScale)),
    roll: toDegrees(Math.atan2(m10, m00)),
  };
}

function rotationFromLandmarks(
  landmarks: Float32Array,
  leftEye: FacePoint,
  rightEye: FacePoint,
  forehead: FacePoint,
  chin: FacePoint,
): HeadRotation {
  const eyeDistance = Math.max(
    0.001,
    Math.hypot(leftEye.x - rightEye.x, leftEye.y - rightEye.y),
  );
  const eyeMidX = (leftEye.x + rightEye.x) * 0.5;
  const noseOffset = NOSE_TIP * LANDMARK_STRIDE;
  const noseX = landmarks[noseOffset];
  const noseY = landmarks[noseOffset + 1];
  const faceHeight = Math.max(0.001, chin.y - forehead.y);
  const expectedNoseY = forehead.y + faceHeight * 0.55;
  return {
    pitch: clamp(((noseY - expectedNoseY) / faceHeight) * 70, -45, 45),
    yaw: clamp(((noseX - eyeMidX) / eyeDistance) * 55, -60, 60),
    roll: toDegrees(
      Math.atan2(leftEye.y - rightEye.y, leftEye.x - rightEye.x),
    ),
  };
}

function facingFromRotation(rotation: HeadRotation): FacingDirection {
  const horizontal =
    rotation.yaw > 12
      ? 'left'
      : rotation.yaw < -12
        ? 'right'
        : 'center';
  const vertical =
    rotation.pitch > 10
      ? 'down'
      : rotation.pitch < -10
        ? 'up'
        : 'center';
  const label =
    horizontal === 'center' && vertical === 'center'
      ? 'forward'
      : horizontal === 'center'
        ? vertical
        : vertical === 'center'
          ? horizontal
          : `${vertical}-${horizontal}`;
  return { horizontal, vertical, label };
}

export class FaceTrackingManager {
  private worker: Worker | null = null;
  private workerInitialization: Promise<MediaPipeDelegate> | null = null;
  private resolveWorkerInitialization:
    | ((delegate: MediaPipeDelegate) => void)
    | null = null;
  private rejectWorkerInitialization: ((error: Error) => void) | null = null;
  private activeDelegate: MediaPipeDelegate | null = null;
  private video: HTMLVideoElement | null = null;
  private frameSubscription: VideoFrameSubscription | null = null;
  private latestSnapshot: FaceTrackingSnapshot = EMPTY_SNAPSHOT;
  private runToken = 0;
  private disposed = false;
  private inferenceInProgress = false;
  private lastVideoTime = -1;
  private lastInferenceStartedAt = 0;
  private targetInferenceFps = 15;
  private landmarkVisualizationEnabled = false;
  private fpsWindowStartedAt = 0;
  private framesInWindow = 0;
  private measuredFps = 0;
  private averagedInferenceTimeMs = 0;
  private averagedTrackingLatencyMs = 0;
  private readonly statusListeners = new Set<
    (status: FaceTrackingStatus) => void
  >();

  get snapshot(): FaceTrackingSnapshot {
    return this.latestSnapshot;
  }

  subscribeStatus(listener: (status: FaceTrackingStatus) => void): () => void {
    this.statusListeners.add(listener);
    listener(this.latestSnapshot.status);
    return () => this.statusListeners.delete(listener);
  }

  setLandmarkVisualizationEnabled(enabled: boolean): void {
    if (this.landmarkVisualizationEnabled === enabled) return;
    this.landmarkVisualizationEnabled = enabled;
    if (!enabled && this.latestSnapshot.landmarks.length > 0) {
      this.latestSnapshot = {
        ...this.latestSnapshot,
        landmarks: EMPTY_LANDMARKS,
        landmarkCount: 0,
      };
    }
    this.worker?.postMessage({ type: 'setLandmarkOutput', enabled });
  }

  async start(video: HTMLVideoElement): Promise<boolean> {
    const token = ++this.runToken;
    this.stopFrameLoop();
    this.video = null;
    this.resetMetrics();
    this.setStatus('loading', true);

    try {
      await this.getOrCreateWorker();
      if (this.disposed || token !== this.runToken) return false;
      this.video = video;
      this.setStatus('online', true);
      this.startFrameLoop();
      return true;
    } catch (error) {
      if (!this.disposed && token === this.runToken) {
        this.setStatus('error', true);
      }
      throw error;
    }
  }

  stop(): void {
    this.runToken += 1;
    this.stopFrameLoop();
    this.video = null;
    this.resetMetrics();
    this.setStatus('idle', true);
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.stop();
    this.terminateWorker();
    this.activeDelegate = null;
    this.statusListeners.clear();
    if (this.worker) {
      this.worker.removeEventListener('message', this.handleWorkerMessage);
      this.worker.removeEventListener('error', this.handleWorkerRuntimeError);
      this.worker.postMessage({ type: 'dispose' });
      this.worker.terminate();
      this.worker = null;
    }
    this.workerInitialization = null;
    this.resolveWorkerInitialization = null;
    this.rejectWorkerInitialization = null;
    this.activeDelegate = null;
    this.statusListeners.clear();
  }

  private getOrCreateWorker(): Promise<MediaPipeDelegate> {
    if (this.activeDelegate) return Promise.resolve(this.activeDelegate);
    if (this.workerInitialization) return this.workerInitialization;

    this.worker = new Worker(
      new URL('./faceTracking.worker.ts', import.meta.url),
      { type: 'module' },
    );
    this.worker.addEventListener('message', this.handleWorkerMessage);
    this.worker.addEventListener('error', this.handleWorkerRuntimeError);
    this.workerInitialization = new Promise<MediaPipeDelegate>(
      (resolve, reject) => {
        this.resolveWorkerInitialization = resolve;
        this.rejectWorkerInitialization = reject;
      },
    );
    this.worker.postMessage({
      type: 'init',
      wasmPath: WASM_PATH,
      modelPath: MODEL_PATH,
      outputLandmarks: this.landmarkVisualizationEnabled,
    });
    return this.workerInitialization;
  }

  private startFrameLoop(): void {
    const video = this.video;
    if (!video) return;
    this.frameSubscription = subscribeToVideoFrames(
      video,
      this.processVideoFrame,
    );
    this.latestSnapshot = {
      ...this.latestSnapshot,
      usingVideoFrameCallback:
        this.frameSubscription.usingVideoFrameCallback,
    };
  }

  private readonly processVideoFrame = (
    now: number,
    mediaTime: number,
    capture: VideoFrameCapture,
  ) => {
    const video = this.video;
    if (!video || !this.worker || this.disposed) return;
    this.tryInference(video, now, mediaTime, capture);
  };

  private tryInference(
    video: HTMLVideoElement,
    now: number,
    videoTime: number,
    capture: VideoFrameCapture,
  ): void {
    if (
      video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA ||
      videoTime === this.lastVideoTime ||
      this.inferenceInProgress ||
      now - this.lastInferenceStartedAt < 1000 / this.targetInferenceFps
    ) {
      return;
    }

    this.lastVideoTime = videoTime;
    this.lastInferenceStartedAt = now;
    this.inferenceInProgress = true;
    void this.sendLatestFrame(video, capture, now, this.runToken);
  }

  private async sendLatestFrame(
    video: HTMLVideoElement,
    capture: VideoFrameCapture,
    timestamp: number,
    runToken: number,
  ): Promise<void> {
    try {
      const frame = await capture();
      if (
        this.disposed ||
        runToken !== this.runToken ||
        !this.worker ||
        this.video !== video
      ) {
        frame.close();
        if (runToken === this.runToken) this.inferenceInProgress = false;
        return;
      }
      this.worker.postMessage(
        { type: 'frame', frame, timestamp, runToken },
        [frame],
      );
    } catch {
      if (runToken === this.runToken) this.inferenceInProgress = false;
    }
  }

  private readonly handleWorkerMessage = (
    event: MessageEvent<FaceTrackingWorkerResponse>,
  ) => {
    const message = event.data;
    if (message.type === 'ready') {
      this.activeDelegate = message.delegate;
      this.targetInferenceFps = message.delegate === 'GPU' ? 15 : 8;
      this.latestSnapshot = {
        ...this.latestSnapshot,
        delegate: message.delegate,
        targetInferenceFps: this.targetInferenceFps,
      };
      this.resolveWorkerInitialization?.(message.delegate);
      this.resolveWorkerInitialization = null;
      this.rejectWorkerInitialization = null;
      return;
    }

    if (message.type === 'result') {
      if (message.runToken !== this.runToken) return;
      this.inferenceInProgress = false;
      this.updateSnapshot(message);
      return;
    }

    const errorMsg = 'message' in message && typeof message.message === 'string'
      ? message.message
      : 'Unknown face tracking worker error';
    const error = new Error(errorMsg);
    console.error('[HOLOFORGE] Face tracking worker:', error);
    if (!this.activeDelegate) {
      this.rejectWorkerInitialization?.(error);
      this.terminateWorker();
      return;
    }
    if (message.runToken === undefined || message.runToken === this.runToken) {
      this.inferenceInProgress = false;
      this.stopFrameLoop();
      this.video = null;
      this.setStatus('error');
      this.clearFace();
    }
  };

  private readonly handleWorkerRuntimeError = (event: ErrorEvent) => {
    const error = new Error(event.message || 'Face tracking worker failed.');
    console.error('[HOLOFORGE] Face worker runtime error:', error);
    this.rejectWorkerInitialization?.(error);
    this.terminateWorker();
    this.activeDelegate = null;
    this.inferenceInProgress = false;
    this.stopFrameLoop();
    this.video = null;
    this.setStatus('error');
    this.clearFace();
  };

  private terminateWorker(): void {
    if (this.worker) {
      this.worker.removeEventListener('message', this.handleWorkerMessage);
      this.worker.removeEventListener('error', this.handleWorkerRuntimeError);
      this.worker.terminate();
      this.worker = null;
    }
    this.workerInitialization = null;
    this.resolveWorkerInitialization = null;
    this.rejectWorkerInitialization = null;
  }

  private updateSnapshot(result: FaceWorkerResultMessage): void {
    const completedAt = performance.now();
    this.averagedInferenceTimeMs =
      this.averagedInferenceTimeMs === 0
        ? result.inferenceTimeMs
        : this.averagedInferenceTimeMs * 0.8 + result.inferenceTimeMs * 0.2;
    const latency = Math.max(0, completedAt - result.timestamp);
    this.averagedTrackingLatencyMs =
      this.averagedTrackingLatencyMs === 0
        ? latency
        : this.averagedTrackingLatencyMs * 0.8 + latency * 0.2;

    if (this.fpsWindowStartedAt === 0) {
      this.fpsWindowStartedAt = result.timestamp;
      this.framesInWindow = 0;
    } else {
      this.framesInWindow += 1;
      const elapsed = result.timestamp - this.fpsWindowStartedAt;
      if (elapsed >= 500) {
        this.measuredFps = (this.framesInWindow * 1000) / elapsed;
        this.framesInWindow = 0;
        this.fpsWindowStartedAt = result.timestamp;
      }
    }

    if (!result.faceDetected || result.keyLandmarks.length < 7 * LANDMARK_STRIDE) {
      this.clearFace(result.timestamp);
      return;
    }

    const leftEyeCenter = extractPointFromLandmark(result.keyLandmarks, LEFT_IRIS_CENTER);
    const rightEyeCenter = extractPointFromLandmark(result.keyLandmarks, RIGHT_IRIS_CENTER);
    const leftTemple = extractPointFromLandmark(result.keyLandmarks, LEFT_TEMPLE);
    const rightTemple = extractPointFromLandmark(result.keyLandmarks, RIGHT_TEMPLE);
    const forehead = extractPointFromLandmark(result.keyLandmarks, FOREHEAD);
    const chin = extractPointFromLandmark(result.keyLandmarks, CHIN);
    
    const faceCenter: FacePoint = {
      x: (leftTemple.x + rightTemple.x) * 0.5,
      y: (forehead.y + chin.y) * 0.5,
      z: (leftTemple.z + rightTemple.z + forehead.z + chin.z) * 0.25,
    };

    const rotation =
      rotationFromMatrix(result.transformationMatrix) ??
      rotationFromLandmarks(
        result.keyLandmarks,
        leftEyeCenter,
        rightEyeCenter,
        forehead,
        chin,
      );

    this.latestSnapshot = {
      status: 'online',
      faceDetected: true,
      landmarks: this.landmarkVisualizationEnabled
        ? result.landmarks
        : EMPTY_LANDMARKS,
      landmarkCount: this.landmarkVisualizationEnabled
        ? result.landmarkCount
        : 0,
      leftEyeCenter,
      rightEyeCenter,
      leftTemple,
      rightTemple,
      forehead,
      faceCenter,
      headRotation: rotation,
      facingDirection: facingFromRotation(rotation),
      confidence: result.confidence >= 0 ? result.confidence : null,
      fps: this.measuredFps,
      inferenceTimeMs: this.averagedInferenceTimeMs,
      trackingLatencyMs: this.averagedTrackingLatencyMs,
      targetInferenceFps: this.targetInferenceFps,
      delegate: this.activeDelegate,
      usingVideoFrameCallback:
        this.frameSubscription?.usingVideoFrameCallback ?? false,
      timestamp: result.timestamp,
    };
  }

  private clearFace(timestamp = 0): void {
    this.latestSnapshot = {
      ...this.latestSnapshot,
      status: this.latestSnapshot.status,
      faceDetected: false,
      landmarks: EMPTY_LANDMARKS,
      landmarkCount: 0,
      leftEyeCenter: null,
      rightEyeCenter: null,
      leftTemple: null,
      rightTemple: null,
      forehead: null,
      faceCenter: null,
      headRotation: null,
      facingDirection: null,
      confidence: null,
      fps: this.measuredFps,
      inferenceTimeMs: this.averagedInferenceTimeMs,
      trackingLatencyMs: this.averagedTrackingLatencyMs,
      timestamp,
    };
  }

  private stopFrameLoop(): void {
    this.frameSubscription?.unsubscribe();
    this.frameSubscription = null;
  }

  private resetMetrics(): void {
    this.lastVideoTime = -1;
    this.lastInferenceStartedAt = 0;
    this.fpsWindowStartedAt = 0;
    this.framesInWindow = 0;
    this.measuredFps = 0;
    this.averagedInferenceTimeMs = 0;
    this.averagedTrackingLatencyMs = 0;
    this.inferenceInProgress = false;
  }

  private setStatus(status: FaceTrackingStatus, reset = false): void {
    const previousStatus = this.latestSnapshot.status;
    this.latestSnapshot = reset
      ? {
          ...EMPTY_SNAPSHOT,
          status,
          delegate: this.activeDelegate,
          targetInferenceFps: this.targetInferenceFps,
        }
      : { ...this.latestSnapshot, status };
    if (previousStatus !== status) {
      this.statusListeners.forEach((listener) => listener(status));
    }
  }
}
