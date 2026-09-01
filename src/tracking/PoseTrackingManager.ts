import type { MediaPipeDelegate } from './handTracking.worker.types';
import type {
  PoseTrackingWorkerResponse,
  PoseWorkerResultMessage,
} from './poseTracking.worker.types';
import {
  subscribeToVideoFrames,
  type VideoFrameCapture,
  type VideoFrameSubscription,
} from './VideoFrameSource';

export type PoseTrackingStatus = 'idle' | 'loading' | 'online' | 'error';
export type BodyFacingDirection = 'left' | 'forward' | 'right';

export const POSE_LANDMARK_STRIDE = 5;
const WORLD_LANDMARK_STRIDE = 3;
const KEY_LANDMARK_COUNT = 11;
const NOSE = 0;
const LEFT_EAR = 1;
const RIGHT_EAR = 2;
const LEFT_SHOULDER = 3;
const RIGHT_SHOULDER = 4;
const LEFT_ELBOW = 5;
const RIGHT_ELBOW = 6;
const LEFT_WRIST = 7;
const RIGHT_WRIST = 8;
const LEFT_HIP = 9;
const RIGHT_HIP = 10;
const EMPTY_LANDMARKS = new Float32Array(0);
const WASM_PATH = '/mediapipe/wasm';
const MODEL_PATH = '/models/pose_landmarker.task';

export interface PosePoint {
  /** Normalized coordinates in the unmirrored camera image. */
  x: number;
  y: number;
  z: number;
  visibility: number | null;
  presence: number | null;
}

/** Camera-relative orientation suitable for effects and coarse gestures. */
export interface UpperBodyOrientation {
  /** Turn around the vertical body axis in degrees. */
  yaw: number;
  /** Torso pitch from MediaPipe world coordinates in degrees. */
  pitch: number;
  /** Shoulder-line rotation in the image plane in degrees. */
  roll: number;
  /** Side lean in the image plane in degrees. */
  lean: number;
  facing: BodyFacingDirection;
}

export interface PoseTrackingSnapshot {
  status: PoseTrackingStatus;
  poseDetected: boolean;
  head: PosePoint | null;
  leftShoulder: PosePoint | null;
  rightShoulder: PosePoint | null;
  leftElbow: PosePoint | null;
  rightElbow: PosePoint | null;
  leftWrist: PosePoint | null;
  rightWrist: PosePoint | null;
  chestMidpoint: PosePoint | null;
  upperBodyOrientation: UpperBodyOrientation | null;
  /** Average visibility of the exposed upper-body joints. */
  confidence: number | null;
  /** Complete x/y/z/visibility/presence skeleton when visualization is enabled. */
  landmarks: Float32Array;
  landmarkCount: number;
  fps: number;
  inferenceTimeMs: number;
  trackingLatencyMs: number;
  targetInferenceFps: number;
  delegate: MediaPipeDelegate | null;
  usingVideoFrameCallback: boolean;
  timestamp: number;
}

const EMPTY_SNAPSHOT: PoseTrackingSnapshot = {
  status: 'idle',
  poseDetected: false,
  head: null,
  leftShoulder: null,
  rightShoulder: null,
  leftElbow: null,
  rightElbow: null,
  leftWrist: null,
  rightWrist: null,
  chestMidpoint: null,
  upperBodyOrientation: null,
  confidence: null,
  landmarks: EMPTY_LANDMARKS,
  landmarkCount: 0,
  fps: 0,
  inferenceTimeMs: 0,
  trackingLatencyMs: 0,
  targetInferenceFps: 10,
  delegate: null,
  usingVideoFrameCallback: false,
  timestamp: 0,
};

const toDegrees = (radians: number) => (radians * 180) / Math.PI;

function nullableScore(value: number | undefined): number | null {
  return Number.isFinite(value) && (value ?? -1) >= 0 ? (value ?? 0) : null;
}

function setPoint(
  point: PosePoint,
  landmarks: Float32Array,
  index: number,
): void {
  const offset = index * POSE_LANDMARK_STRIDE;
  point.x = landmarks[offset] ?? 0;
  point.y = landmarks[offset + 1] ?? 0;
  point.z = landmarks[offset + 2] ?? 0;
  point.visibility = nullableScore(landmarks[offset + 3]);
  point.presence = nullableScore(landmarks[offset + 4]);
}

function midpoint(target: PosePoint, first: PosePoint, second: PosePoint): void {
  target.x = (first.x + second.x) * 0.5;
  target.y = (first.y + second.y) * 0.5;
  target.z = (first.z + second.z) * 0.5;
  target.visibility =
    first.visibility === null || second.visibility === null
      ? first.visibility ?? second.visibility
      : (first.visibility + second.visibility) * 0.5;
  target.presence =
    first.presence === null || second.presence === null
      ? first.presence ?? second.presence
      : (first.presence + second.presence) * 0.5;
}

function worldPoint(
  landmarks: Float32Array,
  index: number,
): readonly [number, number, number] {
  const offset = index * WORLD_LANDMARK_STRIDE;
  return [
    landmarks[offset] ?? 0,
    landmarks[offset + 1] ?? 0,
    landmarks[offset + 2] ?? 0,
  ];
}

function calculateOrientation(
  normalized: Float32Array,
  world: Float32Array,
): UpperBodyOrientation {
  const leftShoulderOffset = LEFT_SHOULDER * POSE_LANDMARK_STRIDE;
  const rightShoulderOffset = RIGHT_SHOULDER * POSE_LANDMARK_STRIDE;
  const leftHipOffset = LEFT_HIP * POSE_LANDMARK_STRIDE;
  const rightHipOffset = RIGHT_HIP * POSE_LANDMARK_STRIDE;
  const shoulderDx =
    normalized[leftShoulderOffset] - normalized[rightShoulderOffset];
  const shoulderDy =
    normalized[leftShoulderOffset + 1] - normalized[rightShoulderOffset + 1];
  const shoulderMidX =
    (normalized[leftShoulderOffset] + normalized[rightShoulderOffset]) * 0.5;
  const shoulderMidY =
    (normalized[leftShoulderOffset + 1] + normalized[rightShoulderOffset + 1]) * 0.5;
  const hipMidX =
    (normalized[leftHipOffset] + normalized[rightHipOffset]) * 0.5;
  const hipMidY =
    (normalized[leftHipOffset + 1] + normalized[rightHipOffset + 1]) * 0.5;
  const leftShoulderWorld = worldPoint(world, LEFT_SHOULDER);
  const rightShoulderWorld = worldPoint(world, RIGHT_SHOULDER);
  const leftHipWorld = worldPoint(world, LEFT_HIP);
  const rightHipWorld = worldPoint(world, RIGHT_HIP);
  const shoulderWorldX = Math.max(
    0.001,
    Math.abs(leftShoulderWorld[0] - rightShoulderWorld[0]),
  );
  const yaw = toDegrees(
    Math.atan2(
      rightShoulderWorld[2] - leftShoulderWorld[2],
      shoulderWorldX,
    ),
  );
  const shoulderWorldY =
    (leftShoulderWorld[1] + rightShoulderWorld[1]) * 0.5;
  const shoulderWorldZ =
    (leftShoulderWorld[2] + rightShoulderWorld[2]) * 0.5;
  const hipWorldY = (leftHipWorld[1] + rightHipWorld[1]) * 0.5;
  const hipWorldZ = (leftHipWorld[2] + rightHipWorld[2]) * 0.5;
  const pitch = toDegrees(
    Math.atan2(
      shoulderWorldZ - hipWorldZ,
      Math.max(0.001, Math.abs(shoulderWorldY - hipWorldY)),
    ),
  );
  const roll = toDegrees(Math.atan2(shoulderDy, shoulderDx));
  const lean = toDegrees(
    Math.atan2(
      shoulderMidX - hipMidX,
      Math.max(0.001, hipMidY - shoulderMidY),
    ),
  );
  return {
    yaw,
    pitch,
    roll,
    lean,
    facing: yaw > 15 ? 'right' : yaw < -15 ? 'left' : 'forward',
  };
}

export class PoseTrackingManager {
  static readonly POSE_CONNECTIONS: ReadonlyArray<{
    start: number;
    end: number;
  }> = [
    { start: 0, end: 1 }, { start: 1, end: 2 }, { start: 2, end: 3 },
    { start: 3, end: 7 }, { start: 0, end: 4 }, { start: 4, end: 5 },
    { start: 5, end: 6 }, { start: 6, end: 8 }, { start: 9, end: 10 },
    { start: 11, end: 12 }, { start: 11, end: 13 }, { start: 13, end: 15 },
    { start: 15, end: 17 }, { start: 15, end: 19 }, { start: 15, end: 21 },
    { start: 17, end: 19 }, { start: 12, end: 14 }, { start: 14, end: 16 },
    { start: 16, end: 18 }, { start: 16, end: 20 }, { start: 16, end: 22 },
    { start: 18, end: 20 }, { start: 11, end: 23 }, { start: 12, end: 24 },
    { start: 23, end: 24 }, { start: 23, end: 25 }, { start: 24, end: 26 },
    { start: 25, end: 27 }, { start: 26, end: 28 }, { start: 27, end: 29 },
    { start: 28, end: 30 }, { start: 29, end: 31 }, { start: 30, end: 32 },
    { start: 27, end: 31 }, { start: 28, end: 32 },
  ];

  private worker: Worker | null = null;
  private workerInitialization: Promise<MediaPipeDelegate> | null = null;
  private resolveWorkerInitialization: ((delegate: MediaPipeDelegate) => void) | null = null;
  private rejectWorkerInitialization: ((error: Error) => void) | null = null;
  private activeDelegate: MediaPipeDelegate | null = null;
  private video: HTMLVideoElement | null = null;
  private frameSubscription: VideoFrameSubscription | null = null;
  private latestSnapshot: PoseTrackingSnapshot = EMPTY_SNAPSHOT;
  private runToken = 0;
  private disposed = false;
  private inferenceInProgress = false;
  private lastVideoTime = -1;
  private lastInferenceStartedAt = 0;
  private targetInferenceFps = 10;
  private landmarkVisualizationEnabled = false;
  private fpsWindowStartedAt = 0;
  private framesInWindow = 0;
  private measuredFps = 0;
  private averagedInferenceTimeMs = 0;
  private averagedTrackingLatencyMs = 0;
  private readonly head: PosePoint = { x: 0, y: 0, z: 0, visibility: null, presence: null };
  private readonly nose: PosePoint = { x: 0, y: 0, z: 0, visibility: null, presence: null };
  private readonly leftEar: PosePoint = { x: 0, y: 0, z: 0, visibility: null, presence: null };
  private readonly rightEar: PosePoint = { x: 0, y: 0, z: 0, visibility: null, presence: null };
  private readonly leftShoulder: PosePoint = { x: 0, y: 0, z: 0, visibility: null, presence: null };
  private readonly rightShoulder: PosePoint = { x: 0, y: 0, z: 0, visibility: null, presence: null };
  private readonly leftElbow: PosePoint = { x: 0, y: 0, z: 0, visibility: null, presence: null };
  private readonly rightElbow: PosePoint = { x: 0, y: 0, z: 0, visibility: null, presence: null };
  private readonly leftWrist: PosePoint = { x: 0, y: 0, z: 0, visibility: null, presence: null };
  private readonly rightWrist: PosePoint = { x: 0, y: 0, z: 0, visibility: null, presence: null };
  private readonly leftHip: PosePoint = { x: 0, y: 0, z: 0, visibility: null, presence: null };
  private readonly rightHip: PosePoint = { x: 0, y: 0, z: 0, visibility: null, presence: null };
  private readonly chestMidpoint: PosePoint = { x: 0, y: 0, z: 0, visibility: null, presence: null };
  private readonly shoulderMidpoint: PosePoint = { x: 0, y: 0, z: 0, visibility: null, presence: null };
  private readonly hipMidpoint: PosePoint = { x: 0, y: 0, z: 0, visibility: null, presence: null };
  private readonly confidencePoints = [
    this.leftShoulder,
    this.rightShoulder,
    this.leftElbow,
    this.rightElbow,
    this.leftWrist,
    this.rightWrist,
  ];
  private readonly statusListeners = new Set<(status: PoseTrackingStatus) => void>();

  get snapshot(): PoseTrackingSnapshot {
    return this.latestSnapshot;
  }

  subscribeStatus(listener: (status: PoseTrackingStatus) => void): () => void {
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
      if (!this.disposed && token === this.runToken) this.setStatus('error', true);
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
    this.worker?.postMessage({ type: 'dispose' });
    this.worker?.terminate();
    this.worker = null;
    this.workerInitialization = null;
    this.resolveWorkerInitialization = null;
    this.rejectWorkerInitialization = null;
    this.activeDelegate = null;
    this.statusListeners.clear();
  }

  private getOrCreateWorker(): Promise<MediaPipeDelegate> {
    if (this.activeDelegate) return Promise.resolve(this.activeDelegate);
    if (this.workerInitialization) return this.workerInitialization;
    this.worker = new Worker(new URL('./poseTracking.worker.ts', import.meta.url), {
      type: 'module',
    });
    this.worker.addEventListener('message', this.handleWorkerMessage);
    this.worker.addEventListener('error', this.handleWorkerRuntimeError);
    this.workerInitialization = new Promise<MediaPipeDelegate>((resolve, reject) => {
      this.resolveWorkerInitialization = resolve;
      this.rejectWorkerInitialization = reject;
    });
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
    this.frameSubscription = subscribeToVideoFrames(video, this.processVideoFrame);
    this.latestSnapshot = {
      ...this.latestSnapshot,
      usingVideoFrameCallback: this.frameSubscription.usingVideoFrameCallback,
    };
  }

  private readonly processVideoFrame = (
    now: number,
    mediaTime: number,
    capture: VideoFrameCapture,
  ) => {
    const video = this.video;
    if (!video || !this.worker || this.disposed) return;
    if (
      video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA ||
      mediaTime === this.lastVideoTime ||
      this.inferenceInProgress ||
      now - this.lastInferenceStartedAt < 1000 / this.targetInferenceFps
    ) {
      return;
    }
    this.lastVideoTime = mediaTime;
    this.lastInferenceStartedAt = now;
    this.inferenceInProgress = true;
    void this.sendFrame(video, capture, now, this.runToken);
  };

  private async sendFrame(
    video: HTMLVideoElement,
    capture: VideoFrameCapture,
    timestamp: number,
    runToken: number,
  ): Promise<void> {
    try {
      const frame = await capture();
      if (this.disposed || runToken !== this.runToken || !this.worker || this.video !== video) {
        frame.close();
        if (runToken === this.runToken) this.inferenceInProgress = false;
        return;
      }
      this.worker.postMessage({ type: 'frame', frame, timestamp, runToken }, [frame]);
    } catch {
      if (runToken === this.runToken) this.inferenceInProgress = false;
    }
  }

  private readonly handleWorkerMessage = (
    event: MessageEvent<PoseTrackingWorkerResponse>,
  ) => {
    const message = event.data;
    if (message.type === 'ready') {
      this.activeDelegate = message.delegate;
      // Pose is spatially stable and is intentionally lower cadence than hands.
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
    const error = new Error(message.message);
    console.error('[HOLOFORGE] Pose tracking worker:', error);
    if (!this.activeDelegate) {
      this.rejectWorkerInitialization?.(error);
      this.worker?.terminate();
      this.worker = null;
      this.workerInitialization = null;
      this.resolveWorkerInitialization = null;
      this.rejectWorkerInitialization = null;
      return;
    }
    if (message.runToken === undefined || message.runToken === this.runToken) {
      this.inferenceInProgress = false;
      this.stopFrameLoop();
      this.video = null;
      this.setStatus('error');
      this.clearPose();
    }
  };

  private readonly handleWorkerRuntimeError = (event: ErrorEvent) => {
    const error = new Error(event.message || 'Pose tracking worker failed.');
    console.error('[HOLOFORGE] Pose worker runtime error:', error);
    this.rejectWorkerInitialization?.(error);
    this.worker?.terminate();
    this.worker = null;
    this.workerInitialization = null;
    this.resolveWorkerInitialization = null;
    this.rejectWorkerInitialization = null;
    this.activeDelegate = null;
    this.inferenceInProgress = false;
    this.stopFrameLoop();
    this.video = null;
    this.setStatus('error');
    this.clearPose();
  };

  private updateSnapshot(result: PoseWorkerResultMessage): void {
    const completedAt = performance.now();
    this.averagedInferenceTimeMs = this.averagedInferenceTimeMs === 0
      ? result.inferenceTimeMs
      : this.averagedInferenceTimeMs * 0.8 + result.inferenceTimeMs * 0.2;
    const latency = Math.max(0, completedAt - result.timestamp);
    this.averagedTrackingLatencyMs = this.averagedTrackingLatencyMs === 0
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

    if (
      !result.poseDetected ||
      result.keyLandmarks.length < KEY_LANDMARK_COUNT * POSE_LANDMARK_STRIDE ||
      result.keyWorldLandmarks.length < KEY_LANDMARK_COUNT * WORLD_LANDMARK_STRIDE
    ) {
      this.clearPose(result.timestamp);
      return;
    }

    setPoint(this.nose, result.keyLandmarks, NOSE);
    setPoint(this.leftEar, result.keyLandmarks, LEFT_EAR);
    setPoint(this.rightEar, result.keyLandmarks, RIGHT_EAR);
    setPoint(this.leftShoulder, result.keyLandmarks, LEFT_SHOULDER);
    setPoint(this.rightShoulder, result.keyLandmarks, RIGHT_SHOULDER);
    setPoint(this.leftElbow, result.keyLandmarks, LEFT_ELBOW);
    setPoint(this.rightElbow, result.keyLandmarks, RIGHT_ELBOW);
    setPoint(this.leftWrist, result.keyLandmarks, LEFT_WRIST);
    setPoint(this.rightWrist, result.keyLandmarks, RIGHT_WRIST);
    setPoint(this.leftHip, result.keyLandmarks, LEFT_HIP);
    setPoint(this.rightHip, result.keyLandmarks, RIGHT_HIP);
    midpoint(this.head, this.leftEar, this.rightEar);
    this.head.x = this.head.x * 0.7 + this.nose.x * 0.3;
    this.head.y = this.head.y * 0.7 + this.nose.y * 0.3;
    this.head.z = this.head.z * 0.7 + this.nose.z * 0.3;

    midpoint(this.shoulderMidpoint, this.leftShoulder, this.rightShoulder);
    midpoint(this.hipMidpoint, this.leftHip, this.rightHip);
    const chestDepth = 0.32;
    this.chestMidpoint.x = this.shoulderMidpoint.x +
      (this.hipMidpoint.x - this.shoulderMidpoint.x) * chestDepth;
    this.chestMidpoint.y = this.shoulderMidpoint.y +
      (this.hipMidpoint.y - this.shoulderMidpoint.y) * chestDepth;
    this.chestMidpoint.z = this.shoulderMidpoint.z +
      (this.hipMidpoint.z - this.shoulderMidpoint.z) * chestDepth;
    this.chestMidpoint.visibility = this.shoulderMidpoint.visibility;
    this.chestMidpoint.presence = this.shoulderMidpoint.presence;

    let confidenceTotal = 0;
    let confidenceCount = 0;
    for (const point of this.confidencePoints) {
      if (point.visibility === null) continue;
      confidenceTotal += point.visibility;
      confidenceCount += 1;
    }
    const confidence = confidenceCount > 0
      ? confidenceTotal / confidenceCount
      : null;

    this.latestSnapshot = {
      status: 'online',
      poseDetected: true,
      head: this.head,
      leftShoulder: this.leftShoulder,
      rightShoulder: this.rightShoulder,
      leftElbow: this.leftElbow,
      rightElbow: this.rightElbow,
      leftWrist: this.leftWrist,
      rightWrist: this.rightWrist,
      chestMidpoint: this.chestMidpoint,
      upperBodyOrientation: calculateOrientation(
        result.keyLandmarks,
        result.keyWorldLandmarks,
      ),
      confidence,
      landmarks: this.landmarkVisualizationEnabled ? result.landmarks : EMPTY_LANDMARKS,
      landmarkCount: this.landmarkVisualizationEnabled ? result.landmarkCount : 0,
      fps: this.measuredFps,
      inferenceTimeMs: this.averagedInferenceTimeMs,
      trackingLatencyMs: this.averagedTrackingLatencyMs,
      targetInferenceFps: this.targetInferenceFps,
      delegate: this.activeDelegate,
      usingVideoFrameCallback: this.frameSubscription?.usingVideoFrameCallback ?? false,
      timestamp: result.timestamp,
    };
  }

  private clearPose(timestamp = 0): void {
    this.latestSnapshot = {
      ...this.latestSnapshot,
      poseDetected: false,
      head: null,
      leftShoulder: null,
      rightShoulder: null,
      leftElbow: null,
      rightElbow: null,
      leftWrist: null,
      rightWrist: null,
      chestMidpoint: null,
      upperBodyOrientation: null,
      confidence: null,
      landmarks: EMPTY_LANDMARKS,
      landmarkCount: 0,
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

  private setStatus(status: PoseTrackingStatus, reset = false): void {
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
