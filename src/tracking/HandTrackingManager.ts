import {
  DEFAULT_DROPOUT_FRAME_LIMIT,
  DEFAULT_DROPOUT_TOLERANCE_MS,
  DEFAULT_FINGERTIP_CUTOFF_MULTIPLIER,
  DEFAULT_GESTURE_FILTER_CONFIG,
  DEFAULT_MOTION_RESPONSE_CONFIG,
  DEFAULT_VISUAL_FILTER_CONFIG,
  HandLandmarkFilter,
  clamp,
  sanitizeFilterConfig,
  sanitizeMotionResponseConfig,
  shouldHoldLandmarks,
  shouldResetForReacquisition,
  type LandmarkFilterConfig,
  type MotionResponseConfig,
  type OneEuroFilterConfig,
  type SmoothingPoint,
} from './smoothing';
import type {
  HandTrackingWorkerResponse,
  MediaPipeDelegate,
  MediaPipeTrackingConfig,
  WorkerResultMessage,
} from './handTracking.worker.types';
import { chooseAutoInferenceRate } from './inferenceRate';
import {
  subscribeToVideoFrames,
  type VideoFrameCapture,
  type VideoFrameSubscription,
} from './VideoFrameSource';

export type Handedness = 'Left' | 'Right' | 'Unknown';
export type HandTrackingStatus = 'idle' | 'loading' | 'online' | 'error';
export type HandTrackingRate = 'auto' | 30 | 45 | 60;

export type HandLandmarkPoint = SmoothingPoint;

export interface FingertipPositions {
  thumb: HandLandmarkPoint;
  index: HandLandmarkPoint;
  middle: HandLandmarkPoint;
  ring: HandLandmarkPoint;
  pinky: HandLandmarkPoint;
}

export interface TrackedHand {
  handedness: Handedness;
  /** Slightly stronger visual filtering for effects and the debug overlay. */
  landmarks: readonly HandLandmarkPoint[];
  /** Smoothed visual locations before the optional tiny prediction. */
  filteredLandmarks: readonly HandLandmarkPoint[];
  /** Lower-latency filtering reserved for gesture recognition. */
  gestureLandmarks: readonly HandLandmarkPoint[];
  /** Unfiltered MediaPipe positions retained for diagnostics and comparison. */
  rawLandmarks: readonly HandLandmarkPoint[];
  confidence: number;
  wrist: HandLandmarkPoint;
  fingertips: FingertipPositions;
  gestureWrist: HandLandmarkPoint;
  gestureFingertips: FingertipPositions;
  velocity: number;
  motionSpeed: number;
  palmDisplacement: number;
  fastMotionBlend: number;
  visualSmoothingStrength: number;
  filterDelayMs: number;
  reacquired: boolean;
  largeMovement: boolean;
  isHeld: boolean;
  timestamp: number;
}

export interface HandTrackingSnapshot {
  status: HandTrackingStatus;
  hands: readonly TrackedHand[];
  fps: number;
  cameraFps: number;
  renderFps: number;
  inferenceTimeMs: number;
  trackingLatencyMs: number;
  delegate: MediaPipeDelegate | null;
  cameraWidth: number;
  cameraHeight: number;
  actualCameraFrameRate: number;
  targetInferenceFps: number;
  trackingRate: HandTrackingRate;
  usingVideoFrameCallback: boolean;
  trackingLostCount: number;
  reacquisitionCount: number;
  motionBlurScore: number;
  timestamp: number;
}

const WASM_PATH = '/mediapipe/wasm';
const MODEL_PATH = '/models/hand_landmarker.task';

const EMPTY_SNAPSHOT: HandTrackingSnapshot = {
  status: 'idle',
  hands: [],
  fps: 0,
  cameraFps: 0,
  renderFps: 0,
  inferenceTimeMs: 0,
  trackingLatencyMs: 0,
  delegate: null,
  cameraWidth: 0,
  cameraHeight: 0,
  actualCameraFrameRate: 0,
  targetInferenceFps: 30,
  trackingRate: 'auto',
  usingVideoFrameCallback: false,
  trackingLostCount: 0,
  reacquisitionCount: 0,
  motionBlurScore: -1,
  timestamp: 0,
};

function normalizeHandedness(value: string | undefined): Handedness {
  const normalized = value?.toLowerCase();
  // MediaPipe assumes selfie-mirrored input. getUserMedia frames are raw and
  // only mirrored later with CSS, so swap the model label to physical handedness.
  if (normalized === 'left') return 'Right';
  if (normalized === 'right') return 'Left';
  return 'Unknown';
}

interface HandFilterState {
  filter: HandLandmarkFilter;
  rawLandmarks: HandLandmarkPoint[];
  recentHand: TrackedHand | null;
  lastSeenAt: number;
  missingFrames: number;
  awaitingReacquisition: boolean;
}

function createLandmarkBuffer(): HandLandmarkPoint[] {
  return Array.from(
    { length: 21 },
    (): HandLandmarkPoint => ({ x: 0, y: 0, z: 0 }),
  );
}

/** Owns MediaPipe and keeps frame-rate tracking data outside React state. */
export class HandTrackingManager {
  static readonly HAND_CONNECTIONS: ReadonlyArray<{
    start: number;
    end: number;
  }> = [
    { start: 0, end: 1 }, { start: 1, end: 5 },
    { start: 5, end: 9 }, { start: 9, end: 13 },
    { start: 13, end: 17 }, { start: 0, end: 17 },
    { start: 1, end: 2 }, { start: 2, end: 3 }, { start: 3, end: 4 },
    { start: 5, end: 6 }, { start: 6, end: 7 }, { start: 7, end: 8 },
    { start: 9, end: 10 }, { start: 10, end: 11 }, { start: 11, end: 12 },
    { start: 13, end: 14 }, { start: 14, end: 15 }, { start: 15, end: 16 },
    { start: 17, end: 18 }, { start: 18, end: 19 }, { start: 19, end: 20 },
  ];

  private worker: Worker | null = null;
  private workerInitialization: Promise<MediaPipeDelegate> | null = null;
  private resolveWorkerInitialization:
    | ((delegate: MediaPipeDelegate) => void)
    | null = null;
  private rejectWorkerInitialization: ((error: Error) => void) | null = null;
  private activeDelegate: MediaPipeDelegate | null = null;
  private video: HTMLVideoElement | null = null;
  private frameSubscription: VideoFrameSubscription | null = null;
  private usingVideoFrameCallback = false;
  private latestSnapshot: HandTrackingSnapshot = EMPTY_SNAPSHOT;
  private lastVideoTime = -1;
  private lastInferenceStartedAt = 0;
  private fpsWindowStartedAt = 0;
  private framesInWindow = 0;
  private measuredFps = 0;
  private cameraFpsWindowStartedAt = 0;
  private cameraFramesInWindow = 0;
  private measuredCameraFps = 0;
  private renderFpsWindowStartedAt = 0;
  private renderFramesInWindow = 0;
  private measuredRenderFps = 0;
  private averagedInferenceTimeMs = 0;
  private averagedTrackingLatencyMs = 0;
  private lastAutoRateEvaluationAt = 0;
  private runToken = 0;
  private disposed = false;
  private inferenceInProgress = false;
  private targetInferenceFps = 30;
  private trackingRate: HandTrackingRate = 'auto';
  private optionsUpdatePaused = false;
  private optionsRevision = 0;
  private optionsDebounceId: number | null = null;
  private trackingLostCount = 0;
  private reacquisitionCount = 0;
  private mediaPipeConfig: MediaPipeTrackingConfig = {
    minHandDetectionConfidence: 0.5,
    minHandPresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
  };
  private visualFilterConfig: OneEuroFilterConfig = {
    ...DEFAULT_VISUAL_FILTER_CONFIG,
  };
  private gestureFilterConfig: OneEuroFilterConfig = {
    ...DEFAULT_GESTURE_FILTER_CONFIG,
  };
  private fingertipCutoffMultiplier = DEFAULT_FINGERTIP_CUTOFF_MULTIPLIER;
  private motionResponseConfig: MotionResponseConfig = {
    ...DEFAULT_MOTION_RESPONSE_CONFIG,
  };
  private readonly handFilters = new Map<string, HandFilterState>();
  private readonly statusListeners = new Set<
    (status: HandTrackingStatus) => void
  >();

  get snapshot(): HandTrackingSnapshot {
    return this.latestSnapshot;
  }

  get filterConfig(): LandmarkFilterConfig {
    return {
      visual: { ...this.visualFilterConfig },
      gesture: { ...this.gestureFilterConfig },
      fingertipCutoffMultiplier: this.fingertipCutoffMultiplier,
      motion: { ...this.motionResponseConfig },
    };
  }

  get motionConfig(): MotionResponseConfig {
    return { ...this.motionResponseConfig };
  }

  get trackingConfidenceConfig(): MediaPipeTrackingConfig {
    return { ...this.mediaPipeConfig };
  }

  setVisualFilterConfig(config: Partial<OneEuroFilterConfig>): void {
    this.visualFilterConfig = sanitizeFilterConfig({
      ...this.visualFilterConfig,
      ...config,
    });
  }

  setGestureFilterConfig(config: Partial<OneEuroFilterConfig>): void {
    this.gestureFilterConfig = sanitizeFilterConfig({
      ...this.gestureFilterConfig,
      ...config,
    });
  }

  setFingertipCutoffMultiplier(multiplier: number): void {
    this.fingertipCutoffMultiplier = clamp(multiplier, 0.7, 1);
  }

  setMotionResponseConfig(config: Partial<MotionResponseConfig>): void {
    this.motionResponseConfig = sanitizeMotionResponseConfig({
      ...this.motionResponseConfig,
      ...config,
    });
  }

  setTrackingRate(rate: HandTrackingRate): void {
    this.trackingRate = rate;
    this.lastAutoRateEvaluationAt = 0;
    this.targetInferenceFps = rate === 'auto' ? 30 : rate;
    this.latestSnapshot = {
      ...this.latestSnapshot,
      trackingRate: rate,
      targetInferenceFps: this.targetInferenceFps,
    };
  }

  setMediaPipeConfig(config: Partial<MediaPipeTrackingConfig>): void {
    this.mediaPipeConfig = {
      minHandDetectionConfidence: clamp(
        config.minHandDetectionConfidence ??
          this.mediaPipeConfig.minHandDetectionConfidence,
        0.3,
        0.8,
      ),
      minHandPresenceConfidence: clamp(
        config.minHandPresenceConfidence ??
          this.mediaPipeConfig.minHandPresenceConfidence,
        0.3,
        0.8,
      ),
      minTrackingConfidence: clamp(
        config.minTrackingConfidence ?? this.mediaPipeConfig.minTrackingConfidence,
        0.3,
        0.8,
      ),
    };

    if (!this.worker || !this.activeDelegate) return;
    if (this.optionsDebounceId !== null) {
      window.clearTimeout(this.optionsDebounceId);
    }
    this.optionsDebounceId = window.setTimeout(() => {
      this.optionsDebounceId = null;
      if (!this.worker || !this.activeDelegate) return;
      this.optionsUpdatePaused = true;
      const revision = ++this.optionsRevision;
      this.worker.postMessage({
        type: 'configure',
        revision,
        config: this.mediaPipeConfig,
      });
    }, 120);
  }

  recordRenderFrame(timestamp: number): void {
    if (this.renderFpsWindowStartedAt === 0) {
      this.renderFpsWindowStartedAt = timestamp;
      this.renderFramesInWindow = 0;
      return;
    }
    this.renderFramesInWindow += 1;

    const windowMs = timestamp - this.renderFpsWindowStartedAt;
    if (windowMs < 500) return;

    this.measuredRenderFps =
      (this.renderFramesInWindow * 1000) / windowMs;
    this.renderFramesInWindow = 0;
    this.renderFpsWindowStartedAt = timestamp;
    this.latestSnapshot = {
      ...this.latestSnapshot,
      renderFps: this.measuredRenderFps,
    };
  }

  subscribeStatus(listener: (status: HandTrackingStatus) => void): () => void {
    this.statusListeners.add(listener);
    listener(this.latestSnapshot.status);
    return () => this.statusListeners.delete(listener);
  }

  async start(video: HTMLVideoElement): Promise<boolean> {
    const token = ++this.runToken;
    this.video = null;
    this.cancelLoop();
    this.resetFrameMetrics();
    this.setStatus('loading', true);

    try {
      await this.getOrCreateWorker();
      if (this.disposed || token !== this.runToken) return false;

      this.video = video;
      this.setStatus('online', true);
      this.startFrameLoop();
      return true;
    } catch (error) {
      if (token === this.runToken && !this.disposed) {
        this.setStatus('error', true);
      }
      throw error;
    }
  }

  stop(): void {
    this.runToken += 1;
    this.cancelLoop();
    this.video = null;
    this.resetFrameMetrics();
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
    if (this.optionsDebounceId !== null) {
      window.clearTimeout(this.optionsDebounceId);
      this.optionsDebounceId = null;
    }
    this.statusListeners.clear();
  }

  private getOrCreateWorker(): Promise<MediaPipeDelegate> {
    if (this.activeDelegate) return Promise.resolve(this.activeDelegate);
    if (this.workerInitialization) return this.workerInitialization;

    this.worker = new Worker(
      new URL('./handTracking.worker.ts', import.meta.url),
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
      config: this.mediaPipeConfig,
    });

    return this.workerInitialization;
  }

  private startFrameLoop() {
    const video = this.video;
    if (!video) return;
    this.frameSubscription = subscribeToVideoFrames(
      video,
      this.processVideoFrame,
    );
    this.usingVideoFrameCallback =
      this.frameSubscription.usingVideoFrameCallback;
  }

  private readonly processVideoFrame = (
    now: number,
    videoTime: number,
    capture: VideoFrameCapture,
  ) => {
    const video = this.video;
    if (!video || !this.worker || this.disposed) return;

    this.recordCameraFrame(now);
    this.tryInference(video, now, videoTime, capture);
  };

  private tryInference(
    video: HTMLVideoElement,
    now: number,
    videoTime: number,
    capture: VideoFrameCapture,
  ) {
    const intervalElapsed =
      now - this.lastInferenceStartedAt >= 1000 / this.targetInferenceFps;
    if (
      video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA ||
      videoTime === this.lastVideoTime ||
      this.inferenceInProgress ||
      this.optionsUpdatePaused ||
      !intervalElapsed
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
  ) {
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
    event: MessageEvent<HandTrackingWorkerResponse>,
  ) => {
    const message = event.data;

    if (message.type === 'ready') {
      this.activeDelegate = message.delegate;
      this.latestSnapshot = {
        ...this.latestSnapshot,
        delegate: message.delegate,
      };
      this.resolveWorkerInitialization?.(message.delegate);
      this.resolveWorkerInitialization = null;
      this.rejectWorkerInitialization = null;
      return;
    }

    if (message.type === 'configured') {
      if (message.revision === this.optionsRevision) {
        this.optionsUpdatePaused = false;
      }
      return;
    }

    if (message.type === 'result') {
      if (message.runToken !== this.runToken) return;
      this.inferenceInProgress = false;
      this.updateSnapshot(message);
      return;
    }

    const error = new Error(message.message);
    console.error('[HOLOFORGE] Hand tracking worker:', error);
    if (message.revision !== undefined) {
      if (message.revision === this.optionsRevision) {
        this.optionsUpdatePaused = false;
      }
      return;
    }
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
      this.optionsUpdatePaused = false;
      this.cancelLoop();
      this.video = null;
      this.setStatus('error');
      this.latestSnapshot = { ...this.latestSnapshot, hands: [] };
    }
  };

  private readonly handleWorkerRuntimeError = (event: ErrorEvent) => {
    const error = new Error(event.message || 'Hand tracking worker failed.');
    console.error('[HOLOFORGE] Hand tracking worker runtime error:', error);
    this.rejectWorkerInitialization?.(error);
    this.worker?.terminate();
    this.worker = null;
    this.activeDelegate = null;
    this.workerInitialization = null;
    this.resolveWorkerInitialization = null;
    this.rejectWorkerInitialization = null;
    this.inferenceInProgress = false;
    this.cancelLoop();
    this.video = null;
    this.setStatus('error');
  };

  private recordCameraFrame(timestamp: number) {
    if (this.cameraFpsWindowStartedAt === 0) {
      this.cameraFpsWindowStartedAt = timestamp;
      this.cameraFramesInWindow = 0;
      return;
    }
    this.cameraFramesInWindow += 1;

    const windowMs = timestamp - this.cameraFpsWindowStartedAt;
    if (windowMs < 500) return;
    this.measuredCameraFps =
      (this.cameraFramesInWindow * 1000) / windowMs;
    this.cameraFramesInWindow = 0;
    this.cameraFpsWindowStartedAt = timestamp;
  }

  private evaluateAutoRate(timestamp: number) {
    if (
      this.trackingRate !== 'auto' ||
      timestamp - this.lastAutoRateEvaluationAt < 1500 ||
      this.averagedInferenceTimeMs <= 0
    ) {
      return;
    }
    this.lastAutoRateEvaluationAt = timestamp;

    const cameraRate =
      this.measuredCameraFps > 0
        ? this.measuredCameraFps
        : this.video?.srcObject instanceof MediaStream
          ? (this.video.srcObject.getVideoTracks()[0]?.getSettings().frameRate ?? 30)
          : 30;
    this.targetInferenceFps = chooseAutoInferenceRate(
      cameraRate,
      this.averagedInferenceTimeMs,
      this.measuredRenderFps,
    );
  }

  private updateSnapshot(result: WorkerResultMessage) {
    const { timestamp, inferenceTimeMs } = result;
    const completedAt = performance.now();
    this.averagedInferenceTimeMs =
      this.averagedInferenceTimeMs === 0
        ? inferenceTimeMs
        : this.averagedInferenceTimeMs * 0.8 + inferenceTimeMs * 0.2;
    const trackingLatencyMs = Math.max(0, completedAt - timestamp);
    this.averagedTrackingLatencyMs =
      this.averagedTrackingLatencyMs === 0
        ? trackingLatencyMs
        : this.averagedTrackingLatencyMs * 0.8 + trackingLatencyMs * 0.2;
    if (this.fpsWindowStartedAt === 0) {
      this.fpsWindowStartedAt = timestamp;
      this.framesInWindow = 0;
    } else {
      this.framesInWindow += 1;
    }

    const fpsWindowMs = timestamp - this.fpsWindowStartedAt;
    if (fpsWindowMs >= 500) {
      this.measuredFps = (this.framesInWindow * 1000) / fpsWindowMs;
      this.framesInWindow = 0;
      this.fpsWindowStartedAt = timestamp;
    }

    const visibleKeys = new Set<string>();
    const hands: TrackedHand[] = [];
    const filterConfig = {
      visual: this.visualFilterConfig,
      gesture: this.gestureFilterConfig,
      fingertipCutoffMultiplier: this.fingertipCutoffMultiplier,
      motion: this.motionResponseConfig,
    };

    for (let handIndex = 0; handIndex < result.handCount; handIndex += 1) {
      const classification = result.handedness[handIndex];
      const handedness = normalizeHandedness(classification?.categoryName);
      const trackingKey =
        handedness === 'Unknown' ? `unknown-${handIndex}` : handedness;
      visibleKeys.add(trackingKey);

      let filterState = this.handFilters.get(trackingKey);
      if (!filterState) {
        filterState = {
          filter: new HandLandmarkFilter(),
          rawLandmarks: createLandmarkBuffer(),
          recentHand: null,
          lastSeenAt: timestamp,
          missingFrames: 0,
          awaitingReacquisition: false,
        };
        this.handFilters.set(trackingKey, filterState);
      }
      const rawPoints = filterState.rawLandmarks;
      const reacquired = filterState.awaitingReacquisition;
      const nextPalmX = result.landmarks[handIndex * 21 * 4];
      const nextPalmY = result.landmarks[handIndex * 21 * 4 + 1];
      const reacquisitionDisplacement =
        filterState.recentHand || reacquired
        ? Math.hypot(
            nextPalmX - rawPoints[0].x,
            nextPalmY - rawPoints[0].y,
          )
        : 0;
      const resetForReacquisition = shouldResetForReacquisition(
        reacquired,
        reacquisitionDisplacement,
        this.motionResponseConfig.largeDisplacementThreshold,
      );
      if (reacquired) {
        this.reacquisitionCount += 1;
        if (resetForReacquisition) {
          // A hand returning far from its last location must not fly slowly
          // across the screen through stale filter state.
          filterState.filter.reset();
        }
      }
      for (let landmarkIndex = 0; landmarkIndex < 21; landmarkIndex += 1) {
        const offset = (handIndex * 21 + landmarkIndex) * 4;
        const point = rawPoints[landmarkIndex];
        const visibility = result.landmarks[offset + 3];
        point.x = result.landmarks[offset];
        point.y = result.landmarks[offset + 1];
        point.z = result.landmarks[offset + 2];
        point.visibility = visibility >= 0 ? visibility : undefined;
      }
      const filtered = filterState.filter.filter(
        rawPoints,
        timestamp,
        filterConfig,
        !reacquired && (classification?.score ?? 0) >= 0.6,
      );
      const points = filtered.visualLandmarks;
      const filteredPoints = filtered.filteredVisualLandmarks;
      const gesturePoints = filtered.gestureLandmarks;

      const trackedHand: TrackedHand = {
        handedness,
        landmarks: points,
        filteredLandmarks: filteredPoints,
        gestureLandmarks: gesturePoints,
        rawLandmarks: rawPoints,
        confidence: classification?.score ?? 0,
        wrist: points[0],
        fingertips: {
          thumb: points[4], index: points[8], middle: points[12],
          ring: points[16], pinky: points[20],
        },
        gestureWrist: gesturePoints[0],
        gestureFingertips: {
          thumb: gesturePoints[4], index: gesturePoints[8],
          middle: gesturePoints[12], ring: gesturePoints[16],
          pinky: gesturePoints[20],
        },
        velocity: filtered.velocity,
        motionSpeed: filtered.motionSpeed,
        palmDisplacement: filtered.palmDisplacement,
        fastMotionBlend: filtered.fastMotionBlend,
        visualSmoothingStrength: filtered.visualSmoothingStrength,
        filterDelayMs: filtered.filterDelayMs,
        reacquired,
        largeMovement:
          resetForReacquisition ||
          filtered.palmDisplacement >=
            this.motionResponseConfig.largeDisplacementThreshold,
        isHeld: false,
        timestamp,
      };
      filterState.recentHand = trackedHand;
      filterState.lastSeenAt = timestamp;
      filterState.missingFrames = 0;
      filterState.awaitingReacquisition = false;
      hands.push(trackedHand);
    }

    for (const [trackingKey, filterState] of this.handFilters) {
      if (visibleKeys.has(trackingKey)) continue;
      if (!filterState.recentHand) continue;

      if (filterState.missingFrames === 0) {
        this.trackingLostCount += 1;
        filterState.awaitingReacquisition = true;
      }
      filterState.missingFrames += 1;
      if (
        shouldHoldLandmarks(
          filterState.missingFrames,
          timestamp - filterState.lastSeenAt,
          DEFAULT_DROPOUT_FRAME_LIMIT,
          DEFAULT_DROPOUT_TOLERANCE_MS,
        ) &&
        filterState.recentHand &&
        hands.length < 2
      ) {
        hands.push({ ...filterState.recentHand, isHeld: true });
      } else {
        filterState.filter.reset();
        filterState.recentHand = null;
        filterState.lastSeenAt = 0;
        filterState.missingFrames = 0;
      }
    }

    this.evaluateAutoRate(completedAt);
    const trackSettings = this.video?.srcObject instanceof MediaStream
      ? this.video.srcObject.getVideoTracks()[0]?.getSettings()
      : undefined;

    this.latestSnapshot = {
      status: 'online',
      hands,
      fps: this.measuredFps,
      cameraFps: this.measuredCameraFps,
      renderFps: this.measuredRenderFps,
      inferenceTimeMs: this.averagedInferenceTimeMs,
      trackingLatencyMs: this.averagedTrackingLatencyMs,
      delegate: this.activeDelegate,
      cameraWidth: this.video?.videoWidth ?? 0,
      cameraHeight: this.video?.videoHeight ?? 0,
      actualCameraFrameRate: trackSettings?.frameRate ?? 0,
      targetInferenceFps: this.targetInferenceFps,
      trackingRate: this.trackingRate,
      usingVideoFrameCallback: this.usingVideoFrameCallback,
      trackingLostCount: this.trackingLostCount,
      reacquisitionCount: this.reacquisitionCount,
      motionBlurScore: result.motionBlurScore,
      timestamp,
    };
  }

  private cancelLoop() {
    this.frameSubscription?.unsubscribe();
    this.frameSubscription = null;
  }

  private resetFrameMetrics() {
    this.lastVideoTime = -1;
    this.lastInferenceStartedAt = 0;
    this.fpsWindowStartedAt = 0;
    this.framesInWindow = 0;
    this.measuredFps = 0;
    this.cameraFpsWindowStartedAt = 0;
    this.cameraFramesInWindow = 0;
    this.measuredCameraFps = 0;
    this.renderFpsWindowStartedAt = 0;
    this.renderFramesInWindow = 0;
    this.measuredRenderFps = 0;
    this.averagedInferenceTimeMs = 0;
    this.averagedTrackingLatencyMs = 0;
    this.lastAutoRateEvaluationAt = 0;
    this.inferenceInProgress = false;
    this.optionsUpdatePaused = false;
    this.usingVideoFrameCallback = false;
    this.trackingLostCount = 0;
    this.reacquisitionCount = 0;
    this.handFilters.forEach((state) => {
      state.filter.reset();
      state.recentHand = null;
      state.lastSeenAt = 0;
      state.missingFrames = 0;
      state.awaitingReacquisition = false;
    });
  }

  private setStatus(status: HandTrackingStatus, reset = false) {
    const previousStatus = this.latestSnapshot.status;
    this.latestSnapshot = reset
      ? {
          ...EMPTY_SNAPSHOT,
          status,
          trackingRate: this.trackingRate,
          targetInferenceFps: this.targetInferenceFps,
        }
      : { ...this.latestSnapshot, status };

    if (previousStatus !== status) {
      this.statusListeners.forEach((listener) => listener(status));
    }
  }
}
