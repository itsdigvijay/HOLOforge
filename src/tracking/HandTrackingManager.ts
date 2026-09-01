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

/**
 * ---------------------------------------------------------------------------
 * Refactor summary (see inline comments at each change for the full
 * rationale):
 *
 * 1. FPS stabilization  - `tryInference` now gates on a fixed-timestep
 *    accumulator instead of a "time since last trigger" check. The naive
 *    check silently drifts the achieved rate below target because the small
 *    overshoot past the threshold gets discarded every cycle; the
 *    accumulator carries that remainder forward so the long-run average
 *    converges on `targetInferenceFps`.
 *
 * 2. Screen-tear prevention / rAF sync - added `getInterpolatedHands(now)`.
 *    Rather than the manager owning its own `requestAnimationFrame` loop
 *    (which would race with whatever loop the host app already uses to
 *    paint - three.js, a `<canvas>` 2D loop, React Three Fiber's
 *    `useFrame`, etc.), the manager exposes a pure, on-demand function the
 *    host calls from *its* rAF callback. That keeps the interpolated pose
 *    always time-correct for the exact instant the browser is about to
 *    paint, which is what actually prevents visible tearing/stutter -
 *    spinning up a second independent timer would not.
 *
 * 3. Latency - interpolation is applied only to the visual `landmarks`
 *    path used for rendering. `gestureLandmarks` (already documented as
 *    "Lower-latency filtering reserved for gesture recognition") and the
 *    raw inference `snapshot.hands` are untouched, so gesture recognition
 *    keeps running at native inference cadence with zero added delay.
 *
 * 4. Smoothing & stability - a motion-adaptive extrapolation ceiling
 *    (`motionTrust`, derived from the filter's existing `fastMotionBlend`)
 *    plus an EMA-smoothed inference-interval estimate
 *    (`smoothedInferenceIntervalMs`) act as a dynamic trust factor: fast
 *    hands get a short lookahead (less overshoot risk), slow/steady hands
 *    get a longer one (smoother glide), and the interpolation timebase
 *    itself is insulated from irregular arrival jitter.
 *
 * 5. Resource efficiency - filter/media-pipe config setters now short-
 *    circuit on a shallow-equality check before re-sanitizing or touching
 *    the worker, and the `filterConfig` getter (plus the internal per-frame
 *    variant used in `updateSnapshot`) is cached behind a revision counter
 *    instead of allocating fresh nested objects on every read/frame.
 * ---------------------------------------------------------------------------
 */

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

/**
 * A lightweight, render-only hand pose returned by
 * `HandTrackingManager.getInterpolatedHands()`. Deliberately a subset of
 * `TrackedHand` - it omits `gestureLandmarks`/`rawLandmarks` because those
 * feed gesture recognition, which must never be smoothed/interpolated (that
 * would add latency to detection). Call this from your own
 * `requestAnimationFrame` / render-loop callback, once per paint, to get a
 * pose whose timing matches the instant the frame will actually be drawn.
 */
export interface RenderHandPose {
  handedness: Handedness;
  landmarks: readonly HandLandmarkPoint[];
  wrist: HandLandmarkPoint;
  fingertips: FingertipPositions;
  confidence: number;
  isHeld: boolean;
  /** True when `now` is past the last inference sample and this pose was extrapolated rather than interpolated. */
  isExtrapolated: boolean;
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

/**
 * Cheap flat-object equality used to short-circuit config setters. All the
 * config types here (`OneEuroFilterConfig`, `MotionResponseConfig`,
 * `MediaPipeTrackingConfig`) are single-level records of primitives, so a
 * `for...in` comparison is sufficient and avoids pulling in a deep-equal
 * dependency for what is a hot-ish path (called from UI sliders/panels).
 */
function shallowEqualConfig<T extends object>(
  a: T,
  b: T,
): boolean {
  for (const key of Object.keys(a) as Array<keyof T>) {
    if (a[key] !== b[key]) return false;
  }
  for (const key of Object.keys(b) as Array<keyof T>) {
    if (!(key in a)) return false;
  }
  return true;
}

function lerpPoint(
  from: HandLandmarkPoint,
  to: HandLandmarkPoint,
  t: number,
): HandLandmarkPoint {
  return {
    x: from.x + (to.x - from.x) * t,
    y: from.y + (to.y - from.y) * t,
    z: from.z + (to.z - from.z) * t,
    // Visibility isn't a spatial quantity - carry the destination's value
    // rather than blending it.
    visibility: to.visibility,
  };
}

/** Builds a render pose straight from a `TrackedHand`, no interpolation. */
function toRenderPose(hand: TrackedHand, isExtrapolated: boolean): RenderHandPose {
  return {
    handedness: hand.handedness,
    landmarks: hand.landmarks,
    wrist: hand.wrist,
    fingertips: hand.fingertips,
    confidence: hand.confidence,
    isHeld: hand.isHeld,
    isExtrapolated,
  };
}

/**
 * Interpolates (t <= 1) or lightly extrapolates (t > 1) the *visual*
 * landmark set between two consecutive inference results. Only the 21-point
 * `landmarks` array is touched - this is the array `TrackedHand` documents
 * as being used "for effects and the debug overlay", i.e. exactly what a
 * renderer draws.
 */
function interpolateHand(
  previous: TrackedHand,
  current: TrackedHand,
  t: number,
): RenderHandPose {
  const landmarks = current.landmarks.map((point, index) =>
    lerpPoint(previous.landmarks[index] ?? point, point, t),
  );
  return {
    handedness: current.handedness,
    landmarks,
    wrist: landmarks[0],
    fingertips: {
      thumb: landmarks[4],
      index: landmarks[8],
      middle: landmarks[12],
      ring: landmarks[16],
      pinky: landmarks[20],
    },
    confidence: current.confidence,
    isHeld: current.isHeld,
    isExtrapolated: t > 1,
  };
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

  // --- Fixed-timestep accumulator for inference gating (FPS stabilization).
  // See the class-level comment (item 1) for why this replaces a naive
  // "time since last trigger" check.
  private lastFrameCallbackAt = 0;
  private inferenceAccumulatorMs = 0;

  // --- Render-interpolation state (screen-tear prevention / item 2 & 4).
  // "current"/"previous" refer to the two most recent inference results,
  // keyed by the same per-hand tracking key used elsewhere in this class.
  private previousInferenceHands = new Map<string, TrackedHand>();
  private currentInferenceHands = new Map<string, TrackedHand>();
  private currentInferenceReceivedAt = 0;
  /** EMA of the wall-clock gap between inference results, used as the interpolation timebase instead of the raw (jittery) gap. */
  private smoothedInferenceIntervalMs = 0;

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

  // --- Config caching (resource efficiency / item 5). A revision counter
  // rather than a single dirty flag, because two independent caches
  // (`filterConfig` and the internal per-frame variant) each need to know
  // whether *they* are stale, and a single boolean cleared by whichever one
  // reads first would leave the other silently stale.
  private filterConfigRevision = 0;
  private cachedFilterConfig: LandmarkFilterConfig | null = null;
  private cachedFilterConfigRevision = -1;
  private cachedInternalFilterConfig: LandmarkFilterConfig | null = null;
  private cachedInternalFilterConfigRevision = -1;

  private readonly handFilters = new Map<string, HandFilterState>();
  private readonly statusListeners = new Set<
    (status: HandTrackingStatus) => void
  >();

  get snapshot(): HandTrackingSnapshot {
    return this.latestSnapshot;
  }

  get filterConfig(): LandmarkFilterConfig {
    // Cached behind a revision counter so components that read this every
    // render (e.g. a debug/tuning panel) don't force a fresh object plus
    // three nested-object allocations on every access - only setters that
    // actually change a value bump the revision.
    if (
      !this.cachedFilterConfig ||
      this.cachedFilterConfigRevision !== this.filterConfigRevision
    ) {
      this.cachedFilterConfig = {
        visual: { ...this.visualFilterConfig },
        gesture: { ...this.gestureFilterConfig },
        fingertipCutoffMultiplier: this.fingertipCutoffMultiplier,
        motion: { ...this.motionResponseConfig },
      };
      this.cachedFilterConfigRevision = this.filterConfigRevision;
    }
    return this.cachedFilterConfig;
  }

  get motionConfig(): MotionResponseConfig {
    return { ...this.motionResponseConfig };
  }

  get trackingConfidenceConfig(): MediaPipeTrackingConfig {
    return { ...this.mediaPipeConfig };
  }

  setVisualFilterConfig(config: Partial<OneEuroFilterConfig>): void {
    const next = sanitizeFilterConfig({
      ...this.visualFilterConfig,
      ...config,
    });
    // Skip the write (and the cache invalidation it would trigger) when
    // sanitization resolves to the same values, e.g. a UI slider re-firing
    // its current value.
    if (shallowEqualConfig(next, this.visualFilterConfig)) return;
    this.visualFilterConfig = next;
    this.filterConfigRevision += 1;
  }

  setGestureFilterConfig(config: Partial<OneEuroFilterConfig>): void {
    const next = sanitizeFilterConfig({
      ...this.gestureFilterConfig,
      ...config,
    });
    if (shallowEqualConfig(next, this.gestureFilterConfig)) return;
    this.gestureFilterConfig = next;
    this.filterConfigRevision += 1;
  }

  setFingertipCutoffMultiplier(multiplier: number): void {
    const next = clamp(multiplier, 0.7, 1);
    if (next === this.fingertipCutoffMultiplier) return;
    this.fingertipCutoffMultiplier = next;
    this.filterConfigRevision += 1;
  }

  setMotionResponseConfig(config: Partial<MotionResponseConfig>): void {
    const next = sanitizeMotionResponseConfig({
      ...this.motionResponseConfig,
      ...config,
    });
    if (shallowEqualConfig(next, this.motionResponseConfig)) return;
    this.motionResponseConfig = next;
    this.filterConfigRevision += 1;
  }

  setTrackingRate(rate: HandTrackingRate): void {
    this.trackingRate = rate;
    this.lastAutoRateEvaluationAt = 0;
    this.targetInferenceFps = rate === 'auto' ? 30 : rate;
    // The accumulator's banked time was measured against the *old*
    // interval; carrying it over could fire an inference immediately under
    // the new rate. Resetting keeps the rate change phase-clean.
    this.inferenceAccumulatorMs = 0;
    this.latestSnapshot = {
      ...this.latestSnapshot,
      trackingRate: rate,
      targetInferenceFps: this.targetInferenceFps,
    };
  }

  setMediaPipeConfig(config: Partial<MediaPipeTrackingConfig>): void {
    const next: MediaPipeTrackingConfig = {
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

    // Resource efficiency: skip the debounce timer and worker round-trip
    // entirely when nothing actually changed.
    if (shallowEqualConfig(next, this.mediaPipeConfig)) return;
    this.mediaPipeConfig = next;

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

  /**
   * Returns a smoothly interpolated/lightly-extrapolated hand pose for
   * `now` (defaults to `performance.now()`). Call this once per tick from
   * your own render loop's `requestAnimationFrame` / `useFrame` callback -
   * see the class-level comment (item 2) for why the manager doesn't run
   * its own rAF loop. Inference typically runs at 30-60fps while a display
   * can refresh at 60-120Hz+; without this, hands visibly "step" between
   * inference samples. With it, the returned pose always matches the exact
   * instant the browser is about to paint, which is what removes both the
   * stepping and any tearing relative to the render loop.
   *
   * Gesture recognition should keep reading `snapshot.hands[*].gestureLandmarks`
   * directly - never this - so recognition latency stays untouched.
   */
  getInterpolatedHands(now: number = performance.now()): RenderHandPose[] {
    const results: RenderHandPose[] = [];
    const interval =
      this.smoothedInferenceIntervalMs > 0
        ? this.smoothedInferenceIntervalMs
        : 1000 / this.targetInferenceFps;

    for (const [trackingKey, currentHand] of this.currentInferenceHands) {
      const previousHand = this.previousInferenceHands.get(trackingKey);
      const elapsed = now - this.currentInferenceReceivedAt;

      if (!previousHand || currentHand.isHeld || elapsed <= 0) {
        // Nothing sensible to interpolate from - a brand-new hand, a
        // dropout-hold frame (already a frozen snapshot), or a render tick
        // that landed before this inference result was processed. Show the
        // latest known pose rather than guessing.
        results.push(toRenderPose(currentHand, false));
        continue;
      }

      // Motion-adaptive extrapolation ceiling (item 4): prediction error
      // compounds with acceleration, so fast-moving hands (`fastMotionBlend`
      // near 1) get pulled back toward a short lookahead, while slow/steady
      // hands are allowed to glide up to 1.6x past the last sample for a
      // smoother feel instead of visibly "waiting" for the next inference.
      const motionTrust = clamp(1 - currentHand.fastMotionBlend, 0.35, 1);
      const maxLookaheadFactor = 1 + motionTrust * 0.6;
      const t = clamp(elapsed / interval, 0, maxLookaheadFactor);

      results.push(interpolateHand(previousHand, currentHand, t));
    }

    return results;
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
    // Fixed-timestep accumulator (item 1). We accumulate real elapsed time
    // every frame callback and, when we actually fire, subtract exactly one
    // target interval rather than resetting to zero. Any leftover carries
    // into the next cycle, so the achieved average rate converges on
    // `targetInferenceFps` instead of drifting low - which is what a naive
    // "now - lastStart >= interval" check does, since it discards whatever
    // small overshoot existed at the moment of the check on every cycle.
    const dt = this.lastFrameCallbackAt === 0 ? 0 : now - this.lastFrameCallbackAt;
    this.lastFrameCallbackAt = now;
    this.inferenceAccumulatorMs += dt;

    const targetIntervalMs = 1000 / this.targetInferenceFps;
    // Cap the bank so a backgrounded/throttled tab regaining focus doesn't
    // cause a burst of queued catch-up inferences all at once.
    if (this.inferenceAccumulatorMs > targetIntervalMs * 2) {
      this.inferenceAccumulatorMs = targetIntervalMs;
    }

    const intervalElapsed = this.inferenceAccumulatorMs >= targetIntervalMs;

    if (
      video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA ||
      videoTime === this.lastVideoTime ||
      this.inferenceInProgress ||
      this.optionsUpdatePaused ||
      !intervalElapsed
    ) {
      return;
    }

    this.inferenceAccumulatorMs -= targetIntervalMs;
    this.lastVideoTime = videoTime;
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
    const nextTargetFps = chooseAutoInferenceRate(
      cameraRate,
      this.averagedInferenceTimeMs,
      this.measuredRenderFps,
    );
    if (nextTargetFps !== this.targetInferenceFps) {
      this.targetInferenceFps = nextTargetFps;
      // Keep the accumulator phase-clean across a rate change, same
      // reasoning as in `setTrackingRate`.
      this.inferenceAccumulatorMs = 0;
    }
  }

  private getInternalFilterConfig(): LandmarkFilterConfig {
    // Same cache-behind-a-revision strategy as the public `filterConfig`
    // getter, but this one holds direct references (no spreads) since it's
    // rebuilt at most once per inference result and consumed synchronously
    // by `HandLandmarkFilter.filter` within the same call - no need to pay
    // for a defensive copy on this path.
    if (
      !this.cachedInternalFilterConfig ||
      this.cachedInternalFilterConfigRevision !== this.filterConfigRevision
    ) {
      this.cachedInternalFilterConfig = {
        visual: this.visualFilterConfig,
        gesture: this.gestureFilterConfig,
        fingertipCutoffMultiplier: this.fingertipCutoffMultiplier,
        motion: this.motionResponseConfig,
      };
      this.cachedInternalFilterConfigRevision = this.filterConfigRevision;
    }
    return this.cachedInternalFilterConfig;
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

    // --- Render-interpolation bookkeeping (item 2 & 4) ------------------
    // Roll "current" into "previous" and smooth the arrival-interval
    // estimate with an EMA so `getInterpolatedHands` has a stable timebase
    // even when individual inference results arrive at irregular intervals
    // (GC pauses, camera jitter, auto-rate changes, etc).
    const previousReceivedAt = this.currentInferenceReceivedAt;
    if (previousReceivedAt > 0) {
      const rawIntervalMs = completedAt - previousReceivedAt;
      if (rawIntervalMs > 0 && rawIntervalMs < 500) {
        this.smoothedInferenceIntervalMs =
          this.smoothedInferenceIntervalMs === 0
            ? rawIntervalMs
            : this.smoothedInferenceIntervalMs * 0.7 + rawIntervalMs * 0.3;
      }
    }
    this.previousInferenceHands = this.currentInferenceHands;
    this.currentInferenceHands = new Map<string, TrackedHand>();
    this.currentInferenceReceivedAt = completedAt;
    // ----------------------------------------------------------------------

    const visibleKeys = new Set<string>();
    const hands: TrackedHand[] = [];
    const filterConfig = this.getInternalFilterConfig();

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
      this.currentInferenceHands.set(trackingKey, trackedHand);
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
        const heldHand: TrackedHand = { ...filterState.recentHand, isHeld: true };
        hands.push(heldHand);
        this.currentInferenceHands.set(trackingKey, heldHand);
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
    this.lastFrameCallbackAt = 0;
    this.inferenceAccumulatorMs = 0;
    this.previousInferenceHands = new Map();
    this.currentInferenceHands = new Map();
    this.previousInferenceReceivedAt = 0;
    this.currentInferenceReceivedAt = 0;
    this.smoothedInferenceIntervalMs = 0;
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
