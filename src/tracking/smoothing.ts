export interface SmoothingPoint {
  x: number;
  y: number;
  z: number;
  visibility?: number;
}

export interface OneEuroFilterConfig {
  minCutoff: number;
  beta: number;
  derivativeCutoff: number;
}

export interface MotionResponseConfig {
  fastVelocityThreshold: number;
  largeDisplacementThreshold: number;
  visualPredictionMs: number;
  maxPredictionDistance: number;
}

export interface LandmarkFilterConfig {
  visual: OneEuroFilterConfig;
  gesture: OneEuroFilterConfig;
  fingertipCutoffMultiplier: number;
  motion: MotionResponseConfig;
}

export interface FilteredLandmarkFrame {
  /** Predicted by a few milliseconds and intended only for rendering/effects. */
  visualLandmarks: SmoothingPoint[];
  /** Filtered visual locations before prediction, retained for diagnostics. */
  filteredVisualLandmarks: SmoothingPoint[];
  /** Responsive, never-predicted locations reserved for future gesture input. */
  gestureLandmarks: SmoothingPoint[];
  velocity: number;
  motionSpeed: number;
  palmDisplacement: number;
  fastMotionBlend: number;
  visualSmoothingStrength: number;
  filterDelayMs: number;
}

export const DEFAULT_VISUAL_FILTER_CONFIG: Readonly<OneEuroFilterConfig> = {
  minCutoff: 1.2,
  beta: 1,
  derivativeCutoff: 1,
};

export const DEFAULT_GESTURE_FILTER_CONFIG: Readonly<OneEuroFilterConfig> = {
  minCutoff: 2,
  beta: 1.35,
  derivativeCutoff: 1,
};

export const DEFAULT_MOTION_RESPONSE_CONFIG: Readonly<MotionResponseConfig> = {
  fastVelocityThreshold: 1.25,
  largeDisplacementThreshold: 0.1,
  visualPredictionMs: 16,
  maxPredictionDistance: 0.03,
};

export const DEFAULT_FINGERTIP_CUTOFF_MULTIPLIER = 0.88;
export const DEFAULT_DROPOUT_TOLERANCE_MS = 75;
export const DEFAULT_DROPOUT_FRAME_LIMIT = 2;

const FINGERTIP_INDICES = new Set([4, 8, 12, 16, 20]);
const HIGH_PRIORITY_TIPS = new Set([4, 8]);
const PALM_INDICES = new Set([0, 1, 5, 9, 13, 17]);
const MIN_DELTA_SECONDS = 1 / 240;
const MAX_DELTA_SECONDS = 0.25;
const HIGH_VELOCITY_RESPONSE = 1.5;
const FAST_BYPASS_ALPHA = 0.94;

interface AxisMotionResponse {
  lowVelocityThreshold: number;
  fastVelocityThreshold: number;
  globalBoost: number;
}

export function clamp(value: number, minimum: number, maximum: number): number {
  return Math.min(maximum, Math.max(minimum, value));
}

export function smoothStep(edge0: number, edge1: number, value: number): number {
  const progress = clamp((value - edge0) / Math.max(0.0001, edge1 - edge0), 0, 1);
  return progress * progress * (3 - 2 * progress);
}

export function shouldHoldLandmarks(
  missingFrames: number,
  elapsedMs: number,
  frameLimit = DEFAULT_DROPOUT_FRAME_LIMIT,
  toleranceMs = DEFAULT_DROPOUT_TOLERANCE_MS,
): boolean {
  return missingFrames <= frameLimit && elapsedMs <= toleranceMs;
}

export function shouldResetForReacquisition(
  wasMissing: boolean,
  normalizedPalmDisplacement: number,
  largeDisplacementThreshold: number,
): boolean {
  return (
    wasMissing &&
    normalizedPalmDisplacement >= largeDisplacementThreshold * 0.6
  );
}

export function smoothingAlpha(cutoff: number, deltaSeconds: number): number {
  const safeCutoff = Math.max(cutoff, 0.0001);
  const safeDelta = clamp(deltaSeconds, MIN_DELTA_SECONDS, MAX_DELTA_SECONDS);
  const timeConstant = 1 / (2 * Math.PI * safeCutoff);
  return 1 / (1 + timeConstant / safeDelta);
}

export class OneEuroFilter {
  private previousRaw: number | null = null;
  private previousFiltered: number | null = null;
  private previousDerivative = 0;
  private previousTimestamp = 0;
  private readonly output = { value: 0, velocity: 0, alpha: 1 };

  filter(
    value: number,
    timestamp: number,
    config: OneEuroFilterConfig,
    motion?: AxisMotionResponse,
  ): { value: number; velocity: number; alpha: number } {
    if (
      this.previousRaw === null ||
      this.previousFiltered === null ||
      timestamp <= this.previousTimestamp
    ) {
      this.previousRaw = value;
      this.previousFiltered = value;
      this.previousDerivative = 0;
      this.previousTimestamp = timestamp;
      this.output.value = value;
      this.output.velocity = 0;
      this.output.alpha = 1;
      return this.output;
    }

    const deltaSeconds = clamp(
      (timestamp - this.previousTimestamp) / 1000,
      MIN_DELTA_SECONDS,
      MAX_DELTA_SECONDS,
    );
    const rawDerivative = (value - this.previousRaw) / deltaSeconds;
    const derivativeAlpha = smoothingAlpha(config.derivativeCutoff, deltaSeconds);
    const filteredDerivative =
      derivativeAlpha * rawDerivative +
      (1 - derivativeAlpha) * this.previousDerivative;
    const velocityMagnitude = Math.abs(filteredDerivative);
    const adaptiveVelocity =
      velocityMagnitude +
      HIGH_VELOCITY_RESPONSE * velocityMagnitude * velocityMagnitude;
    const adaptiveCutoff = config.minCutoff + config.beta * adaptiveVelocity;
    let alpha = smoothingAlpha(adaptiveCutoff, deltaSeconds);

    if (motion) {
      const localFastMotion = smoothStep(
        motion.lowVelocityThreshold,
        motion.fastVelocityThreshold,
        velocityMagnitude,
      );
      const combinedFastMotion =
        1 - (1 - localFastMotion) * (1 - motion.globalBoost);
      // This smoothly raises the alpha floor toward near-raw tracking. There
      // are no hard mode switches, so acceleration and deceleration stay calm.
      alpha = Math.max(alpha, FAST_BYPASS_ALPHA * combinedFastMotion);
    }

    const filteredValue = alpha * value + (1 - alpha) * this.previousFiltered;
    this.previousRaw = value;
    this.previousFiltered = filteredValue;
    this.previousDerivative = filteredDerivative;
    this.previousTimestamp = timestamp;
    this.output.value = filteredValue;
    this.output.velocity = filteredDerivative;
    this.output.alpha = alpha;
    return this.output;
  }

  reset(): void {
    this.previousRaw = null;
    this.previousFiltered = null;
    this.previousDerivative = 0;
    this.previousTimestamp = 0;
  }
}

class VectorOneEuroFilter {
  private readonly x = new OneEuroFilter();
  private readonly y = new OneEuroFilter();
  private readonly z = new OneEuroFilter();
  private readonly outputPoint: SmoothingPoint = { x: 0, y: 0, z: 0 };
  private readonly output = {
    point: this.outputPoint,
    velocity: 0,
    velocityX: 0,
    velocityY: 0,
    velocityZ: 0,
    smoothingStrength: 0,
  };

  filter(
    point: SmoothingPoint,
    timestamp: number,
    config: OneEuroFilterConfig,
    response: AxisMotionResponse,
  ) {
    const x = this.x.filter(point.x, timestamp, config, response);
    const y = this.y.filter(point.y, timestamp, config, response);
    const z = this.z.filter(point.z, timestamp, config, response);
    this.outputPoint.x = x.value;
    this.outputPoint.y = y.value;
    this.outputPoint.z = z.value;
    this.outputPoint.visibility = point.visibility;
    this.output.velocityX = x.velocity;
    this.output.velocityY = y.velocity;
    this.output.velocityZ = z.velocity;
    this.output.velocity = Math.hypot(x.velocity, y.velocity, z.velocity * 0.35);
    this.output.smoothingStrength = 1 - (x.alpha + y.alpha + z.alpha) / 3;
    return this.output;
  }

  reset(): void {
    this.x.reset();
    this.y.reset();
    this.z.reset();
  }
}

/** Persistent visual and gesture filters for one landmark. */
export class AdaptiveLandmarkFilter {
  private readonly visualFilter = new VectorOneEuroFilter();
  private readonly gestureFilter = new VectorOneEuroFilter();
  private output: {
    visual: ReturnType<VectorOneEuroFilter['filter']>;
    gesture: ReturnType<VectorOneEuroFilter['filter']>;
  } | null = null;

  filter(
    point: SmoothingPoint,
    timestamp: number,
    visualConfig: OneEuroFilterConfig,
    gestureConfig: OneEuroFilterConfig,
    visualResponse: AxisMotionResponse,
    gestureResponse: AxisMotionResponse,
  ) {
    const visual = this.visualFilter.filter(
      point,
      timestamp,
      visualConfig,
      visualResponse,
    );
    const gesture = this.gestureFilter.filter(
      point,
      timestamp,
      gestureConfig,
      gestureResponse,
    );
    if (!this.output) this.output = { visual, gesture };
    else {
      this.output.visual = visual;
      this.output.gesture = gesture;
    }
    return this.output;
  }

  reset(): void {
    this.visualFilter.reset();
    this.gestureFilter.reset();
  }
}

/** Owns exactly 21 persistent landmark filters and prediction buffers. */
export class HandLandmarkFilter {
  private readonly filters = Array.from(
    { length: 21 },
    () => new AdaptiveLandmarkFilter(),
  );
  private readonly filteredVisualLandmarks = Array.from(
    { length: 21 },
    (): SmoothingPoint => ({ x: 0, y: 0, z: 0 }),
  );
  private readonly predictedVisualLandmarks = Array.from(
    { length: 21 },
    (): SmoothingPoint => ({ x: 0, y: 0, z: 0 }),
  );
  private readonly gestureLandmarks = Array.from(
    { length: 21 },
    (): SmoothingPoint => ({ x: 0, y: 0, z: 0 }),
  );
  private readonly previousPredictionVelocity = new Float32Array(21 * 3);
  private readonly previousRawPositions = new Float32Array(21 * 2);
  private readonly previousRawDelta = new Float32Array(21 * 2);
  private hasPreviousRawFrame = false;
  private readonly fingertipVisualConfig: OneEuroFilterConfig = {
    ...DEFAULT_VISUAL_FILTER_CONFIG,
  };
  private readonly visualResponse: AxisMotionResponse = {
    lowVelocityThreshold: 0,
    fastVelocityThreshold: 1,
    globalBoost: 0,
  };
  private readonly gestureResponse: AxisMotionResponse = {
    lowVelocityThreshold: 0,
    fastVelocityThreshold: 1,
    globalBoost: 0,
  };
  private previousRawPalmX: number | null = null;
  private previousRawPalmY: number | null = null;
  private previousTimestamp = 0;
  private readonly output: FilteredLandmarkFrame = {
    visualLandmarks: this.predictedVisualLandmarks,
    filteredVisualLandmarks: this.filteredVisualLandmarks,
    gestureLandmarks: this.gestureLandmarks,
    velocity: 0,
    motionSpeed: 0,
    palmDisplacement: 0,
    fastMotionBlend: 0,
    visualSmoothingStrength: 0,
    filterDelayMs: 0,
  };

  filter(
    rawLandmarks: readonly SmoothingPoint[],
    timestamp: number,
    config: LandmarkFilterConfig,
    allowPrediction = true,
  ): FilteredLandmarkFrame {
    if (rawLandmarks.length < this.filters.length) {
      throw new Error('HandLandmarkFilter requires 21 landmarks.');
    }

    const deltaSeconds =
      this.previousTimestamp > 0
        ? clamp(
            (timestamp - this.previousTimestamp) / 1000,
            MIN_DELTA_SECONDS,
            MAX_DELTA_SECONDS,
          )
        : 0;
    const rawPalm = rawLandmarks[0];
    const palmDisplacement =
      this.previousRawPalmX === null || this.previousRawPalmY === null
        ? 0
        : Math.hypot(
            rawPalm.x - this.previousRawPalmX,
            rawPalm.y - this.previousRawPalmY,
          );
    const palmSpeed = deltaSeconds > 0 ? palmDisplacement / deltaSeconds : 0;
    const displacementBoost = smoothStep(
      config.motion.largeDisplacementThreshold * 0.3,
      config.motion.largeDisplacementThreshold,
      palmDisplacement,
    );
    const palmFastMotion = smoothStep(
      config.motion.fastVelocityThreshold * 0.25,
      config.motion.fastVelocityThreshold,
      palmSpeed,
    );
    const globalBoost = Math.max(displacementBoost, palmFastMotion);
    this.previousRawPalmX = rawPalm.x;
    this.previousRawPalmY = rawPalm.y;
    this.previousTimestamp = timestamp;

    this.fingertipVisualConfig.minCutoff =
      config.visual.minCutoff * config.fingertipCutoffMultiplier;
    this.fingertipVisualConfig.beta = config.visual.beta;
    this.fingertipVisualConfig.derivativeCutoff = config.visual.derivativeCutoff;

    let velocityTotal = 0;
    let smoothingTotal = 0;
    let filterDelayMs = 0;
    const predictionSeconds = allowPrediction
      ? config.motion.visualPredictionMs / 1000
      : 0;

    for (let index = 0; index < this.filters.length; index += 1) {
      const isTip = FINGERTIP_INDICES.has(index);
      const responsivenessScale = HIGH_PRIORITY_TIPS.has(index)
        ? 0.58
        : isTip
          ? 0.72
          : PALM_INDICES.has(index)
            ? 1.12
            : 0.9;
      this.visualResponse.lowVelocityThreshold =
        config.motion.fastVelocityThreshold * 0.25 * responsivenessScale;
      this.visualResponse.fastVelocityThreshold =
        config.motion.fastVelocityThreshold * responsivenessScale;
      this.visualResponse.globalBoost = globalBoost;
      this.gestureResponse.lowVelocityThreshold =
        this.visualResponse.lowVelocityThreshold * 0.75;
      this.gestureResponse.fastVelocityThreshold =
        this.visualResponse.fastVelocityThreshold * 0.75;
      this.gestureResponse.globalBoost = Math.min(1, globalBoost * 1.15);

      const filtered = this.filters[index].filter(
        rawLandmarks[index],
        timestamp,
        isTip ? this.fingertipVisualConfig : config.visual,
        config.gesture,
        this.visualResponse,
        this.gestureResponse,
      );
      const visualPoint = filtered.visual.point;
      const predictedPoint = this.predictedVisualLandmarks[index];
      this.filteredVisualLandmarks[index] = visualPoint;
      this.gestureLandmarks[index] = filtered.gesture.point;

      const velocityOffset = index * 3;
      const rawOffset = index * 2;
      const rawDeltaX = this.hasPreviousRawFrame
        ? rawLandmarks[index].x - this.previousRawPositions[rawOffset]
        : 0;
      const rawDeltaY = this.hasPreviousRawFrame
        ? rawLandmarks[index].y - this.previousRawPositions[rawOffset + 1]
        : 0;
      const rawChangedDirection =
        this.hasPreviousRawFrame &&
        this.previousRawDelta[rawOffset] * rawDeltaX +
          this.previousRawDelta[rawOffset + 1] * rawDeltaY <
          0;
      const previousVelocityX = this.previousPredictionVelocity[velocityOffset];
      const previousVelocityY = this.previousPredictionVelocity[velocityOffset + 1];
      const velocityX = filtered.visual.velocityX;
      const velocityY = filtered.visual.velocityY;
      const velocityZ = filtered.visual.velocityZ;
      const changedDirection =
        rawChangedDirection ||
        previousVelocityX * velocityX + previousVelocityY * velocityY < 0;
      let predictionX = changedDirection ? 0 : velocityX * predictionSeconds;
      let predictionY = changedDirection ? 0 : velocityY * predictionSeconds;
      let predictionZ = changedDirection ? 0 : velocityZ * predictionSeconds;
      const predictionDistance = Math.hypot(predictionX, predictionY);
      if (predictionDistance > config.motion.maxPredictionDistance) {
        const scale = config.motion.maxPredictionDistance / predictionDistance;
        predictionX *= scale;
        predictionY *= scale;
        predictionZ *= scale;
      }
      predictedPoint.x = clamp(visualPoint.x + predictionX, -0.1, 1.1);
      predictedPoint.y = clamp(visualPoint.y + predictionY, -0.1, 1.1);
      predictedPoint.z = visualPoint.z + predictionZ;
      predictedPoint.visibility = visualPoint.visibility;
      this.previousPredictionVelocity[velocityOffset] = velocityX;
      this.previousPredictionVelocity[velocityOffset + 1] = velocityY;
      this.previousPredictionVelocity[velocityOffset + 2] = velocityZ;
      this.previousRawPositions[rawOffset] = rawLandmarks[index].x;
      this.previousRawPositions[rawOffset + 1] = rawLandmarks[index].y;
      this.previousRawDelta[rawOffset] = rawDeltaX;
      this.previousRawDelta[rawOffset + 1] = rawDeltaY;

      if ((index === 0 || index === 8) && filtered.visual.velocity > 0.08) {
        const spatialDelay = Math.hypot(
          rawLandmarks[index].x - visualPoint.x,
          rawLandmarks[index].y - visualPoint.y,
        );
        filterDelayMs = Math.max(
          filterDelayMs,
          clamp((spatialDelay / filtered.visual.velocity) * 1000, 0, 250),
        );
      }
      velocityTotal += filtered.visual.velocity;
      smoothingTotal += filtered.visual.smoothingStrength;
    }

    this.output.velocity = velocityTotal / this.filters.length;
    this.output.motionSpeed = palmSpeed;
    this.output.palmDisplacement = palmDisplacement;
    this.output.fastMotionBlend = Math.max(displacementBoost, palmFastMotion);
    this.output.visualSmoothingStrength = smoothingTotal / this.filters.length;
    this.output.filterDelayMs = filterDelayMs;
    this.hasPreviousRawFrame = true;
    return this.output;
  }

  reset(): void {
    this.filters.forEach((filter) => filter.reset());
    this.previousPredictionVelocity.fill(0);
    this.previousRawPositions.fill(0);
    this.previousRawDelta.fill(0);
    this.hasPreviousRawFrame = false;
    this.previousRawPalmX = null;
    this.previousRawPalmY = null;
    this.previousTimestamp = 0;
  }
}

export function sanitizeFilterConfig(
  config: OneEuroFilterConfig,
): OneEuroFilterConfig {
  return {
    minCutoff: clamp(config.minCutoff, 0.05, 10),
    beta: clamp(config.beta, 0, 5),
    derivativeCutoff: clamp(config.derivativeCutoff, 0.05, 10),
  };
}

export function sanitizeMotionResponseConfig(
  config: MotionResponseConfig,
): MotionResponseConfig {
  return {
    fastVelocityThreshold: clamp(config.fastVelocityThreshold, 0.25, 4),
    largeDisplacementThreshold: clamp(
      config.largeDisplacementThreshold,
      0.03,
      0.25,
    ),
    visualPredictionMs: clamp(config.visualPredictionMs, 0, 30),
    maxPredictionDistance: clamp(config.maxPredictionDistance, 0, 0.08),
  };
}
