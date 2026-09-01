import type { ManagerStatus } from '../types';

export type GestureState =
  | 'IDLE'
  | 'DETECTING'
  | 'CHARGING'
  | 'ACTIVE'
  | 'COOLDOWN';

export type GestureEventType = 'gesture:start' | 'gesture:hold' | 'gesture:end';

export type GestureEndReason =
  | 'released'
  | 'cancelled'
  | 'superseded'
  | 'unregistered'
  | 'disposed';

/**
 * A gesture definition only answers "how well does this input match?". The
 * manager owns all timing, smoothing and lifecycle decisions.
 */
export interface GestureDefinition<TInput> {
  name: string;
  evaluate: (input: TInput) => number;
  priority?: number;
  confidenceThreshold?: number;
  releaseThreshold?: number;
  minimumHoldDurationMs?: number;
  debounceDurationMs?: number;
  cooldownDurationMs?: number;
  smoothingTimeConstantMs?: number;
}

export interface GestureSnapshot {
  name: string;
  confidence: number;
  rawConfidence: number;
  state: GestureState;
  holdProgress: number;
  activeDurationMs: number;
  cooldownRemainingMs: number;
  priority: number;
}

export interface GestureEvent {
  type: GestureEventType;
  timestampMs: number;
  gesture: GestureSnapshot;
  reason?: GestureEndReason;
}

export type GestureEventListener = (event: GestureEvent) => void;

interface ResolvedGestureDefinition<TInput> {
  name: string;
  evaluate: (input: TInput) => number;
  priority: number;
  confidenceThreshold: number;
  releaseThreshold: number;
  minimumHoldDurationMs: number;
  debounceDurationMs: number;
  cooldownDurationMs: number;
  smoothingTimeConstantMs: number;
}

interface GestureRuntime<TInput> {
  definition: ResolvedGestureDefinition<TInput>;
  registrationOrder: number;
  state: GestureState;
  rawConfidence: number;
  smoothedConfidence: number;
  hasConfidenceSample: boolean;
  detectionStartedAtMs: number | null;
  chargingStartedAtMs: number | null;
  activeStartedAtMs: number | null;
  activeEndedAtMs: number | null;
  releaseStartedAtMs: number | null;
  cooldownEndsAtMs: number | null;
}

const DEFAULT_CONFIDENCE_THRESHOLD = 0.75;
const DEFAULT_MINIMUM_HOLD_DURATION_MS = 250;
const DEFAULT_DEBOUNCE_DURATION_MS = 80;
const DEFAULT_COOLDOWN_DURATION_MS = 500;
const DEFAULT_SMOOTHING_TIME_CONSTANT_MS = 60;

function clamp01(value: number): number {
  if (!Number.isFinite(value)) return 0;
  return Math.min(1, Math.max(0, value));
}

function requireNonNegative(value: number, field: string): number {
  if (!Number.isFinite(value) || value < 0) {
    throw new RangeError(`${field} must be a finite, non-negative number.`);
  }
  return value;
}

function resolveDefinition<TInput>(
  definition: GestureDefinition<TInput>,
): ResolvedGestureDefinition<TInput> {
  const name = definition.name.trim();
  if (!name) throw new Error('Gesture names cannot be empty.');

  const confidenceThreshold = clamp01(
    definition.confidenceThreshold ?? DEFAULT_CONFIDENCE_THRESHOLD,
  );
  const releaseThreshold = clamp01(
    definition.releaseThreshold ?? confidenceThreshold * 0.75,
  );

  if (releaseThreshold > confidenceThreshold) {
    throw new RangeError(
      `Gesture "${name}" releaseThreshold cannot exceed confidenceThreshold.`,
    );
  }

  const minimumHoldDurationMs = requireNonNegative(
    definition.minimumHoldDurationMs ?? DEFAULT_MINIMUM_HOLD_DURATION_MS,
    `Gesture "${name}" minimumHoldDurationMs`,
  );
  if (minimumHoldDurationMs === 0) {
    throw new RangeError(
      `Gesture "${name}" minimumHoldDurationMs must be greater than zero so it cannot activate from one frame.`,
    );
  }

  return {
    name,
    evaluate: definition.evaluate,
    priority: requireNonNegative(
      definition.priority ?? 0,
      `Gesture "${name}" priority`,
    ),
    confidenceThreshold,
    releaseThreshold,
    minimumHoldDurationMs,
    debounceDurationMs: requireNonNegative(
      definition.debounceDurationMs ?? DEFAULT_DEBOUNCE_DURATION_MS,
      `Gesture "${name}" debounceDurationMs`,
    ),
    cooldownDurationMs: requireNonNegative(
      definition.cooldownDurationMs ?? DEFAULT_COOLDOWN_DURATION_MS,
      `Gesture "${name}" cooldownDurationMs`,
    ),
    smoothingTimeConstantMs: requireNonNegative(
      definition.smoothingTimeConstantMs ??
        DEFAULT_SMOOTHING_TIME_CONSTANT_MS,
      `Gesture "${name}" smoothingTimeConstantMs`,
    ),
  };
}

/**
 * Reusable temporal gesture engine. It accepts arbitrary tracking input, so a
 * gesture can later consume hands, face, pose, or a combined tracking frame.
 */
export class GestureManager<TInput = unknown> {
  private currentStatus: ManagerStatus = 'idle';
  private readonly gestures = new Map<string, GestureRuntime<TInput>>();
  private readonly listeners = new Set<GestureEventListener>();
  private lastTimestampMs: number | null = null;
  private activeGestureName: string | null = null;
  private registrationCounter = 0;

  get status(): ManagerStatus {
    return this.currentStatus;
  }

  initialize(): void {
    if (this.currentStatus === 'idle') this.currentStatus = 'ready';
  }

  registerGesture(definition: GestureDefinition<TInput>): () => void {
    const resolved = resolveDefinition(definition);
    if (this.gestures.has(resolved.name)) {
      throw new Error(`Gesture "${resolved.name}" is already registered.`);
    }

    this.gestures.set(resolved.name, {
      definition: resolved,
      registrationOrder: this.registrationCounter++,
      state: 'IDLE',
      rawConfidence: 0,
      smoothedConfidence: 0,
      hasConfidenceSample: false,
      detectionStartedAtMs: null,
      chargingStartedAtMs: null,
      activeStartedAtMs: null,
      activeEndedAtMs: null,
      releaseStartedAtMs: null,
      cooldownEndsAtMs: null,
    });

    return () => this.unregisterGesture(resolved.name);
  }

  unregisterGesture(
    name: string,
    timestampMs = this.lastTimestampMs ?? 0,
  ): boolean {
    const runtime = this.gestures.get(name);
    if (!runtime) return false;

    if (runtime.state === 'ACTIVE') {
      const events: GestureEvent[] = [];
      this.endActiveGesture(runtime, timestampMs, 'unregistered', events);
      this.dispatch(events);
    }
    this.gestures.delete(name);
    return true;
  }

  subscribe(listener: GestureEventListener): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  /** Advances every registered gesture using a shared tracking input/frame. */
  update(input: TInput, timestampMs: number): readonly GestureEvent[] {
    this.assertTimestamp(timestampMs);
    if (this.currentStatus === 'idle') this.initialize();
    this.currentStatus = 'running';
    const events: GestureEvent[] = [];
    const activeAtFrameStart = this.activeGestureName;
    const deltaMs =
      this.lastTimestampMs === null ? 0 : timestampMs - this.lastTimestampMs;

    try {
      const runtimes = this.getPrioritizedRuntimes();
      for (const runtime of runtimes) {
        this.updateConfidence(runtime, input, deltaMs);
        this.expireCooldown(runtime, timestampMs);
      }

      const active = this.getActiveRuntime();
      if (active) this.processActiveGesture(active, timestampMs, events);

      for (const runtime of runtimes) {
        if (runtime.state !== 'ACTIVE' && runtime.state !== 'COOLDOWN') {
          this.processCandidate(runtime, timestampMs);
        }
      }

      const candidate = runtimes.find(
        (runtime) =>
          runtime.state === 'CHARGING' &&
          runtime.chargingStartedAtMs !== null &&
          timestampMs - runtime.chargingStartedAtMs >=
            runtime.definition.minimumHoldDurationMs,
      );
      const currentActive = this.getActiveRuntime();

      if (
        candidate &&
        (!currentActive ||
          candidate.definition.priority > currentActive.definition.priority)
      ) {
        if (currentActive) {
          this.endActiveGesture(
            currentActive,
            timestampMs,
            'superseded',
            events,
          );
        }
        this.activateGesture(candidate, timestampMs, events);
      }

      const held = this.getActiveRuntime();
      if (
        held &&
        held.definition.name === activeAtFrameStart &&
        held.state === 'ACTIVE'
      ) {
        events.push(this.createEvent('gesture:hold', held, timestampMs));
      }

      this.lastTimestampMs = timestampMs;
      this.dispatch(events);
      return events;
    } finally {
      this.currentStatus = 'ready';
    }
  }

  cancelGesture(
    name: string,
    timestampMs = this.lastTimestampMs ?? 0,
  ): readonly GestureEvent[] {
    this.assertExternalTimestamp(timestampMs);
    const runtime = this.gestures.get(name);
    if (!runtime) return [];

    const events: GestureEvent[] = [];
    if (runtime.state === 'ACTIVE') {
      this.endActiveGesture(runtime, timestampMs, 'cancelled', events);
    } else if (runtime.state !== 'COOLDOWN') {
      this.resetRuntime(runtime);
    }
    this.dispatch(events);
    return events;
  }

  cancelAll(timestampMs = this.lastTimestampMs ?? 0): readonly GestureEvent[] {
    this.assertExternalTimestamp(timestampMs);
    const events: GestureEvent[] = [];
    for (const runtime of this.gestures.values()) {
      if (runtime.state === 'ACTIVE') {
        this.endActiveGesture(runtime, timestampMs, 'cancelled', events);
      } else if (runtime.state !== 'COOLDOWN') {
        this.resetRuntime(runtime);
      }
    }
    this.dispatch(events);
    return events;
  }

  getGesture(name: string, timestampMs = this.lastTimestampMs ?? 0): GestureSnapshot | null {
    const runtime = this.gestures.get(name);
    return runtime ? this.createSnapshot(runtime, timestampMs) : null;
  }

  getGestures(timestampMs = this.lastTimestampMs ?? 0): readonly GestureSnapshot[] {
    return this.getPrioritizedRuntimes().map((runtime) =>
      this.createSnapshot(runtime, timestampMs),
    );
  }

  dispose(timestampMs = this.lastTimestampMs ?? 0): void {
    const active = this.getActiveRuntime();
    if (active) {
      const events: GestureEvent[] = [];
      this.endActiveGesture(active, timestampMs, 'disposed', events);
      this.dispatch(events);
    }
    this.gestures.clear();
    this.listeners.clear();
    this.activeGestureName = null;
    this.lastTimestampMs = null;
    this.currentStatus = 'idle';
  }

  private updateConfidence(
    runtime: GestureRuntime<TInput>,
    input: TInput,
    deltaMs: number,
  ): void {
    const rawConfidence = clamp01(runtime.definition.evaluate(input));
    runtime.rawConfidence = rawConfidence;

    if (!runtime.hasConfidenceSample) {
      runtime.smoothedConfidence = rawConfidence;
      runtime.hasConfidenceSample = true;
      return;
    }

    const timeConstantMs = runtime.definition.smoothingTimeConstantMs;
    if (timeConstantMs === 0) {
      runtime.smoothedConfidence = rawConfidence;
      return;
    }

    // Time-aware exponential smoothing behaves consistently at different FPS.
    const alpha = 1 - Math.exp(-deltaMs / timeConstantMs);
    runtime.smoothedConfidence +=
      alpha * (rawConfidence - runtime.smoothedConfidence);
  }

  private processCandidate(
    runtime: GestureRuntime<TInput>,
    timestampMs: number,
  ): void {
    const { confidenceThreshold, releaseThreshold, debounceDurationMs } =
      runtime.definition;
    const isConfident =
      runtime.rawConfidence >= confidenceThreshold &&
      runtime.smoothedConfidence >= confidenceThreshold;
    const isReleased =
      runtime.rawConfidence < releaseThreshold ||
      runtime.smoothedConfidence < releaseThreshold;

    if (runtime.state === 'IDLE') {
      if (isConfident) {
        runtime.state = 'DETECTING';
        runtime.detectionStartedAtMs = timestampMs;
      }
      return;
    }

    if (isReleased) {
      this.resetRuntime(runtime);
      return;
    }

    if (
      runtime.state === 'DETECTING' &&
      runtime.detectionStartedAtMs !== null &&
      isConfident &&
      timestampMs - runtime.detectionStartedAtMs >= debounceDurationMs
    ) {
      runtime.state = 'CHARGING';
      runtime.chargingStartedAtMs = timestampMs;
    }
  }

  private processActiveGesture(
    runtime: GestureRuntime<TInput>,
    timestampMs: number,
    events: GestureEvent[],
  ): void {
    const isReleased =
      runtime.rawConfidence < runtime.definition.releaseThreshold ||
      runtime.smoothedConfidence < runtime.definition.releaseThreshold;

    if (!isReleased) {
      runtime.releaseStartedAtMs = null;
      return;
    }

    if (runtime.releaseStartedAtMs === null) {
      runtime.releaseStartedAtMs = timestampMs;
      return;
    }

    if (
      timestampMs - runtime.releaseStartedAtMs >=
      runtime.definition.debounceDurationMs
    ) {
      this.endActiveGesture(runtime, timestampMs, 'released', events);
    }
  }

  private activateGesture(
    runtime: GestureRuntime<TInput>,
    timestampMs: number,
    events: GestureEvent[],
  ): void {
    runtime.state = 'ACTIVE';
    runtime.activeStartedAtMs = timestampMs;
    runtime.activeEndedAtMs = null;
    runtime.releaseStartedAtMs = null;
    this.activeGestureName = runtime.definition.name;
    events.push(this.createEvent('gesture:start', runtime, timestampMs));
  }

  private endActiveGesture(
    runtime: GestureRuntime<TInput>,
    timestampMs: number,
    reason: GestureEndReason,
    events: GestureEvent[] = [],
  ): void {
    runtime.state = 'COOLDOWN';
    runtime.activeEndedAtMs = timestampMs;
    runtime.cooldownEndsAtMs =
      timestampMs + runtime.definition.cooldownDurationMs;
    runtime.releaseStartedAtMs = null;
    if (this.activeGestureName === runtime.definition.name) {
      this.activeGestureName = null;
    }
    events.push(this.createEvent('gesture:end', runtime, timestampMs, reason));
  }

  private expireCooldown(
    runtime: GestureRuntime<TInput>,
    timestampMs: number,
  ): void {
    if (
      runtime.state === 'COOLDOWN' &&
      runtime.cooldownEndsAtMs !== null &&
      timestampMs >= runtime.cooldownEndsAtMs
    ) {
      this.resetRuntime(runtime);
    }
  }

  private resetRuntime(runtime: GestureRuntime<TInput>): void {
    runtime.state = 'IDLE';
    runtime.detectionStartedAtMs = null;
    runtime.chargingStartedAtMs = null;
    runtime.activeStartedAtMs = null;
    runtime.activeEndedAtMs = null;
    runtime.releaseStartedAtMs = null;
    runtime.cooldownEndsAtMs = null;
  }

  private createSnapshot(
    runtime: GestureRuntime<TInput>,
    timestampMs: number,
  ): GestureSnapshot {
    const chargingDurationMs =
      runtime.state === 'CHARGING' && runtime.chargingStartedAtMs !== null
        ? Math.max(0, timestampMs - runtime.chargingStartedAtMs)
        : 0;
    const activeDurationMs =
      runtime.activeStartedAtMs === null
        ? 0
        : runtime.state === 'ACTIVE'
          ? Math.max(0, timestampMs - runtime.activeStartedAtMs)
          : runtime.state === 'COOLDOWN' && runtime.activeEndedAtMs !== null
            ? Math.max(0, runtime.activeEndedAtMs - runtime.activeStartedAtMs)
            : 0;

    return {
      name: runtime.definition.name,
      confidence: runtime.smoothedConfidence,
      rawConfidence: runtime.rawConfidence,
      state: runtime.state,
      holdProgress:
        runtime.state === 'ACTIVE'
          ? 1
          : runtime.state === 'CHARGING'
            ? clamp01(
                chargingDurationMs /
                  runtime.definition.minimumHoldDurationMs,
              )
            : 0,
      activeDurationMs,
      cooldownRemainingMs:
        runtime.state === 'COOLDOWN' && runtime.cooldownEndsAtMs !== null
          ? Math.max(0, runtime.cooldownEndsAtMs - timestampMs)
          : 0,
      priority: runtime.definition.priority,
    };
  }

  private createEvent(
    type: GestureEventType,
    runtime: GestureRuntime<TInput>,
    timestampMs: number,
    reason?: GestureEndReason,
  ): GestureEvent {
    return {
      type,
      timestampMs,
      gesture: this.createSnapshot(runtime, timestampMs),
      ...(reason ? { reason } : {}),
    };
  }

  private dispatch(events: readonly GestureEvent[]): void {
    for (const event of events) {
      for (const listener of this.listeners) listener(event);
    }
  }

  private getActiveRuntime(): GestureRuntime<TInput> | null {
    return this.activeGestureName
      ? (this.gestures.get(this.activeGestureName) ?? null)
      : null;
  }

  private getPrioritizedRuntimes(): GestureRuntime<TInput>[] {
    return [...this.gestures.values()].sort(
      (left, right) =>
        right.definition.priority - left.definition.priority ||
        left.registrationOrder - right.registrationOrder,
    );
  }

  private assertTimestamp(timestampMs: number): void {
    if (!Number.isFinite(timestampMs)) {
      throw new RangeError('timestampMs must be finite.');
    }
    if (this.lastTimestampMs !== null && timestampMs < this.lastTimestampMs) {
      throw new RangeError('timestampMs must be monotonic.');
    }
  }

  private assertExternalTimestamp(timestampMs: number): void {
    if (!Number.isFinite(timestampMs)) {
      throw new RangeError('timestampMs must be finite.');
    }
  }
}
