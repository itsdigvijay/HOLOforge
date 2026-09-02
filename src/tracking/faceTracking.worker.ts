/// <reference lib="webworker" />

import { FaceLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';
import type {
  FaceTrackingWorkerRequest,
  FaceTrackingWorkerResponse,
} from './faceTracking.worker.types';
import type { MediaPipeDelegate } from './handTracking.worker.types';

const workerScope: DedicatedWorkerGlobalScope = self as DedicatedWorkerGlobalScope;
let landmarker: FaceLandmarker | null = null;
let outputLandmarks = false;
const KEY_LANDMARK_INDICES = [473, 468, 454, 234, 10, 152, 1] as const;

// Guards against MediaPipe's VIDEO-mode requirement that timestamps passed to
// detectForVideo() strictly increase. Out-of-order postMessage delivery (rare
// but possible under load) would otherwise throw and kill the worker.
let lastTimestamp: number | null = null;

const loadPublicModule = Function(
  'url',
  'return import(url)',
) as (url: string) => Promise<unknown>;
(workerScope as DedicatedWorkerGlobalScope & {
  import: (url: string) => Promise<unknown>;
}).import = loadPublicModule;

function post(message: FaceTrackingWorkerResponse, transfer?: Transferable[]) {
  workerScope.postMessage(message, transfer ?? []);
}

function supportsHardwareGpu(): boolean {
  if (typeof OffscreenCanvas === 'undefined') return false;
  try {
    const canvas = new OffscreenCanvas(1, 1);
    const context = canvas.getContext('webgl2') ?? canvas.getContext('webgl');
    if (!context) return false;
    const debugInfo = context.getExtension('WEBGL_debug_renderer_info');
    const renderer = debugInfo
      ? String(context.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL))
      : '';
    return !/swiftshader|llvmpipe|software/i.test(renderer);
  } catch {
    return false;
  }
}

// --- Jitter smoothing (One Euro Filter) -----------------------------------
// Raw landmark/matrix output from the model is noisy frame-to-frame even
// when the face is still, which shows up as visible shake in any overlay
// driven by this data. One Euro adapts its cutoff to signal speed: it stays
// tight (low lag) during fast motion and heavier (low jitter) when still,
// which is why it's the standard choice for this over a plain EMA.
// Tune minCutoff/beta per-signal below if you still see jitter or feel lag.
class OneEuroFilter {
  private xPrev: number | null = null;
  private dxPrev = 0;
  private tPrev: number | null = null;

  constructor(
    private minCutoff = 1.0,
    private beta = 0.007,
    private dCutoff = 1.0,
  ) {}

  private alpha(cutoff: number, dt: number): number {
    const tau = 1 / (2 * Math.PI * cutoff);
    return 1 / (1 + tau / dt);
  }

  filter(x: number, tMs: number): number {
    if (this.tPrev === null) {
      this.tPrev = tMs;
      this.xPrev = x;
      return x;
    }
    const dt = Math.max((tMs - this.tPrev) / 1000, 1e-3);
    this.tPrev = tMs;

    const prevX = this.xPrev ?? x;
    const dx = (x - prevX) / dt;
    const aD = this.alpha(this.dCutoff, dt);
    const dxHat = aD * dx + (1 - aD) * this.dxPrev;
    this.dxPrev = dxHat;

    const cutoff = this.minCutoff + this.beta * Math.abs(dxHat);
    const a = this.alpha(cutoff, dt);
    const xHat = a * x + (1 - a) * prevX;
    this.xPrev = xHat;
    return xHat;
  }
}

class VectorSmoother {
  private filters: OneEuroFilter[] = [];

  constructor(
    private minCutoff: number,
    private beta: number,
    private dCutoff = 1.0,
  ) {}

  filter(values: Float32Array, tMs: number): Float32Array {
    if (this.filters.length !== values.length) {
      this.filters = Array.from(
        { length: values.length },
        () => new OneEuroFilter(this.minCutoff, this.beta, this.dCutoff),
      );
    }
    const out = new Float32Array(values.length);
    for (let i = 0; i < values.length; i += 1) {
      out[i] = this.filters[i].filter(values[i], tMs);
    }
    return out;
  }

  reset() {
    this.filters = [];
  }
}

// Normalized landmark coords are small (0-1) and slow-moving -> low beta.
// The transform matrix carries translation in larger units and needs to
// track head motion more responsively -> higher beta so it doesn't lag.
const keyLandmarkSmoother = new VectorSmoother(1.0, 0.02);
const landmarkSmoother = new VectorSmoother(1.0, 0.02);
const matrixSmoother = new VectorSmoother(1.0, 15);

function resetSmoothers() {
  keyLandmarkSmoother.reset();
  landmarkSmoother.reset();
  matrixSmoother.reset();
}

async function createLandmarker(
  wasmPath: string,
  modelPath: string,
): Promise<MediaPipeDelegate> {
  const vision = await FilesetResolver.forVisionTasks(wasmPath, true);
  const initialize = (delegate: MediaPipeDelegate) =>
    FaceLandmarker.createFromOptions(vision, {
      runningMode: 'VIDEO',
      numFaces: 1,
      minFaceDetectionConfidence: 0.5,
      minFacePresenceConfidence: 0.5,
      minTrackingConfidence: 0.5,
      outputFaceBlendshapes: false,
      outputFacialTransformationMatrixes: true,
      baseOptions: { modelAssetPath: modelPath, delegate },
    });

  if (supportsHardwareGpu()) {
    try {
      landmarker = await initialize('GPU');
      lastTimestamp = null;
      resetSmoothers();
      return 'GPU';
    } catch {
      landmarker?.close();
      landmarker = null;
    }
  }

  landmarker = await initialize('CPU');
  lastTimestamp = null;
  resetSmoothers();
  return 'CPU';
}

function processFrame(
  frame: ImageBitmap | VideoFrame,
  timestamp: number,
  runToken: number,
) {
  if (!landmarker) {
    frame.close();
    post({
      type: 'error',
      message: 'Face Landmarker is not initialized.',
      runToken,
    });
    return;
  }

  // Drop non-increasing timestamps instead of letting detectForVideo throw.
  if (lastTimestamp !== null && timestamp <= lastTimestamp) {
    frame.close();
    return;
  }
  lastTimestamp = timestamp;

  try {
    const startedAt = performance.now();
    const result = landmarker.detectForVideo(frame, timestamp);
    const inferenceTimeMs = performance.now() - startedAt;
    const face = result.faceLandmarks[0];
    const detectedLandmarkCount = face?.length ?? 0;
    const faceDetected = detectedLandmarkCount > 0;

    if (!faceDetected) {
      // Don't let filters blend stale state into the next detection -
      // otherwise re-acquiring the face produces a visible slide-in.
      resetSmoothers();
    }

    const landmarkCount = outputLandmarks ? detectedLandmarkCount : 0;
    let landmarks = new Float32Array(landmarkCount * 3);
    let keyLandmarks = new Float32Array(
      detectedLandmarkCount > 0 ? KEY_LANDMARK_INDICES.length * 3 : 0,
    );
    let visibilityTotal = 0;
    let visibilityCount = 0;

    for (let index = 0; index < detectedLandmarkCount; index += 1) {
      const landmark = face[index];
      const visibility = landmark?.visibility;
      if (outputLandmarks) {
        const offset = index * 3;
        landmarks[offset] = landmark?.x ?? 0;
        landmarks[offset + 1] = landmark?.y ?? 0;
        landmarks[offset + 2] = landmark?.z ?? 0;
      }
      if (Number.isFinite(visibility) && visibility > 0) {
        visibilityTotal += visibility;
        visibilityCount += 1;
      }
    }

    for (let keyIndex = 0; keyIndex < keyLandmarks.length / 3; keyIndex += 1) {
      const landmark = face[KEY_LANDMARK_INDICES[keyIndex]];
      const offset = keyIndex * 3;
      keyLandmarks[offset] = landmark?.x ?? 0;
      keyLandmarks[offset + 1] = landmark?.y ?? 0;
      keyLandmarks[offset + 2] = landmark?.z ?? 0;
    }

    const matrixData = result.facialTransformationMatrixes[0]?.data ?? [];
    let transformationMatrix = new Float32Array(matrixData);

    if (faceDetected) {
      keyLandmarks = keyLandmarkSmoother.filter(keyLandmarks, timestamp);
      if (outputLandmarks && landmarks.length > 0) {
        landmarks = landmarkSmoother.filter(landmarks, timestamp);
      }
      if (transformationMatrix.length > 0) {
        transformationMatrix = matrixSmoother.filter(
          transformationMatrix,
          timestamp,
        );
      }
    }

    post(
      {
        type: 'result',
        runToken,
        timestamp,
        inferenceTimeMs,
        faceDetected,
        keyLandmarks,
        landmarkCount,
        landmarks,
        transformationMatrix,
        confidence:
          visibilityCount > 0 ? visibilityTotal / visibilityCount : -1,
      },
      [keyLandmarks.buffer, landmarks.buffer, transformationMatrix.buffer],
    );
  } catch (error) {
    post({
      type: 'error',
      runToken,
      message:
        error instanceof Error ? error.message : 'Face inference failed.',
    });
  } finally {
    frame.close();
  }
}

workerScope.onmessage = (event: MessageEvent<FaceTrackingWorkerRequest>) => {
  const message = event.data;

  if (message.type === 'init') {
    outputLandmarks = message.outputLandmarks;
    void createLandmarker(message.wasmPath, message.modelPath)
      .then((delegate) => post({ type: 'ready', delegate }))
      .catch((error: unknown) =>
        post({
          type: 'error',
          message:
            error instanceof Error
              ? error.message
              : 'Unable to initialize Face Landmarker.',
        }),
      );
    return;
  }

  if (message.type === 'setLandmarkOutput') {
    outputLandmarks = message.enabled;
    return;
  }

  if (message.type === 'frame') {
    processFrame(message.frame, message.timestamp, message.runToken);
    return;
  }

  try {
    landmarker?.close();
  } catch {
    // Already torn down or never fully initialized - safe to ignore.
  }
  landmarker = null;
  workerScope.close();
};
