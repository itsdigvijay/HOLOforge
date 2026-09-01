/// <reference lib="webworker" />

import { FilesetResolver, HandLandmarker } from '@mediapipe/tasks-vision';
import type {
  HandTrackingWorkerRequest,
  HandTrackingWorkerResponse,
  MediaPipeDelegate,
  MediaPipeTrackingConfig,
  WorkerResultMessage,
} from './handTracking.worker.types';

const workerScope: DedicatedWorkerGlobalScope = self as DedicatedWorkerGlobalScope;
let landmarker: HandLandmarker | null = null;
let trackingConfig: MediaPipeTrackingConfig = {
  minHandDetectionConfidence: 0.5,
  minHandPresenceConfidence: 0.5,
  minTrackingConfidence: 0.5,
};
let blurCanvas: OffscreenCanvas | null = null;
let blurContext: OffscreenCanvasRenderingContext2D | null = null;
let lastBlurMeasuredAt = -Infinity;
let motionBlurScore = -1;

// MediaPipe falls back to a dynamic import when importScripts is unavailable in
// a module worker. Supplying this hook keeps Vite from rewriting public WASM
// loader URLs as source imports during development.
const loadPublicModule = Function(
  'url',
  'return import(url)',
) as (url: string) => Promise<unknown>;
(workerScope as DedicatedWorkerGlobalScope & {
  import: (url: string) => Promise<unknown>;
}).import = loadPublicModule;

function post(message: HandTrackingWorkerResponse, transfer?: Transferable[]) {
  workerScope.postMessage(message, transfer ?? []);
}

function supportsHardwareGpu(): boolean {
  if (typeof OffscreenCanvas === 'undefined') return false;

  try {
    const canvas = new OffscreenCanvas(1, 1);
    const context =
      canvas.getContext('webgl2') ?? canvas.getContext('webgl');
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

function pixelLuminance(pixels: Uint8ClampedArray, offset: number): number {
  return (
    pixels[offset] * 0.299 +
    pixels[offset + 1] * 0.587 +
    pixels[offset + 2] * 0.114
  );
}

function measureMotionBlur(
  frame: ImageBitmap | VideoFrame,
  timestamp: number,
): number {
  if (timestamp - lastBlurMeasuredAt < 500) return motionBlurScore;
  lastBlurMeasuredAt = timestamp;

  try {
    blurCanvas ??= new OffscreenCanvas(64, 48);
    blurContext ??= blurCanvas.getContext('2d', {
      willReadFrequently: true,
    });
    if (!blurContext) return motionBlurScore;

    blurContext.drawImage(frame, 0, 0, 64, 48);
    const pixels = blurContext.getImageData(0, 0, 64, 48).data;
    let sum = 0;
    let squaredSum = 0;
    let samples = 0;
    for (let y = 1; y < 47; y += 2) {
      for (let x = 1; x < 63; x += 2) {
        const center = (y * 64 + x) * 4;
        const left = center - 4;
        const right = center + 4;
        const above = center - 64 * 4;
        const below = center + 64 * 4;
        const laplacian =
          pixelLuminance(pixels, left) +
          pixelLuminance(pixels, right) +
          pixelLuminance(pixels, above) +
          pixelLuminance(pixels, below) -
          4 * pixelLuminance(pixels, center);
        sum += laplacian;
        squaredSum += laplacian * laplacian;
        samples += 1;
      }
    }
    const mean = sum / Math.max(1, samples);
    motionBlurScore = squaredSum / Math.max(1, samples) - mean * mean;
  } catch {
    // Blur diagnostics must never interfere with hand inference.
  }
  return motionBlurScore;
}

async function createLandmarker(
  wasmPath: string,
  modelPath: string,
): Promise<MediaPipeDelegate> {
  // Module workers need MediaPipe's ES module loader, which explicitly exports
  // its factory on globalThis after the dynamic import completes.
  const vision = await FilesetResolver.forVisionTasks(wasmPath, true);
  const initializeDelegate = (delegate: MediaPipeDelegate) =>
    HandLandmarker.createFromOptions(vision, {
      runningMode: 'VIDEO',
      numHands: 2,
      ...trackingConfig,
      baseOptions: { modelAssetPath: modelPath, delegate },
    });

  if (supportsHardwareGpu()) {
    try {
      landmarker = await initializeDelegate('GPU');
      return 'GPU';
    } catch {
      landmarker?.close();
      landmarker = null;
    }
  }

  landmarker = await initializeDelegate('CPU');
  return 'CPU';
}

function processFrame(
  frame: ImageBitmap | VideoFrame,
  timestamp: number,
  runToken: number,
) {
  if (!landmarker) {
    frame.close();
    post({ type: 'error', message: 'Hand Landmarker is not initialized.', runToken });
    return;
  }

  try {
    const startedAt = performance.now();
    const result = landmarker.detectForVideo(frame, timestamp);
    const inferenceTimeMs = performance.now() - startedAt;
    const currentBlurScore = measureMotionBlur(frame, timestamp);
    const handCount = Math.min(result.landmarks.length, 2);
    const landmarks = new Float32Array(handCount * 21 * 4);
    const handedness: WorkerResultMessage['handedness'] = [];

    for (let handIndex = 0; handIndex < handCount; handIndex += 1) {
      const handLandmarks = result.landmarks[handIndex];
      const classification = result.handedness[handIndex]?.[0];
      handedness.push({
        categoryName: classification?.categoryName ?? 'Unknown',
        score: classification?.score ?? 0,
      });

      for (let landmarkIndex = 0; landmarkIndex < 21; landmarkIndex += 1) {
        const landmark = handLandmarks[landmarkIndex];
        const offset = (handIndex * 21 + landmarkIndex) * 4;
        landmarks[offset] = landmark?.x ?? 0;
        landmarks[offset + 1] = landmark?.y ?? 0;
        landmarks[offset + 2] = landmark?.z ?? 0;
        landmarks[offset + 3] = landmark?.visibility ?? -1;
      }
    }

    post(
      {
        type: 'result',
        runToken,
        timestamp,
        inferenceTimeMs,
        handCount,
        landmarks,
        handedness,
        motionBlurScore: currentBlurScore,
      },
      [landmarks.buffer],
    );
  } catch (error) {
    post({
      type: 'error',
      runToken,
      message: error instanceof Error ? error.message : 'Hand inference failed.',
    });
  } finally {
    frame.close();
  }
}

workerScope.onmessage = (event: MessageEvent<HandTrackingWorkerRequest>) => {
  const message = event.data;

  if (message.type === 'init') {
    trackingConfig = { ...message.config };
    void createLandmarker(message.wasmPath, message.modelPath)
      .then((delegate) => post({ type: 'ready', delegate }))
      .catch((error: unknown) =>
        post({
          type: 'error',
          message:
            error instanceof Error
              ? error.message
              : 'Unable to initialize Hand Landmarker.',
        }),
      );
    return;
  }

  if (message.type === 'frame') {
    processFrame(message.frame, message.timestamp, message.runToken);
    return;
  }


  if (message.type === 'configure') {
    if (!landmarker) {
      post({
        type: 'error',
        revision: message.revision,
        message: 'Hand Landmarker is not initialized.',
      });
      return;
    }
    const revision = message.revision;
    void landmarker
      .setOptions(message.config)
      .then(() => {
        trackingConfig = { ...message.config };
        post({ type: 'configured', revision });
      })
      .catch((error: unknown) =>
        post({
          type: 'error',
          revision,
          message:
            error instanceof Error
              ? error.message
              : 'Unable to update tracking confidence.',
        }),
      );
    return;
  }

  landmarker?.close();
  landmarker = null;
  workerScope.close();
};
