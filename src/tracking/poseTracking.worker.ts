/// <reference lib="webworker" />

import { FilesetResolver, PoseLandmarker } from '@mediapipe/tasks-vision';
import type {
  PoseTrackingWorkerRequest,
  PoseTrackingWorkerResponse,
} from './poseTracking.worker.types';
import type { MediaPipeDelegate } from './handTracking.worker.types';

const workerScope: DedicatedWorkerGlobalScope = self as DedicatedWorkerGlobalScope;
let landmarker: PoseLandmarker | null = null;
let outputLandmarks = false;

// Nose, ears, shoulders, elbows, wrists, and hips. These are sufficient for
// upper-body effects and gesture input without transferring all 33 points.
const KEY_LANDMARK_INDICES = [0, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24] as const;
const LANDMARK_STRIDE = 5;
const WORLD_LANDMARK_STRIDE = 3;

const loadPublicModule = Function(
  'url',
  'return import(url)',
) as (url: string) => Promise<unknown>;
(workerScope as DedicatedWorkerGlobalScope & {
  import: (url: string) => Promise<unknown>;
}).import = loadPublicModule;

function post(message: PoseTrackingWorkerResponse, transfer?: Transferable[]) {
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

async function createLandmarker(
  wasmPath: string,
  modelPath: string,
): Promise<MediaPipeDelegate> {
  const vision = await FilesetResolver.forVisionTasks(wasmPath, true);
  const initialize = (delegate: MediaPipeDelegate) =>
    PoseLandmarker.createFromOptions(vision, {
      runningMode: 'VIDEO',
      numPoses: 1,
      minPoseDetectionConfidence: 0.5,
      minPosePresenceConfidence: 0.5,
      minTrackingConfidence: 0.5,
      outputSegmentationMasks: false,
      baseOptions: { modelAssetPath: modelPath, delegate },
    });

  if (supportsHardwareGpu()) {
    try {
      landmarker = await initialize('GPU');
      return 'GPU';
    } catch {
      landmarker?.close();
      landmarker = null;
    }
  }

  landmarker = await initialize('CPU');
  return 'CPU';
}

function packNormalizedLandmark(
  target: Float32Array,
  targetOffset: number,
  landmark: {
    x: number;
    y: number;
    z: number;
    visibility?: number;
    presence?: number;
  } | undefined,
): void {
  target[targetOffset] = landmark?.x ?? 0;
  target[targetOffset + 1] = landmark?.y ?? 0;
  target[targetOffset + 2] = landmark?.z ?? 0;
  target[targetOffset + 3] = Number.isFinite(landmark?.visibility)
    ? (landmark?.visibility ?? -1)
    : -1;
  target[targetOffset + 4] = Number.isFinite(landmark?.presence)
    ? (landmark?.presence ?? -1)
    : -1;
}

function processFrame(
  frame: ImageBitmap | VideoFrame,
  timestamp: number,
  runToken: number,
): void {
  if (!landmarker) {
    frame.close();
    post({
      type: 'error',
      message: 'Pose Landmarker is not initialized.',
      runToken,
    });
    return;
  }

  try {
    const startedAt = performance.now();
    const result = landmarker.detectForVideo(frame, timestamp);
    const inferenceTimeMs = performance.now() - startedAt;
    const pose = result.landmarks[0];
    const worldPose = result.worldLandmarks[0];
    const detectedLandmarkCount = pose?.length ?? 0;
    const poseDetected = detectedLandmarkCount >= 25;
    const keyLandmarks = new Float32Array(
      poseDetected ? KEY_LANDMARK_INDICES.length * LANDMARK_STRIDE : 0,
    );
    const keyWorldLandmarks = new Float32Array(
      poseDetected ? KEY_LANDMARK_INDICES.length * WORLD_LANDMARK_STRIDE : 0,
    );
    const landmarkCount = outputLandmarks && poseDetected
      ? detectedLandmarkCount
      : 0;
    const landmarks = new Float32Array(landmarkCount * LANDMARK_STRIDE);

    for (let keyIndex = 0; keyIndex < KEY_LANDMARK_INDICES.length && poseDetected; keyIndex += 1) {
      const sourceIndex = KEY_LANDMARK_INDICES[keyIndex];
      packNormalizedLandmark(
        keyLandmarks,
        keyIndex * LANDMARK_STRIDE,
        pose[sourceIndex],
      );
      const worldLandmark = worldPose?.[sourceIndex];
      const worldOffset = keyIndex * WORLD_LANDMARK_STRIDE;
      keyWorldLandmarks[worldOffset] = worldLandmark?.x ?? 0;
      keyWorldLandmarks[worldOffset + 1] = worldLandmark?.y ?? 0;
      keyWorldLandmarks[worldOffset + 2] = worldLandmark?.z ?? 0;
    }

    for (let index = 0; index < landmarkCount; index += 1) {
      packNormalizedLandmark(
        landmarks,
        index * LANDMARK_STRIDE,
        pose[index],
      );
    }

    post(
      {
        type: 'result',
        runToken,
        timestamp,
        inferenceTimeMs,
        poseDetected,
        keyLandmarks,
        keyWorldLandmarks,
        landmarks,
        landmarkCount,
      },
      [keyLandmarks.buffer, keyWorldLandmarks.buffer, landmarks.buffer],
    );
  } catch (error) {
    post({
      type: 'error',
      runToken,
      message: error instanceof Error ? error.message : 'Pose inference failed.',
    });
  } finally {
    frame.close();
  }
}

workerScope.onmessage = (event: MessageEvent<PoseTrackingWorkerRequest>) => {
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
              : 'Unable to initialize Pose Landmarker.',
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

  landmarker?.close();
  landmarker = null;
  workerScope.close();
};
