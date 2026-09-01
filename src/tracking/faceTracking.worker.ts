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
      return 'GPU';
    } catch {
      landmarker?.close();
      landmarker = null;
    }
  }

  landmarker = await initialize('CPU');
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

  try {
    const startedAt = performance.now();
    const result = landmarker.detectForVideo(frame, timestamp);
    const inferenceTimeMs = performance.now() - startedAt;
    const face = result.faceLandmarks[0];
    const detectedLandmarkCount = face?.length ?? 0;
    const landmarkCount = outputLandmarks ? detectedLandmarkCount : 0;
    const landmarks = new Float32Array(landmarkCount * 3);
    const keyLandmarks = new Float32Array(
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
    const transformationMatrix = new Float32Array(matrixData);
    post(
      {
        type: 'result',
        runToken,
        timestamp,
        inferenceTimeMs,
        faceDetected: detectedLandmarkCount > 0,
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

  landmarker?.close();
  landmarker = null;
  workerScope.close();
};
