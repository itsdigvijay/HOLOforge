import { useEffect, useRef, type RefObject } from 'react';
import type { OverlayDetail } from '../settings';
import {
  type FaceTrackingManager,
  HandTrackingManager,
  type HandLandmarkPoint,
  POSE_LANDMARK_STRIDE,
  PoseTrackingManager,
  type TrackedHand,
} from '../tracking';

export type LandmarkViewMode = 'smoothed' | 'raw' | 'compare';

interface HandLandmarkOverlayProps {
  handEnabled: boolean;
  faceEnabled: boolean;
  poseEnabled: boolean;
  manager: HandTrackingManager;
  faceManager: FaceTrackingManager;
  poseManager: PoseTrackingManager;
  videoRef: RefObject<HTMLVideoElement | null>;
  viewMode: LandmarkViewMode;
  detail: OverlayDetail;
  pixelRatioCap: 1 | 1.5 | 2;
  targetInferenceFps: number;
  fastMotionDebug: boolean;
}

interface RenderHandState {
  key: string;
  sourceTimestamp: number;
  transitionStartedAt: number;
  transitionDuration: number;
  initialized: boolean;
  start: Float32Array;
  current: Float32Array;
  target: Float32Array;
}

const LANDMARK_COUNT = 21;
const FOCUSED_LANDMARKS = [0, 4, 8] as const;
const TAU = Math.PI * 2;

function createRenderHandState(): RenderHandState {
  return {
    key: '',
    sourceTimestamp: -1,
    transitionStartedAt: 0,
    transitionDuration: 0,
    initialized: false,
    start: new Float32Array(LANDMARK_COUNT * 2),
    current: new Float32Array(LANDMARK_COUNT * 2),
    target: new Float32Array(LANDMARK_COUNT * 2),
  };
}

function copyLandmarksToBuffer(
  landmarks: readonly HandLandmarkPoint[],
  target: Float32Array,
) {
  for (let index = 0; index < LANDMARK_COUNT; index += 1) {
    const point = landmarks[index];
    const offset = index * 2;
    target[offset] = point?.x ?? 0;
    target[offset + 1] = point?.y ?? 0;
  }
}

export function HandLandmarkOverlay({
  handEnabled,
  faceEnabled,
  poseEnabled,
  manager,
  faceManager,
  poseManager,
  videoRef,
  viewMode,
  detail,
  pixelRatioCap,
  targetInferenceFps,
  fastMotionDebug,
}: HandLandmarkOverlayProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const context = canvas.getContext('2d', { alpha: true });
    if (!context) return;

    let animationFrameId = 0;
    let cssWidth = 0;
    let cssHeight = 0;
    let pixelRatio = 1;
    const renderHands = [createRenderHandState(), createRenderHandState()];

    // Backing-store changes clear and reallocate the canvas, so keep them
    // entirely outside the frame loop.
    const resizeCanvas = () => {
      const bounds = canvas.getBoundingClientRect();
      cssWidth = Math.max(1, Math.round(bounds.width));
      cssHeight = Math.max(1, Math.round(bounds.height));
      pixelRatio = Math.min(window.devicePixelRatio || 1, pixelRatioCap);
      const backingWidth = Math.round(cssWidth * pixelRatio);
      const backingHeight = Math.round(cssHeight * pixelRatio);
      if (canvas.width !== backingWidth) canvas.width = backingWidth;
      if (canvas.height !== backingHeight) canvas.height = backingHeight;
      context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
    };

    const resizeObserver = new ResizeObserver(resizeCanvas);
    resizeObserver.observe(canvas);
    window.addEventListener('resize', resizeCanvas, { passive: true });
    resizeCanvas();

    const updateRenderState = (
      state: RenderHandState,
      hand: TrackedHand,
      renderTimestamp: number,
    ) => {
      if (state.key !== hand.handedness) {
        state.key = hand.handedness;
        state.initialized = false;
        state.sourceTimestamp = -1;
      }
      if (state.sourceTimestamp === hand.timestamp) return;

      if (state.initialized) state.start.set(state.current);
      copyLandmarksToBuffer(hand.landmarks, state.target);
      state.sourceTimestamp = hand.timestamp;
      state.transitionStartedAt = renderTimestamp;
      // Blend over only part of an inference interval. This removes 30 FPS
      // stepping without adding a full-frame delay to fast motion.
      state.transitionDuration = Math.min(
        1000 / 60,
        500 /
          Math.max(
            1,
            manager.snapshot.targetInferenceFps || targetInferenceFps,
          ),
      );

      if (
        hand.fastMotionBlend >= 0.7 ||
        hand.reacquired ||
        hand.largeMovement
      ) {
        state.transitionDuration = 0;
        state.current.set(state.target);
        state.start.set(state.target);
      } else {
        state.transitionDuration *= 1 - hand.fastMotionBlend;
      }

      if (!state.initialized) {
        state.current.set(state.target);
        state.start.set(state.target);
        state.initialized = true;
      }
    };

    const interpolateRenderState = (
      state: RenderHandState,
      renderTimestamp: number,
    ) => {
      if (state.transitionDuration <= 0) {
        state.current.set(state.target);
        return;
      }
      const progress = Math.min(
        1,
        Math.max(
          0,
          (renderTimestamp - state.transitionStartedAt) /
            Math.max(1, state.transitionDuration),
        ),
      );
      for (let index = 0; index < state.current.length; index += 1) {
        state.current[index] =
          state.start[index] +
          (state.target[index] - state.start[index]) * progress;
      }
    };

    const draw = (renderTimestamp: number) => {
      manager.recordRenderFrame(renderTimestamp);
      context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
      context.clearRect(0, 0, cssWidth, cssHeight);

      const video = videoRef.current;
      if (
        (handEnabled || faceEnabled || poseEnabled) &&
        video &&
        video.videoWidth > 0 &&
        cssWidth > 0
      ) {
        const snapshot = manager.snapshot;
        const faceSnapshot = faceManager.snapshot;
        const poseSnapshot = poseManager.snapshot;
        const videoWidth = video.videoWidth;
        const videoHeight = video.videoHeight;
        const scale = Math.max(cssWidth / videoWidth, cssHeight / videoHeight);
        const renderedWidth = videoWidth * scale;
        const renderedHeight = videoHeight * scale;
        const offsetX = (cssWidth - renderedWidth) / 2;
        const offsetY = (cssHeight - renderedHeight) / 2;
        const projectX = (x: number) =>
          cssWidth - (offsetX + x * renderedWidth);
        const projectY = (y: number) => offsetY + y * renderedHeight;

        const drawBufferedSkeleton = (
          points: Float32Array,
          color: string,
          alpha: number,
          radius: number,
        ) => {
          context.strokeStyle = color;
          context.fillStyle = color;
          context.globalAlpha = alpha;
          context.lineWidth = 1.5;
          context.setLineDash([]);
          for (const connection of HandTrackingManager.HAND_CONNECTIONS) {
            const startOffset = connection.start * 2;
            const endOffset = connection.end * 2;
            context.beginPath();
            context.moveTo(
              projectX(points[startOffset]),
              projectY(points[startOffset + 1]),
            );
            context.lineTo(
              projectX(points[endOffset]),
              projectY(points[endOffset + 1]),
            );
            context.stroke();
          }
          for (let index = 0; index < LANDMARK_COUNT; index += 1) {
            const offset = index * 2;
            context.beginPath();
            context.arc(
              projectX(points[offset]),
              projectY(points[offset + 1]),
              radius,
              0,
              TAU,
            );
            context.fill();
          }
        };

        const drawDirectSkeleton = (
          points: readonly HandLandmarkPoint[],
          color: string,
          alpha: number,
          radius: number,
          dashed: boolean,
        ) => {
          context.strokeStyle = color;
          context.fillStyle = color;
          context.globalAlpha = alpha;
          context.lineWidth = 1;
          context.setLineDash(dashed ? [3, 4] : []);
          for (const connection of HandTrackingManager.HAND_CONNECTIONS) {
            const start = points[connection.start];
            const end = points[connection.end];
            context.beginPath();
            context.moveTo(projectX(start.x), projectY(start.y));
            context.lineTo(projectX(end.x), projectY(end.y));
            context.stroke();
          }
          context.setLineDash([]);
          for (let index = 0; index < LANDMARK_COUNT; index += 1) {
            const point = points[index];
            context.beginPath();
            context.arc(projectX(point.x), projectY(point.y), radius, 0, TAU);
            context.fill();
          }
        };

        if (
          poseEnabled &&
          poseSnapshot.poseDetected &&
          poseSnapshot.landmarkCount > 0
        ) {
          const landmarks = poseSnapshot.landmarks;
          const visible = (index: number) => {
            const visibility =
              landmarks[index * POSE_LANDMARK_STRIDE + 3] ?? -1;
            return visibility < 0 || visibility >= 0.35;
          };
          context.globalAlpha = 0.8;
          context.strokeStyle = '#69d7ff';
          context.fillStyle = '#a6e9ff';
          context.lineWidth = 2;
          context.setLineDash([]);
          for (const connection of PoseTrackingManager.POSE_CONNECTIONS) {
            if (!visible(connection.start) || !visible(connection.end)) continue;
            const startOffset = connection.start * POSE_LANDMARK_STRIDE;
            const endOffset = connection.end * POSE_LANDMARK_STRIDE;
            context.beginPath();
            context.moveTo(
              projectX(landmarks[startOffset]),
              projectY(landmarks[startOffset + 1]),
            );
            context.lineTo(
              projectX(landmarks[endOffset]),
              projectY(landmarks[endOffset + 1]),
            );
            context.stroke();
          }
          for (let index = 0; index < poseSnapshot.landmarkCount; index += 1) {
            if (!visible(index)) continue;
            const offset = index * POSE_LANDMARK_STRIDE;
            context.beginPath();
            context.arc(
              projectX(landmarks[offset]),
              projectY(landmarks[offset + 1]),
              index === 0 || (index >= 11 && index <= 16) ? 3.4 : 2.2,
              0,
              TAU,
            );
            context.fill();
          }

          const chest = poseSnapshot.chestMidpoint;
          if (chest) {
            const chestX = projectX(chest.x);
            const chestY = projectY(chest.y);
            context.globalAlpha = 0.95;
            context.strokeStyle = '#e1f8ff';
            context.beginPath();
            context.arc(chestX, chestY, 7, 0, TAU);
            context.stroke();

            if (detail === 'full') {
              const orientation = poseSnapshot.upperBodyOrientation;
              const confidence = poseSnapshot.confidence === null
                ? ''
                : ` · ${(poseSnapshot.confidence * 100).toFixed(0)}%`;
              const label = orientation
                ? `POSE · ${orientation.facing.toUpperCase()}${confidence} · Y ${orientation.yaw.toFixed(0)}° R ${orientation.roll.toFixed(0)}°`
                : `POSE${confidence}`;
              context.font = '600 11px Inter, sans-serif';
              const labelWidth = context.measureText(label).width + 12;
              const labelX = Math.max(
                6,
                Math.min(cssWidth - labelWidth - 6, chestX - labelWidth * 0.5),
              );
              const labelY = Math.max(22, chestY - 14);
              context.fillStyle = 'rgba(3, 7, 8, 0.78)';
              context.fillRect(labelX, labelY - 15, labelWidth, 20);
              context.fillStyle = '#a6e9ff';
              context.fillText(label, labelX + 6, labelY);
            }
          }
          context.globalAlpha = 1;
        }

        for (
          let handIndex = 0;
          handEnabled && handIndex < snapshot.hands.length;
          handIndex += 1
        ) {
          const hand = snapshot.hands[handIndex];
          const state = renderHands[handIndex];
          const color = hand.handedness === 'Left' ? '#72f5e7' : '#b994ff';
          const showRaw = viewMode === 'raw' || viewMode === 'compare';
          const showSmoothed =
            viewMode === 'smoothed' || viewMode === 'compare';

          updateRenderState(state, hand, renderTimestamp);
          interpolateRenderState(state, renderTimestamp);

          if (showRaw) {
            drawDirectSkeleton(
              hand.rawLandmarks,
              '#ffb766',
              viewMode === 'compare' ? 0.42 : 0.9,
              2,
              true,
            );
          }
          if (showSmoothed) {
            drawBufferedSkeleton(state.current, color, 0.9, 3);
          }

          if (viewMode === 'compare') {
            context.lineWidth = 1;
            context.globalAlpha = 0.8;
            for (const index of FOCUSED_LANDMARKS) {
              const raw = hand.rawLandmarks[index];
              const offset = index * 2;
              const rawX = projectX(raw.x);
              const rawY = projectY(raw.y);
              const smoothX = projectX(state.current[offset]);
              const smoothY = projectY(state.current[offset + 1]);
              context.strokeStyle = '#fff1cc';
              context.beginPath();
              context.moveTo(rawX, rawY);
              context.lineTo(smoothX, smoothY);
              context.stroke();
              context.strokeStyle = '#ffb766';
              context.beginPath();
              context.arc(rawX, rawY, 5, 0, TAU);
              context.stroke();
              context.strokeStyle = color;
              context.beginPath();
              context.arc(smoothX, smoothY, 6, 0, TAU);
              context.stroke();
            }
          }

          if (fastMotionDebug) {
            drawDirectSkeleton(hand.rawLandmarks, '#ff9f43', 0.38, 2, true);
            drawDirectSkeleton(
              hand.filteredLandmarks,
              '#53a7ff',
              0.58,
              2.5,
              false,
            );
            drawDirectSkeleton(hand.landmarks, '#ff5fe1', 0.7, 3, false);
            context.globalAlpha = 0.95;
            context.lineWidth = 1.5;
            context.setLineDash([]);
            for (const index of [0, 8]) {
              const raw = hand.rawLandmarks[index];
              const filteredPoint = hand.filteredLandmarks[index];
              const predicted = hand.landmarks[index];
              const rawX = projectX(raw.x);
              const rawY = projectY(raw.y);
              const filteredX = projectX(filteredPoint.x);
              const filteredY = projectY(filteredPoint.y);
              const predictedX = projectX(predicted.x);
              const predictedY = projectY(predicted.y);
              context.strokeStyle = '#ffe6b8';
              context.beginPath();
              context.moveTo(rawX, rawY);
              context.lineTo(filteredX, filteredY);
              context.lineTo(predictedX, predictedY);
              context.stroke();
              for (const [x, y, markerColor, radius] of [
                [rawX, rawY, '#ff9f43', 5],
                [filteredX, filteredY, '#53a7ff', 6],
                [predictedX, predictedY, '#ff5fe1', 7],
              ] as const) {
                context.strokeStyle = markerColor;
                context.beginPath();
                context.arc(x, y, radius, 0, TAU);
                context.stroke();
              }
            }
          }

          if (detail === 'full') {
            const wrist = hand.rawLandmarks[0];
            const wristX = showSmoothed
              ? projectX(state.current[0])
              : projectX(wrist.x);
            const wristY = showSmoothed
              ? projectY(state.current[1])
              : projectY(wrist.y);
            const suffix =
              viewMode === 'compare'
                ? ' · RAW / SMOOTH'
                : viewMode === 'raw'
                  ? ' · RAW'
                  : '';
            const label = `${hand.handedness.toUpperCase()} ${(hand.confidence * 100).toFixed(0)}%${suffix}`;
            context.globalAlpha = 1;
            context.font = '600 11px Inter, sans-serif';
            const labelWidth = context.measureText(label).width + 12;
            const labelX = Math.max(
              6,
              Math.min(cssWidth - labelWidth - 6, wristX + 9),
            );
            const labelY = Math.max(22, wristY - 10);
            context.fillStyle = 'rgba(3, 7, 8, 0.78)';
            context.fillRect(labelX, labelY - 15, labelWidth, 20);
            context.fillStyle = color;
            context.fillText(label, labelX + 6, labelY);
          }
          context.globalAlpha = 1;
        }

        if (faceEnabled && faceSnapshot.faceDetected) {
          const landmarks = faceSnapshot.landmarks;
          context.globalAlpha = 0.62;
          context.fillStyle = '#ffd36e';
          context.beginPath();
          for (let index = 0; index < faceSnapshot.landmarkCount; index += 1) {
            const offset = index * 3;
            context.moveTo(
              projectX(landmarks[offset]) + 1.15,
              projectY(landmarks[offset + 1]),
            );
            context.arc(
              projectX(landmarks[offset]),
              projectY(landmarks[offset + 1]),
              1.15,
              0,
              TAU,
            );
          }
          context.fill();

          context.globalAlpha = 0.95;
          context.strokeStyle = '#fff0ad';
          context.lineWidth = 1.5;
          for (const index of [473, 468, 454, 234, 10]) {
            const offset = index * 3;
            context.beginPath();
            context.arc(
              projectX(landmarks[offset]),
              projectY(landmarks[offset + 1]),
              4,
              0,
              TAU,
            );
            context.stroke();
          }

          if (detail === 'full' && faceSnapshot.forehead) {
            const rotation = faceSnapshot.headRotation;
            const confidence =
              faceSnapshot.confidence === null
                ? ''
                : ` · ${(faceSnapshot.confidence * 100).toFixed(0)}%`;
            const direction =
              faceSnapshot.facingDirection?.label.toUpperCase() ?? 'FORWARD';
            const label = rotation
              ? `FACE · ${direction}${confidence} · Y ${rotation.yaw.toFixed(0)}° P ${rotation.pitch.toFixed(0)}°`
              : `FACE · ${direction}${confidence}`;
            const anchorX = projectX(faceSnapshot.forehead.x);
            const anchorY = projectY(faceSnapshot.forehead.y) - 12;
            context.globalAlpha = 1;
            context.font = '600 11px Inter, sans-serif';
            const labelWidth = context.measureText(label).width + 12;
            const labelX = Math.max(
              6,
              Math.min(cssWidth - labelWidth - 6, anchorX - labelWidth * 0.5),
            );
            const labelY = Math.max(22, anchorY);
            context.fillStyle = 'rgba(3, 7, 8, 0.78)';
            context.fillRect(labelX, labelY - 15, labelWidth, 20);
            context.fillStyle = '#ffd36e';
            context.fillText(label, labelX + 6, labelY);
          }
          context.globalAlpha = 1;
        }
      }

      animationFrameId = requestAnimationFrame(draw);
    };

    animationFrameId = requestAnimationFrame(draw);
    return () => {
      cancelAnimationFrame(animationFrameId);
      resizeObserver.disconnect();
      window.removeEventListener('resize', resizeCanvas);
    };
  }, [
    detail,
    faceEnabled,
    faceManager,
    fastMotionDebug,
    handEnabled,
    manager,
    pixelRatioCap,
    poseEnabled,
    poseManager,
    targetInferenceFps,
    videoRef,
    viewMode,
  ]);

  return (
    <canvas
      ref={canvasRef}
      className="hand-landmark-overlay"
      aria-hidden="true"
    />
  );
}
