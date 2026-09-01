import { useEffect, useRef } from 'react';
import type {
  FaceTrackingManager,
  HandTrackingManager,
  HandTrackingStatus,
  PoseTrackingManager,
} from '../tracking';

interface TopHudProps {
  cameraOnline: boolean;
  trackingManager: HandTrackingManager;
  trackingStatus: HandTrackingStatus;
  faceTrackingManager: FaceTrackingManager;
  poseTrackingManager: PoseTrackingManager;
  fastMotionDebug: boolean;
}

export function TopHud({
  cameraOnline,
  trackingManager,
  trackingStatus,
  faceTrackingManager,
  poseTrackingManager,
  fastMotionDebug,
}: TopHudProps) {
  const faceStateRef = useRef<HTMLSpanElement>(null);
  const faceFpsRef = useRef<HTMLSpanElement>(null);
  const faceInferenceRef = useRef<HTMLSpanElement>(null);
  const faceDelegateRef = useRef<HTMLSpanElement>(null);
  const poseStateRef = useRef<HTMLSpanElement>(null);
  const poseFpsRef = useRef<HTMLSpanElement>(null);
  const poseInferenceRef = useRef<HTMLSpanElement>(null);
  const poseDelegateRef = useRef<HTMLSpanElement>(null);
  const handsRef = useRef<HTMLSpanElement>(null);
  const handFpsRef = useRef<HTMLSpanElement>(null);
  const cameraFpsRef = useRef<HTMLSpanElement>(null);
  const renderFpsRef = useRef<HTMLSpanElement>(null);
  const inferenceRef = useRef<HTMLSpanElement>(null);
  const delegateRef = useRef<HTMLSpanElement>(null);
  const resolutionRef = useRef<HTMLSpanElement>(null);
  const actualCameraRateRef = useRef<HTMLSpanElement>(null);
  const targetRateRef = useRef<HTMLSpanElement>(null);
  const latencyRef = useRef<HTMLSpanElement>(null);
  const schedulerRef = useRef<HTMLSpanElement>(null);
  const lostRef = useRef<HTMLSpanElement>(null);
  const reacquisitionRef = useRef<HTMLSpanElement>(null);
  const filterDelayRef = useRef<HTMLSpanElement>(null);
  const blurRef = useRef<HTMLSpanElement>(null);
  const velocityRef = useRef<HTMLSpanElement>(null);
  const handVelocityRef = useRef<HTMLSpanElement>(null);
  const smoothingRef = useRef<HTMLSpanElement>(null);

  useEffect(() => {
    const updateMetrics = () => {
      const snapshot = trackingManager.snapshot;
      const faceSnapshot = faceTrackingManager.snapshot;
      const poseSnapshot = poseTrackingManager.snapshot;
      if (faceStateRef.current) {
        faceStateRef.current.textContent = faceSnapshot.faceDetected
          ? 'FACE: LOCKED'
          : 'FACE: SEARCHING';
        faceStateRef.current.classList.toggle(
          'is-online',
          faceSnapshot.faceDetected,
        );
      }
      if (faceFpsRef.current) {
        faceFpsRef.current.textContent = faceSnapshot.fps.toFixed(1);
      }
      if (faceInferenceRef.current) {
        faceInferenceRef.current.textContent =
          `${faceSnapshot.inferenceTimeMs.toFixed(1)} MS`;
      }
      if (faceDelegateRef.current) {
        faceDelegateRef.current.textContent = faceSnapshot.delegate ?? '—';
      }
      if (poseStateRef.current) {
        poseStateRef.current.textContent = poseSnapshot.poseDetected
          ? 'POSE: LOCKED'
          : 'POSE: SEARCHING';
        poseStateRef.current.classList.toggle(
          'is-online',
          poseSnapshot.poseDetected,
        );
      }
      if (poseFpsRef.current) {
        poseFpsRef.current.textContent = poseSnapshot.fps.toFixed(1);
      }
      if (poseInferenceRef.current) {
        poseInferenceRef.current.textContent =
          `${poseSnapshot.inferenceTimeMs.toFixed(1)} MS`;
      }
      if (poseDelegateRef.current) {
        poseDelegateRef.current.textContent = poseSnapshot.delegate ?? '—';
      }
      if (handsRef.current) {
        handsRef.current.textContent = `${snapshot.hands.length}`;
      }
      if (handFpsRef.current) {
        handFpsRef.current.textContent = snapshot.fps.toFixed(1);
      }
      if (cameraFpsRef.current) {
        cameraFpsRef.current.textContent = snapshot.cameraFps.toFixed(1);
      }
      if (renderFpsRef.current) {
        renderFpsRef.current.textContent = snapshot.renderFps.toFixed(1);
      }
      if (inferenceRef.current) {
        inferenceRef.current.textContent = `${snapshot.inferenceTimeMs.toFixed(1)} MS`;
      }
      if (delegateRef.current) {
        delegateRef.current.textContent = snapshot.delegate ?? '—';
      }
      if (resolutionRef.current) {
        resolutionRef.current.textContent =
          snapshot.cameraWidth > 0
            ? `${snapshot.cameraWidth}×${snapshot.cameraHeight}`
            : '—';
      }
      if (actualCameraRateRef.current) {
        actualCameraRateRef.current.textContent = snapshot.actualCameraFrameRate
          ? `${snapshot.actualCameraFrameRate.toFixed(1)} FPS`
          : '—';
      }
      if (targetRateRef.current) {
        targetRateRef.current.textContent = `${snapshot.targetInferenceFps} FPS`;
      }
      if (latencyRef.current) {
        latencyRef.current.textContent = `${snapshot.trackingLatencyMs.toFixed(1)} MS`;
      }
      if (schedulerRef.current) {
        schedulerRef.current.textContent = snapshot.usingVideoFrameCallback
          ? 'VIDEO FRAME CALLBACK'
          : 'RAF FALLBACK';
      }
      if (lostRef.current) {
        lostRef.current.textContent = `${snapshot.trackingLostCount}`;
      }
      if (reacquisitionRef.current) {
        reacquisitionRef.current.textContent = `${snapshot.reacquisitionCount}`;
      }
      if (blurRef.current) {
        blurRef.current.textContent =
          snapshot.motionBlurScore < 0
            ? '—'
            : snapshot.motionBlurScore < 120
              ? 'POSSIBLE'
              : 'LOW';
      }
      let activeHandCount = 0;
      let velocity = 0;
      let handVelocity = 0;
      let smoothingTotal = 0;
      let filterDelay = 0;
      for (const hand of snapshot.hands) {
        if (hand.isHeld) continue;
        activeHandCount += 1;
        velocity = Math.max(velocity, hand.motionSpeed);
        handVelocity = Math.max(handVelocity, hand.velocity);
        smoothingTotal += hand.visualSmoothingStrength;
        filterDelay = Math.max(filterDelay, hand.filterDelayMs);
      }
      const smoothing = activeHandCount
        ? smoothingTotal / activeHandCount
        : 0;
      if (velocityRef.current) {
        velocityRef.current.textContent = velocity.toFixed(2);
      }
      if (handVelocityRef.current) {
        handVelocityRef.current.textContent = handVelocity.toFixed(2);
      }
      if (smoothingRef.current) {
        smoothingRef.current.textContent = smoothing.toFixed(2);
      }
      if (filterDelayRef.current) {
        filterDelayRef.current.textContent = `${filterDelay.toFixed(1)} MS`;
      }
    };

    updateMetrics();
    const intervalId = window.setInterval(updateMetrics, 250);
    return () => window.clearInterval(intervalId);
  }, [faceTrackingManager, poseTrackingManager, trackingManager]);

  return (
    <header className="top-hud">
      <div className="brand-lockup" aria-label="HOLOFORGE">
        <span className="brand-mark" aria-hidden="true" />
        <div>
          <p className="eyebrow">AR SYSTEM</p>
          <h1>HOLOFORGE</h1>
        </div>
      </div>

      <div className="hud-telemetry" role="status" aria-live="polite">
        <span className={`hud-value ${cameraOnline ? 'is-online' : ''}`}>
          <span className="status-dot" aria-hidden="true" />
          CAMERA: {cameraOnline ? 'ONLINE' : 'OFFLINE'}
        </span>
        <span
          className={`hud-value ${trackingStatus === 'online' ? 'is-online' : ''}`}
        >
          HAND TRACKING:{' '}
          {trackingStatus === 'online'
            ? 'ONLINE'
            : trackingStatus === 'loading'
              ? 'LOADING'
              : 'OFFLINE'}
        </span>
        <span ref={faceStateRef} className="hud-value">
          FACE: SEARCHING
        </span>
        <span ref={poseStateRef} className="hud-value">
          POSE: SEARCHING
        </span>
        <span className="hud-value">
          HANDS: <span ref={handsRef}>0</span>
        </span>
        <span className="hud-value hud-value--diagnostic">
          RENDER FPS: <span ref={renderFpsRef}>0.0</span>
        </span>
        <span className="hud-value hud-value--diagnostic">
          CAMERA FPS: <span ref={cameraFpsRef}>0.0</span>
        </span>
        <span className="hud-value hud-value--diagnostic">
          HAND FPS: <span ref={handFpsRef}>0.0</span>
        </span>
        <span className="hud-value hud-value--diagnostic">
          FACE FPS: <span ref={faceFpsRef}>0.0</span>
        </span>
        <span className="hud-value hud-value--diagnostic">
          POSE FPS: <span ref={poseFpsRef}>0.0</span>
        </span>
        <span className="hud-value hud-value--diagnostic">
          INFERENCE: <span ref={inferenceRef}>0.0 MS</span>
        </span>
        <span className="hud-value hud-value--diagnostic">
          FACE INFERENCE: <span ref={faceInferenceRef}>0.0 MS</span>
        </span>
        <span className="hud-value hud-value--diagnostic">
          POSE INFERENCE: <span ref={poseInferenceRef}>0.0 MS</span>
        </span>
        <span className="hud-value hud-value--diagnostic">
          MEDIAPIPE: <span ref={delegateRef}>—</span>
        </span>
        <span className="hud-value hud-value--diagnostic">
          FACE MEDIAPIPE: <span ref={faceDelegateRef}>—</span>
        </span>
        <span className="hud-value hud-value--diagnostic">
          POSE MEDIAPIPE: <span ref={poseDelegateRef}>—</span>
        </span>
        <span className="hud-value hud-value--diagnostic">
          CAMERA: <span ref={resolutionRef}>—</span>
        </span>
        <span className="hud-value hud-value--diagnostic">
          CAMERA SOURCE: <span ref={actualCameraRateRef}>—</span>
        </span>
        <span className="hud-value hud-value--diagnostic">
          HAND TARGET: <span ref={targetRateRef}>30 FPS</span>
        </span>
        <span className="hud-value hud-value--diagnostic">
          TRACKING LATENCY: <span ref={latencyRef}>0.0 MS</span>
        </span>
        <span className="hud-value hud-value--diagnostic">
          FRAME SYNC: <span ref={schedulerRef}>—</span>
        </span>
        <span className="hud-value hud-value--diagnostic">
          MOTION SPEED: <span ref={velocityRef}>0.00</span>
        </span>
        <span className="hud-value hud-value--diagnostic">
          SMOOTHING: <span ref={smoothingRef}>0.00</span>
        </span>
        {fastMotionDebug && (
          <>
            <span className="hud-value hud-value--diagnostic">
              HAND VELOCITY: <span ref={handVelocityRef}>0.00</span>
            </span>
            <span className="hud-value hud-value--diagnostic">
              TRACKING LOST: <span ref={lostRef}>0</span>
            </span>
            <span className="hud-value hud-value--diagnostic">
              REACQUISITIONS: <span ref={reacquisitionRef}>0</span>
            </span>
            <span className="hud-value hud-value--diagnostic">
              FILTER DELAY: <span ref={filterDelayRef}>0.0 MS</span>
            </span>
            <span className="hud-value hud-value--diagnostic">
              MOTION BLUR: <span ref={blurRef}>—</span>
            </span>
          </>
        )}
      </div>
    </header>
  );
}
