import { useEffect, type RefObject } from 'react';
import type { CameraStatus } from '../hooks';
import type {
  FaceTrackingManager,
  HandTrackingManager,
  PoseTrackingManager,
} from '../tracking';
import type { OverlayDetail } from '../settings';
import {
  HandLandmarkOverlay,
  type LandmarkViewMode,
} from './HandLandmarkOverlay';

interface WebcamViewportProps {
  status: CameraStatus;
  stream: MediaStream | null;
  videoRef: RefObject<HTMLVideoElement | null>;
  trackingManager: HandTrackingManager;
  faceTrackingManager: FaceTrackingManager;
  poseTrackingManager: PoseTrackingManager;
  showDeveloperLandmarks: boolean;
  showFaceLandmarks: boolean;
  showPoseLandmarks: boolean;
  landmarkViewMode: LandmarkViewMode;
  overlayDetail: OverlayDetail;
  overlayPixelRatioCap: 1 | 1.5 | 2;
  targetInferenceFps: number;
  fastMotionDebug: boolean;
  onInitialize: () => void;
}

const stateCopy: Record<
  Exclude<CameraStatus, 'idle' | 'ready'>,
  { title: string; detail: string }
> = {
  loading: {
    title: 'INITIALIZING CAMERA',
    detail: 'Waiting for browser camera access…',
  },
  'permission-denied': {
    title: 'CAMERA ACCESS DENIED',
    detail: 'Allow camera access in your browser settings, then try again.',
  },
  'not-found': {
    title: 'NO CAMERA FOUND',
    detail: 'Connect a camera and try again.',
  },
  error: {
    title: 'CAMERA INITIALIZATION FAILED',
    detail: 'Check that another app is not using your camera, then try again.',
  },
};

export function WebcamViewport({
  status,
  stream,
  videoRef,
  trackingManager,
  faceTrackingManager,
  poseTrackingManager,
  showDeveloperLandmarks,
  showFaceLandmarks,
  showPoseLandmarks,
  landmarkViewMode,
  overlayDetail,
  overlayPixelRatioCap,
  targetInferenceFps,
  fastMotionDebug,
  onInitialize,
}: WebcamViewportProps) {
  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    video.srcObject = stream;
    if (stream) void video.play().catch(() => undefined);

    return () => {
      video.srcObject = null;
    };
  }, [stream]);

  const message = status !== 'idle' && status !== 'ready' ? stateCopy[status] : null;

  return (
    <section className="webcam-viewport" aria-label="Live webcam viewport">
      <video
        ref={videoRef}
        className="webcam-video"
        autoPlay
        muted
        playsInline
        aria-label="Mirrored live camera feed"
      />
      <HandLandmarkOverlay
        handEnabled={showDeveloperLandmarks && status === 'ready'}
        faceEnabled={showFaceLandmarks && status === 'ready'}
        poseEnabled={showPoseLandmarks && status === 'ready'}
        manager={trackingManager}
        faceManager={faceTrackingManager}
        poseManager={poseTrackingManager}
        videoRef={videoRef}
        viewMode={landmarkViewMode}
        detail={overlayDetail}
        pixelRatioCap={overlayPixelRatioCap}
        targetInferenceFps={targetInferenceFps}
        fastMotionDebug={fastMotionDebug}
      />

      <span className="frame-corner frame-corner--top-left" aria-hidden="true" />
      <span className="frame-corner frame-corner--top-right" aria-hidden="true" />
      <span className="frame-corner frame-corner--bottom-left" aria-hidden="true" />
      <span className="frame-corner frame-corner--bottom-right" aria-hidden="true" />

      {status === 'idle' && (
        <div className="viewport-message">
          <span className="camera-glyph" aria-hidden="true" />
          <p>CAMERA MODULE STANDBY</p>
          <small>Camera processing remains on this device</small>
          <button className="initialize-button" type="button" onClick={onInitialize}>
            INITIALIZE HOLOFORGE
          </button>
        </div>
      )}

      {message && (
        <div className="viewport-message" role="status" aria-live="polite">
          <span
            className={`camera-glyph ${status === 'loading' ? 'is-loading' : ''}`}
            aria-hidden="true"
          />
          <p>{message.title}</p>
          <small>{message.detail}</small>
          {status !== 'loading' && (
            <button className="initialize-button" type="button" onClick={onInitialize}>
              TRY AGAIN
            </button>
          )}
        </div>
      )}

      {status === 'ready' && (
        <p className="camera-ready-label" role="status">
          LIVE · LOCAL PROCESSING
        </p>
      )}
    </section>
  );
}
