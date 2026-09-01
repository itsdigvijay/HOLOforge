import { useEffect, useRef, useState } from 'react';
import {
  DeveloperPanel,
  PowerSelector,
  TopHud,
  WebcamViewport,
  type LandmarkViewMode,
} from './components';
import {
  useCamera,
  useFaceTracking,
  useHandTracking,
  usePoseTracking,
} from './hooks';
import {
  DEFAULT_PERFORMANCE_MODE,
  PERFORMANCE_PROFILES,
  type PerformanceMode,
} from './settings';
import {
  DEFAULT_FINGERTIP_CUTOFF_MULTIPLIER,
  DEFAULT_GESTURE_FILTER_CONFIG,
  DEFAULT_MOTION_RESPONSE_CONFIG,
  DEFAULT_VISUAL_FILTER_CONFIG,
  type HandTrackingRate,
  type MediaPipeTrackingConfig,
  type MotionResponseConfig,
  type OneEuroFilterConfig,
} from './tracking';

export function App() {
  const camera = useCamera();
  const videoRef = useRef<HTMLVideoElement>(null);
  const tracking = useHandTracking(videoRef, camera.status === 'ready');
  const faceTracking = useFaceTracking(videoRef, camera.status === 'ready');
  const poseTracking = usePoseTracking(videoRef, camera.status === 'ready');
  const [performanceMode, setPerformanceMode] = useState<PerformanceMode>(
    DEFAULT_PERFORMANCE_MODE,
  );
  const [showDeveloperLandmarks, setShowDeveloperLandmarks] = useState(true);
  const [showFaceLandmarks, setShowFaceLandmarks] = useState(false);
  const [showPoseLandmarks, setShowPoseLandmarks] = useState(false);
  const [fastMotionDebug, setFastMotionDebug] = useState(false);
  const [trackingRate, setTrackingRate] = useState<HandTrackingRate>('auto');
  const [motionConfig, setMotionConfig] = useState<MotionResponseConfig>({
    ...DEFAULT_MOTION_RESPONSE_CONFIG,
  });
  const [mediaPipeConfig, setMediaPipeConfig] =
    useState<MediaPipeTrackingConfig>({
      minHandDetectionConfidence: 0.5,
      minHandPresenceConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });
  const [visualFilterConfig, setVisualFilterConfig] =
    useState<OneEuroFilterConfig>({ ...DEFAULT_VISUAL_FILTER_CONFIG });
  const [gestureFilterConfig, setGestureFilterConfig] =
    useState<OneEuroFilterConfig>({ ...DEFAULT_GESTURE_FILTER_CONFIG });
  const [fingertipStabilization, setFingertipStabilization] = useState(
    1 - DEFAULT_FINGERTIP_CUTOFF_MULTIPLIER,
  );
  const [landmarkViewMode, setLandmarkViewMode] =
    useState<LandmarkViewMode>('smoothed');

  useEffect(() => {
    faceTracking.manager.setLandmarkVisualizationEnabled(showFaceLandmarks);
  }, [faceTracking.manager, showFaceLandmarks]);

  useEffect(() => {
    poseTracking.manager.setLandmarkVisualizationEnabled(showPoseLandmarks);
  }, [poseTracking.manager, showPoseLandmarks]);

  const updateVisualFilter = (
    key: keyof OneEuroFilterConfig,
    value: number,
  ) => {
    tracking.manager.setVisualFilterConfig({ [key]: value });
    setVisualFilterConfig((current) => ({ ...current, [key]: value }));
  };

  const updateGestureFilter = (
    key: keyof OneEuroFilterConfig,
    value: number,
  ) => {
    tracking.manager.setGestureFilterConfig({ [key]: value });
    setGestureFilterConfig((current) => ({ ...current, [key]: value }));
  };

  const updateFingertipStabilization = (value: number) => {
    tracking.manager.setFingertipCutoffMultiplier(1 - value);
    setFingertipStabilization(value);
  };

  const updateTrackingRate = (rate: HandTrackingRate) => {
    tracking.manager.setTrackingRate(rate);
    setTrackingRate(rate);
  };

  const updateMotionConfig = (
    key: keyof MotionResponseConfig,
    value: number,
  ) => {
    tracking.manager.setMotionResponseConfig({ [key]: value });
    setMotionConfig((current) => ({ ...current, [key]: value }));
  };

  const updateMediaPipeConfig = (
    key: keyof MediaPipeTrackingConfig,
    value: number,
  ) => {
    tracking.manager.setMediaPipeConfig({ [key]: value });
    setMediaPipeConfig((current) => ({ ...current, [key]: value }));
  };

  const updatePerformanceMode = (mode: PerformanceMode) => {
    const profile = PERFORMANCE_PROFILES[mode];
    setPerformanceMode(mode);
    camera.setCaptureProfile(profile.camera);

    if (camera.status === 'ready') {
      void camera.startCamera(
        camera.selectedCameraId || undefined,
        profile.camera,
      );
    }
  };

  const activePerformanceProfile = PERFORMANCE_PROFILES[performanceMode];

  return (
    <main className="app-shell">
      <TopHud
        cameraOnline={camera.status === 'ready'}
        trackingManager={tracking.manager}
        trackingStatus={tracking.status}
        faceTrackingManager={faceTracking.manager}
        poseTrackingManager={poseTracking.manager}
        fastMotionDebug={fastMotionDebug}
      />
      <WebcamViewport
        status={camera.status}
        stream={camera.stream}
        videoRef={videoRef}
        trackingManager={tracking.manager}
        faceTrackingManager={faceTracking.manager}
        poseTrackingManager={poseTracking.manager}
        showDeveloperLandmarks={showDeveloperLandmarks}
        showFaceLandmarks={showFaceLandmarks}
        showPoseLandmarks={showPoseLandmarks}
        landmarkViewMode={landmarkViewMode}
        overlayDetail={activePerformanceProfile.overlayDetail}
        overlayPixelRatioCap={activePerformanceProfile.overlayPixelRatioCap}
        targetInferenceFps={activePerformanceProfile.handInferenceFps}
        fastMotionDebug={fastMotionDebug}
        onInitialize={() => void camera.startCamera()}
      />
      <DeveloperPanel
        cameras={camera.cameras}
        cameraStatus={camera.status}
        selectedCameraId={camera.selectedCameraId}
        onCameraChange={camera.selectCamera}
        trackingStatus={tracking.status}
        faceTrackingStatus={faceTracking.status}
        poseTrackingStatus={poseTracking.status}
        showDeveloperLandmarks={showDeveloperLandmarks}
        onDeveloperLandmarksChange={setShowDeveloperLandmarks}
        showFaceLandmarks={showFaceLandmarks}
        onFaceLandmarksChange={setShowFaceLandmarks}
        showPoseLandmarks={showPoseLandmarks}
        onPoseLandmarksChange={setShowPoseLandmarks}
        visualFilterConfig={visualFilterConfig}
        onVisualFilterChange={updateVisualFilter}
        gestureFilterConfig={gestureFilterConfig}
        onGestureFilterChange={updateGestureFilter}
        fingertipStabilization={fingertipStabilization}
        onFingertipStabilizationChange={updateFingertipStabilization}
        landmarkViewMode={landmarkViewMode}
        onLandmarkViewModeChange={setLandmarkViewMode}
        performanceMode={performanceMode}
        onPerformanceModeChange={updatePerformanceMode}
        trackingRate={trackingRate}
        onTrackingRateChange={updateTrackingRate}
        motionConfig={motionConfig}
        onMotionConfigChange={updateMotionConfig}
        mediaPipeConfig={mediaPipeConfig}
        onMediaPipeConfigChange={updateMediaPipeConfig}
        fastMotionDebug={fastMotionDebug}
        onFastMotionDebugChange={setFastMotionDebug}
      />
      <PowerSelector />
    </main>
  );
}
