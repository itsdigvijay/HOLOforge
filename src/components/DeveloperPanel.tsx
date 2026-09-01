import type { CameraDevice, CameraStatus } from '../hooks';
import type {
  HandTrackingRate,
  HandTrackingStatus,
  MediaPipeTrackingConfig,
  MotionResponseConfig,
  OneEuroFilterConfig,
} from '../tracking';
import type {
  FaceTrackingStatus,
  PoseTrackingStatus,
} from '../tracking';
import type { PerformanceMode } from '../settings';
import type { LandmarkViewMode } from './HandLandmarkOverlay';

interface DeveloperPanelProps {
  cameras: CameraDevice[];
  cameraStatus: CameraStatus;
  selectedCameraId: string;
  onCameraChange: (deviceId: string) => void;
  trackingStatus: HandTrackingStatus;
  faceTrackingStatus: FaceTrackingStatus;
  poseTrackingStatus: PoseTrackingStatus;
  showDeveloperLandmarks: boolean;
  onDeveloperLandmarksChange: (enabled: boolean) => void;
  showFaceLandmarks: boolean;
  onFaceLandmarksChange: (enabled: boolean) => void;
  showPoseLandmarks: boolean;
  onPoseLandmarksChange: (enabled: boolean) => void;
  visualFilterConfig: OneEuroFilterConfig;
  onVisualFilterChange: (
    key: keyof OneEuroFilterConfig,
    value: number,
  ) => void;
  gestureFilterConfig: OneEuroFilterConfig;
  onGestureFilterChange: (
    key: keyof OneEuroFilterConfig,
    value: number,
  ) => void;
  fingertipStabilization: number;
  onFingertipStabilizationChange: (value: number) => void;
  landmarkViewMode: LandmarkViewMode;
  onLandmarkViewModeChange: (mode: LandmarkViewMode) => void;
  performanceMode: PerformanceMode;
  onPerformanceModeChange: (mode: PerformanceMode) => void;
  trackingRate: HandTrackingRate;
  onTrackingRateChange: (rate: HandTrackingRate) => void;
  motionConfig: MotionResponseConfig;
  onMotionConfigChange: (
    key: keyof MotionResponseConfig,
    value: number,
  ) => void;
  mediaPipeConfig: MediaPipeTrackingConfig;
  onMediaPipeConfigChange: (
    key: keyof MediaPipeTrackingConfig,
    value: number,
  ) => void;
  fastMotionDebug: boolean;
  onFastMotionDebugChange: (enabled: boolean) => void;
}

export function DeveloperPanel({
  cameras,
  cameraStatus,
  selectedCameraId,
  onCameraChange,
  trackingStatus,
  faceTrackingStatus,
  poseTrackingStatus,
  showDeveloperLandmarks,
  onDeveloperLandmarksChange,
  showFaceLandmarks,
  onFaceLandmarksChange,
  showPoseLandmarks,
  onPoseLandmarksChange,
  visualFilterConfig,
  onVisualFilterChange,
  gestureFilterConfig,
  onGestureFilterChange,
  fingertipStabilization,
  onFingertipStabilizationChange,
  landmarkViewMode,
  onLandmarkViewModeChange,
  performanceMode,
  onPerformanceModeChange,
  trackingRate,
  onTrackingRateChange,
  motionConfig,
  onMotionConfigChange,
  mediaPipeConfig,
  onMediaPipeConfigChange,
  fastMotionDebug,
  onFastMotionDebugChange,
}: DeveloperPanelProps) {
  const cameraOnline = cameraStatus === 'ready';

  return (
    <aside className="developer-panel" aria-label="Camera settings">
      <div className="panel-heading">
        <span>SETTINGS</span>
        <span className={`panel-badge ${cameraOnline ? 'is-online' : ''}`}>
          {cameraOnline ? 'CAMERA ONLINE' : 'CAMERA OFFLINE'}
        </span>
      </div>

      <div className="camera-setting">
        <label htmlFor="camera-select">CAMERA SOURCE</label>
        <select
          id="camera-select"
          value={selectedCameraId}
          disabled={!cameraOnline || cameras.length === 0}
          onChange={(event) => onCameraChange(event.target.value)}
        >
          {cameras.length === 0 && <option value="">NO CAMERA AVAILABLE</option>}
          {cameras.map((camera) => (
            <option key={camera.deviceId} value={camera.deviceId}>
              {camera.label}
            </option>
          ))}
        </select>
        <small>Video never leaves this device</small>

        <label htmlFor="performance-mode">PERFORMANCE MODE</label>
        <select
          id="performance-mode"
          value={performanceMode}
          onChange={(event) =>
            onPerformanceModeChange(event.target.value as PerformanceMode)
          }
        >
          <option value="performance">PERFORMANCE · 640×480 / 60</option>
          <option value="balanced">BALANCED · 720P / 30</option>
          <option value="quality">QUALITY · 720P / 60</option>
        </select>

        <label htmlFor="hand-tracking-rate">HAND TRACKING RATE</label>
        <select
          id="hand-tracking-rate"
          value={trackingRate}
          onChange={(event) => {
            const value = event.target.value;
            onTrackingRateChange(
              value === 'auto' ? 'auto' : (Number(value) as 30 | 45 | 60),
            );
          }}
        >
          <option value="auto">AUTO</option>
          <option value="30">30 FPS</option>
          <option value="45">45 FPS</option>
          <option value="60">60 FPS</option>
        </select>

        <div className="settings-divider" />
        <p className="settings-group-title">VISUAL FILTER</p>
        <FilterControl
          id="visual-min-cutoff"
          label="MIN CUTOFF"
          value={visualFilterConfig.minCutoff}
          min={0.1}
          max={4}
          step={0.1}
          onChange={(value) => onVisualFilterChange('minCutoff', value)}
        />
        <FilterControl
          id="visual-beta"
          label="BETA"
          value={visualFilterConfig.beta}
          min={0}
          max={2}
          step={0.05}
          onChange={(value) => onVisualFilterChange('beta', value)}
        />
        <FilterControl
          id="visual-derivative-cutoff"
          label="DERIVATIVE CUTOFF"
          value={visualFilterConfig.derivativeCutoff}
          min={0.1}
          max={4}
          step={0.1}
          onChange={(value) =>
            onVisualFilterChange('derivativeCutoff', value)
          }
        />
        <FilterControl
          id="fingertip-stabilization"
          label="TIP STABILIZATION"
          value={fingertipStabilization}
          min={0}
          max={0.3}
          step={0.01}
          onChange={onFingertipStabilizationChange}
        />
        <FilterControl
          id="fast-motion-threshold"
          label="FAST MOTION THRESHOLD"
          value={motionConfig.fastVelocityThreshold}
          min={0.4}
          max={3}
          step={0.05}
          onChange={(value) =>
            onMotionConfigChange('fastVelocityThreshold', value)
          }
        />
        <FilterControl
          id="large-movement-threshold"
          label="LARGE MOVE DISTANCE"
          value={motionConfig.largeDisplacementThreshold}
          min={0.03}
          max={0.2}
          step={0.01}
          onChange={(value) =>
            onMotionConfigChange('largeDisplacementThreshold', value)
          }
        />
        <FilterControl
          id="visual-prediction"
          label="VISUAL PREDICTION MS"
          value={motionConfig.visualPredictionMs}
          min={0}
          max={30}
          step={1}
          onChange={(value) =>
            onMotionConfigChange('visualPredictionMs', value)
          }
        />

        <div className="settings-divider" />
        <p className="settings-group-title">GESTURE FILTER</p>
        <FilterControl
          id="gesture-min-cutoff"
          label="MIN CUTOFF"
          value={gestureFilterConfig.minCutoff}
          min={0.1}
          max={6}
          step={0.1}
          onChange={(value) => onGestureFilterChange('minCutoff', value)}
        />
        <FilterControl
          id="gesture-beta"
          label="BETA"
          value={gestureFilterConfig.beta}
          min={0}
          max={3}
          step={0.05}
          onChange={(value) => onGestureFilterChange('beta', value)}
        />
        <FilterControl
          id="gesture-derivative-cutoff"
          label="DERIVATIVE CUTOFF"
          value={gestureFilterConfig.derivativeCutoff}
          min={0.1}
          max={4}
          step={0.1}
          onChange={(value) =>
            onGestureFilterChange('derivativeCutoff', value)
          }
        />

        <div className="settings-divider" />
        <p className="settings-group-title">MEDIAPIPE TRACKING</p>
        <FilterControl
          id="hand-detection-confidence"
          label="DETECTION CONFIDENCE"
          value={mediaPipeConfig.minHandDetectionConfidence}
          min={0.3}
          max={0.8}
          step={0.05}
          onChange={(value) =>
            onMediaPipeConfigChange('minHandDetectionConfidence', value)
          }
        />
        <FilterControl
          id="hand-presence-confidence"
          label="PRESENCE CONFIDENCE"
          value={mediaPipeConfig.minHandPresenceConfidence}
          min={0.3}
          max={0.8}
          step={0.05}
          onChange={(value) =>
            onMediaPipeConfigChange('minHandPresenceConfidence', value)
          }
        />
        <FilterControl
          id="hand-tracking-confidence"
          label="TRACKING CONFIDENCE"
          value={mediaPipeConfig.minTrackingConfidence}
          min={0.3}
          max={0.8}
          step={0.05}
          onChange={(value) =>
            onMediaPipeConfigChange('minTrackingConfidence', value)
          }
        />

        <div className="settings-divider" />
        <label className="toggle-setting" htmlFor="landmark-toggle">
          <span>DEV LANDMARKS</span>
          <input
            id="landmark-toggle"
            type="checkbox"
            checked={showDeveloperLandmarks}
            disabled={trackingStatus !== 'online'}
            onChange={(event) =>
              onDeveloperLandmarksChange(event.target.checked)
            }
          />
          <span className="toggle-track" aria-hidden="true" />
        </label>

        <label className="toggle-setting" htmlFor="fast-motion-debug">
          <span>FAST MOTION DEBUG</span>
          <input
            id="fast-motion-debug"
            type="checkbox"
            checked={fastMotionDebug}
            disabled={trackingStatus !== 'online'}
            onChange={(event) =>
              onFastMotionDebugChange(event.target.checked)
            }
          />
          <span className="toggle-track" aria-hidden="true" />
        </label>

        <label className="toggle-setting" htmlFor="face-landmark-toggle">
          <span>FACE LANDMARKS</span>
          <input
            id="face-landmark-toggle"
            type="checkbox"
            checked={showFaceLandmarks}
            disabled={faceTrackingStatus !== 'online'}
            onChange={(event) =>
              onFaceLandmarksChange(event.target.checked)
            }
          />
          <span className="toggle-track" aria-hidden="true" />
        </label>

        <label className="toggle-setting" htmlFor="pose-landmark-toggle">
          <span>POSE SKELETON</span>
          <input
            id="pose-landmark-toggle"
            type="checkbox"
            checked={showPoseLandmarks}
            disabled={poseTrackingStatus !== 'online'}
            onChange={(event) =>
              onPoseLandmarksChange(event.target.checked)
            }
          />
          <span className="toggle-track" aria-hidden="true" />
        </label>

        <label htmlFor="landmark-view">LANDMARK VIEW</label>
        <select
          id="landmark-view"
          value={landmarkViewMode}
          disabled={!showDeveloperLandmarks}
          onChange={(event) =>
            onLandmarkViewModeChange(event.target.value as LandmarkViewMode)
          }
        >
          <option value="smoothed">SMOOTHED</option>
          <option value="raw">RAW</option>
          <option value="compare">RAW + SMOOTHED</option>
        </select>
      </div>
    </aside>
  );
}

interface FilterControlProps {
  id: string;
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (value: number) => void;
}

function FilterControl({
  id,
  label,
  value,
  min,
  max,
  step,
  onChange,
}: FilterControlProps) {
  return (
    <div className="smoothing-control">
      <label htmlFor={id}>
        <span>{label}</span>
        <output htmlFor={id}>{value.toFixed(2)}</output>
      </label>
      <input
        id={id}
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
      />
    </div>
  );
}
