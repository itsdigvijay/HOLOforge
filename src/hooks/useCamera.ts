import { useCallback, useEffect, useRef, useState } from 'react';
import type { CameraCaptureProfile } from '../settings';

export type CameraStatus =
  | 'idle'
  | 'loading'
  | 'ready'
  | 'permission-denied'
  | 'not-found'
  | 'error';

export interface CameraDevice {
  deviceId: string;
  label: string;
}

const DEFAULT_CAPTURE_PROFILE: CameraCaptureProfile = {
  width: 640,
  height: 480,
  frameRate: 60,
};

function stopStream(stream: MediaStream | null) {
  stream?.getTracks().forEach((track) => track.stop());
}

function statusFromError(error: unknown): CameraStatus {
  if (error instanceof DOMException) {
    if (error.name === 'NotAllowedError' || error.name === 'SecurityError') {
      return 'permission-denied';
    }

    if (
      error.name === 'NotFoundError' ||
      error.name === 'DevicesNotFoundError' ||
      error.name === 'OverconstrainedError'
    ) {
      return 'not-found';
    }
  }

  return 'error';
}

export function useCamera() {
  const [status, setStatus] = useState<CameraStatus>('idle');
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [cameras, setCameras] = useState<CameraDevice[]>([]);
  const [selectedCameraId, setSelectedCameraId] = useState('');
  const streamRef = useRef<MediaStream | null>(null);
  const requestIdRef = useRef(0);
  const mountedRef = useRef(true);
  const initializedRef = useRef(false);
  const captureProfileRef = useRef<CameraCaptureProfile>(
    DEFAULT_CAPTURE_PROFILE,
  );

  const refreshCameras = useCallback(async () => {
    if (!initializedRef.current || !navigator.mediaDevices?.enumerateDevices) {
      return;
    }

    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      if (!mountedRef.current) return;

      const videoInputs = devices
        .filter((device) => device.kind === 'videoinput')
        .map((device, index) => ({
          deviceId: device.deviceId,
          label: device.label || `Camera ${index + 1}`,
        }));

      setCameras(videoInputs);
    } catch {
      // An active stream can continue even when device enumeration is blocked.
    }
  }, []);

  const startCamera = useCallback(
    async (deviceId?: string, captureProfile?: CameraCaptureProfile) => {
      if (captureProfile) captureProfileRef.current = captureProfile;
      const activeProfile = captureProfileRef.current;
      initializedRef.current = true;
      const requestId = ++requestIdRef.current;

      stopStream(streamRef.current);
      streamRef.current = null;
      setStream(null);
      setStatus('loading');

      if (!navigator.mediaDevices?.getUserMedia) {
        setStatus('error');
        return;
      }

      try {
        const videoConstraints: MediaTrackConstraints = {
          ...(deviceId ? { deviceId: { exact: deviceId } } : {}),
          width: { ideal: activeProfile.width, max: activeProfile.width },
          height: { ideal: activeProfile.height, max: activeProfile.height },
          frameRate: {
            ideal: activeProfile.frameRate,
            min: Math.min(30, activeProfile.frameRate),
            max: activeProfile.frameRate,
          },
        };
        let nextStream: MediaStream;
        try {
          nextStream = await navigator.mediaDevices.getUserMedia({
            audio: false,
            video: videoConstraints,
          });
        } catch (error) {
          // Some cameras report incomplete capabilities. Retry without a hard
          // minimum so a genuine lower-rate camera still works gracefully.
          if (!(error instanceof DOMException) || error.name !== 'OverconstrainedError') {
            throw error;
          }
          nextStream = await navigator.mediaDevices.getUserMedia({
            audio: false,
            video: {
              ...videoConstraints,
              frameRate: {
                ideal: activeProfile.frameRate,
                max: activeProfile.frameRate,
              },
            },
          });
        }

        if (!mountedRef.current || requestId !== requestIdRef.current) {
          stopStream(nextStream);
          return;
        }

        streamRef.current = nextStream;
        setStream(nextStream);
        setSelectedCameraId(
          nextStream.getVideoTracks()[0]?.getSettings().deviceId ?? deviceId ?? '',
        );
        setStatus('ready');
        await refreshCameras();
      } catch (error) {
        if (!mountedRef.current || requestId !== requestIdRef.current) return;
        setStatus(statusFromError(error));
      }
    },
    [refreshCameras],
  );

  const selectCamera = useCallback(
    (deviceId: string) => {
      if (!deviceId || deviceId === selectedCameraId) return;
      setSelectedCameraId(deviceId);
      void startCamera(deviceId);
    },
    [selectedCameraId, startCamera],
  );

  const setCaptureProfile = useCallback((profile: CameraCaptureProfile) => {
    captureProfileRef.current = profile;
  }, []);

  useEffect(() => {
    mountedRef.current = true;

    const handleDeviceChange = () => {
      void refreshCameras();
    };

    navigator.mediaDevices?.addEventListener?.('devicechange', handleDeviceChange);

    return () => {
      mountedRef.current = false;
      requestIdRef.current += 1;
      navigator.mediaDevices?.removeEventListener?.(
        'devicechange',
        handleDeviceChange,
      );
      stopStream(streamRef.current);
      streamRef.current = null;
    };
  }, [refreshCameras]);

  return {
    cameras,
    selectedCameraId,
    selectCamera,
    setCaptureProfile,
    startCamera,
    status,
    stream,
  };
}
