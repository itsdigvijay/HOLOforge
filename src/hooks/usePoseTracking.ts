import { useEffect, useRef, useState, type RefObject } from 'react';
import {
  PoseTrackingManager,
  type PoseTrackingStatus,
} from '../tracking';

export function usePoseTracking(
  videoRef: RefObject<HTMLVideoElement | null>,
  cameraReady: boolean,
) {
  const managerRef = useRef<PoseTrackingManager | null>(null);
  const lifecycleRef = useRef(0);
  const [status, setStatus] = useState<PoseTrackingStatus>('idle');

  if (!managerRef.current) managerRef.current = new PoseTrackingManager();

  useEffect(() => {
    const manager = managerRef.current;
    if (!manager) return;
    return manager.subscribeStatus(setStatus);
  }, []);

  useEffect(() => {
    const manager = managerRef.current;
    if (!manager) return;
    if (!cameraReady || !videoRef.current) {
      manager.stop();
      return;
    }
    void manager
      .start(videoRef.current)
      .then(() => undefined)
      .catch(() => undefined);
  }, [cameraReady, videoRef]);

  useEffect(() => {
    const manager = managerRef.current;
    if (!manager) return;
    const lifecycle = ++lifecycleRef.current;
    return () => {
      manager.stop();
      queueMicrotask(() => {
        if (lifecycleRef.current === lifecycle) manager.dispose();
      });
    };
  }, []);

  return { manager: managerRef.current, status };
}
