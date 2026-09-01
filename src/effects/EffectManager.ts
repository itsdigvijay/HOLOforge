import type { EffectContext, ManagerStatus } from '../types';

/** Coordinates future Three.js effects and their render lifecycle. */
export class EffectManager {
  private currentStatus: ManagerStatus = 'idle';
  private context: EffectContext | null = null;

  get status(): ManagerStatus {
    return this.currentStatus;
  }

  get effectContext(): EffectContext | null {
    return this.context;
  }

  initialize(context: EffectContext): void {
    this.context = context;
    this.currentStatus = 'ready';
  }

  dispose(): void {
    this.context = null;
    this.currentStatus = 'idle';
  }
}
