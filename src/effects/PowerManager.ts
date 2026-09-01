import type { ManagerStatus, PowerDefinition } from '../types';

/** Owns the future power registry and current power selection. */
export class PowerManager {
  private currentStatus: ManagerStatus = 'idle';
  private selectedPower: PowerDefinition | null = null;

  get status(): ManagerStatus {
    return this.currentStatus;
  }

  get activePower(): PowerDefinition | null {
    return this.selectedPower;
  }

  initialize(): void {
    this.currentStatus = 'ready';
  }

  dispose(): void {
    this.selectedPower = null;
    this.currentStatus = 'idle';
  }
}
