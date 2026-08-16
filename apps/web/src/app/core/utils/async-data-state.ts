import { computed, signal } from '@angular/core';

export class AsyncDataState {
  private readonly loadedState = signal(false);
  private readonly errorState = signal<string | null>(null);

  readonly hasLoaded = this.loadedState.asReadonly();
  readonly error = this.errorState.asReadonly();
  readonly initialError = computed(() =>
    this.loadedState() ? null : this.errorState()
  );
  readonly staleError = computed(() =>
    this.loadedState() ? this.errorState() : null
  );

  begin(): void {
    this.errorState.set(null);
  }

  succeed(): void {
    this.loadedState.set(true);
    this.errorState.set(null);
  }

  fail(message: string): void {
    this.errorState.set(message);
  }
}
