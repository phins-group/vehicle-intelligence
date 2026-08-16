import '@angular/compiler';

import { HttpClient } from '@angular/common/http';
import { Injector, runInInjectionContext } from '@angular/core';
import { of } from 'rxjs';
import { describe, expect, it, vi } from 'vitest';

import { ApiClientService } from './api-client.service';

describe('ApiClientService camera health contract', () => {
  it('uses a collection route that cannot shadow a valid camera id', () => {
    const get = vi.fn(() => of({ items: [] }));
    const injector = Injector.create({
      providers: [{ provide: HttpClient, useValue: { get } }],
    });
    const service = runInInjectionContext(
      injector,
      () => new ApiClientService(),
    );

    service.cameraHealthSnapshot();

    expect(get).toHaveBeenCalledWith('/api/camera-health');
  });
});
