import { Routes } from '@angular/router';

import { authGuard } from './core/auth/auth.guard';

export const routes: Routes = [
  {
    path: 'login',
    loadComponent: () =>
      import('./features/login/login.component').then((module) => module.LoginComponent)
  },
  {
    path: '',
    canActivate: [authGuard],
    loadComponent: () =>
      import('./layout/app-shell.component').then((module) => module.AppShellComponent),
    children: [
      { path: '', pathMatch: 'full', redirectTo: 'dashboard' },
      {
        path: 'dashboard',
        title: 'Tổng quan | Vehicle Intelligence',
        loadComponent: () =>
          import('./features/dashboard/dashboard.component').then(
            (module) => module.DashboardComponent
          )
      },
      {
        path: 'live-monitor',
        title: 'Live Monitor | Vehicle Intelligence',
        loadComponent: () =>
          import('./features/live-monitor/live-monitor.component').then(
            (module) => module.LiveMonitorComponent
          )
      },
      {
        path: 'events',
        title: 'Sự kiện phương tiện | Vehicle Intelligence',
        loadComponent: () =>
          import('./features/events/events.component').then((module) => module.EventsComponent)
      },
      {
        path: 'vehicle-search',
        title: 'Tra cứu biển số | Vehicle Intelligence',
        loadComponent: () =>
          import('./features/vehicle-search/vehicle-search.component').then(
            (module) => module.VehicleSearchComponent
          )
      },
      {
        path: 'vehicles/:vehicleId',
        title: 'Chi tiết phương tiện | Vehicle Intelligence',
        loadComponent: () =>
          import('./features/vehicle-detail/vehicle-detail.component').then(
            (module) => module.VehicleDetailComponent
          )
      },
      {
        path: 'ocr-review',
        title: 'Duyệt OCR | Vehicle Intelligence',
        loadComponent: () =>
          import('./features/ocr-review/ocr-review.component').then(
            (module) => module.OcrReviewComponent
          )
      },
      {
        path: 'dataset-review',
        title: 'Duyệt detector dataset | Vehicle Intelligence',
        loadComponent: () =>
          import('./features/dataset-review/dataset-review.component').then(
            (module) => module.DatasetReviewComponent
          )
      },
      {
        path: 'datasets',
        title: 'Quản lý Dataset | Vehicle Intelligence',
        loadComponent: () =>
          import('./features/dataset-management/dataset-management.component').then(
            (module) => module.DatasetManagementComponent
          )
      },
      {
        path: 'cameras',
        title: 'Camera | Vehicle Intelligence',
        loadComponent: () =>
          import('./features/cameras/cameras.component').then(
            (module) => module.CamerasComponent
          )
      },
      {
        path: 'alerts',
        title: 'Cảnh báo | Vehicle Intelligence',
        loadComponent: () =>
          import('./features/alerts/alerts.component').then((module) => module.AlertsComponent)
      },
      {
        path: 'watchlists',
        title: 'Danh sách xe | Vehicle Intelligence',
        loadComponent: () =>
          import('./features/watchlists/watchlists.component').then(
            (module) => module.WatchlistsComponent
          )
      },
      {
        path: 'rules',
        title: 'Luật tự động | Vehicle Intelligence',
        loadComponent: () =>
          import('./features/rules/rules.component').then((module) => module.RulesComponent)
      },
      {
        path: 'model-quality',
        title: 'Chất lượng AI | Vehicle Intelligence',
        loadComponent: () =>
          import('./features/model-quality/model-quality.component').then(
            (module) => module.ModelQualityComponent
          )
      },
      {
        path: 'system-health',
        title: 'Sức khỏe hệ thống | Vehicle Intelligence',
        loadComponent: () =>
          import('./features/system-health/system-health.component').then(
            (module) => module.SystemHealthComponent
          )
      }
    ]
  },
  { path: '**', redirectTo: 'dashboard' }
];
