import { Component, OnDestroy, OnInit, effect, inject, signal } from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { NavigationEnd, Router, RouterLink, RouterLinkActive, RouterOutlet } from '@angular/router';
import {
  LucideActivity,
  LucideBell,
  LucideBoxes,
  LucideCamera,
  LucideCar,
  LucideCctv,
  LucideChevronDown,
  LucideCircleUserRound,
  LucideDatabase,
  LucideGitBranch,
  LucideGauge,
  LucideLayoutDashboard,
  LucideListChecks,
  LucideLogOut,
  LucideMenu,
  LucideServer,
  LucideSearch,
  LucideScanText,
  LucideX
} from '@lucide/angular';
import { filter } from 'rxjs';

import { AuthService } from '../core/auth/auth.service';
import { RealtimeService } from '../core/realtime/realtime.service';

@Component({
  selector: 'app-shell',
  imports: [
    RouterOutlet,
    RouterLink,
    RouterLinkActive,
    LucideActivity,
    LucideBell,
    LucideBoxes,
    LucideCamera,
    LucideCar,
    LucideCctv,
    LucideChevronDown,
    LucideCircleUserRound,
    LucideDatabase,
    LucideGitBranch,
    LucideGauge,
    LucideLayoutDashboard,
    LucideListChecks,
    LucideLogOut,
    LucideMenu,
    LucideServer,
    LucideSearch,
    LucideScanText,
    LucideX
  ],
  templateUrl: './app-shell.component.html'
})
export class AppShellComponent implements OnInit, OnDestroy {
  private readonly router = inject(Router);

  readonly auth = inject(AuthService);
  readonly realtime = inject(RealtimeService);
  readonly menuOpen = signal(false);
  readonly datasetMenuOpen = signal(false);
  readonly datasetSectionActive = signal(false);

  constructor() {
    this.updateDatasetSection(this.router.url);
    this.router.events
      .pipe(
        filter((event): event is NavigationEnd => event instanceof NavigationEnd),
        takeUntilDestroyed()
      )
      .subscribe((event) => {
        this.menuOpen.set(false);
        this.updateDatasetSection(event.urlAfterRedirects);
      });
    effect(() => {
      if (this.auth.state() === 'anonymous') void this.router.navigate(['/login']);
    });
  }

  ngOnInit(): void {
    this.realtime.connect();
  }

  ngOnDestroy(): void {
    this.realtime.disconnect();
  }

  toggleMenu(): void {
    this.menuOpen.update((open) => !open);
  }

  toggleDatasetMenu(): void {
    this.datasetMenuOpen.update((open) => !open);
  }

  logout(): void {
    this.realtime.disconnect();
    this.auth.logout();
    void this.router.navigate(['/login']);
  }

  private updateDatasetSection(url: string): void {
    const path = url.split(/[?#]/, 1)[0];
    const active = path === '/datasets' || path === '/dataset-review';
    this.datasetSectionActive.set(active);
    if (active) this.datasetMenuOpen.set(true);
  }
}
