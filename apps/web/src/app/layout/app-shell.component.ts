import {
  Component,
  ElementRef,
  HostListener,
  Injector,
  OnDestroy,
  OnInit,
  ViewChild,
  afterNextRender,
  effect,
  inject,
  signal
} from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { NavigationEnd, Router, RouterLink, RouterLinkActive, RouterOutlet } from '@angular/router';
import {
  LucideActivity,
  LucideBell,
  LucideBrainCircuit,
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
import { dialogTabTarget } from '../shared/accessibility/accessible-dialog.directive';

@Component({
  selector: 'app-shell',
  imports: [
    RouterOutlet,
    RouterLink,
    RouterLinkActive,
    LucideActivity,
    LucideBell,
    LucideBrainCircuit,
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
  private readonly injector = inject(Injector);
  private activePath = this.pathFromUrl(this.router.url);

  @ViewChild('mainContent', { read: ElementRef })
  private mainContent?: ElementRef<HTMLElement>;

  @ViewChild('menuButton', { read: ElementRef })
  private menuButton?: ElementRef<HTMLButtonElement>;

  @ViewChild('menuBackdrop', { read: ElementRef })
  private menuBackdrop?: ElementRef<HTMLButtonElement>;

  @ViewChild('sidebar', { read: ElementRef })
  private sidebar?: ElementRef<HTMLElement>;

  readonly auth = inject(AuthService);
  readonly realtime = inject(RealtimeService);
  readonly menuOpen = signal(false);
  readonly datasetMenuOpen = signal(false);
  readonly datasetSectionActive = signal(false);
  readonly vehicleSearchSectionActive = signal(false);

  constructor() {
    this.updateDatasetSection(this.router.url);
    this.router.events
      .pipe(
        filter((event): event is NavigationEnd => event instanceof NavigationEnd),
        takeUntilDestroyed()
      )
      .subscribe((event) => {
        const nextPath = this.pathFromUrl(event.urlAfterRedirects);
        const shouldFocusContent = this.menuOpen() || nextPath !== this.activePath;
        this.activePath = nextPath;
        this.menuOpen.set(false);
        this.updateDatasetSection(event.urlAfterRedirects);
        if (shouldFocusContent) {
          this.focusAfterRender(() => {
            if (!this.menuOpen()) this.focusMainContent();
          });
        }
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
    if (this.menuOpen()) {
      this.closeMenu();
      return;
    }
    this.menuOpen.set(true);
    this.focusAfterRender(() => {
      if (this.menuOpen()) this.firstMenuControl()?.focus();
    });
  }

  closeMenu(restoreFocus = true): void {
    if (!this.menuOpen()) return;
    this.menuOpen.set(false);
    if (restoreFocus) {
      this.focusAfterRender(() => {
        if (!this.menuOpen()) this.menuButton?.nativeElement.focus();
      });
    }
  }

  focusMainContent(event?: Event): void {
    event?.preventDefault();
    this.mainContent?.nativeElement.focus();
  }

  @HostListener('document:keydown.escape', ['$event'])
  closeMenuOnEscape(event: Event): void {
    const keyboardEvent = event as KeyboardEvent;
    if (!this.menuOpen() || keyboardEvent.defaultPrevented || keyboardEvent.isComposing) return;
    keyboardEvent.preventDefault();
    this.closeMenu();
  }

  @HostListener('document:keydown', ['$event'])
  trapMenuFocus(event: Event): void {
    const keyboardEvent = event as KeyboardEvent;
    if (!this.menuOpen() || keyboardEvent.key !== 'Tab' || keyboardEvent.defaultPrevented) return;
    const sidebar = this.sidebar?.nativeElement;
    if (!sidebar) return;
    const focusable = Array.from(
      sidebar.querySelectorAll<HTMLElement>(
        'a[href], button:not([disabled]), [tabindex]:not([tabindex="-1"])'
      )
    ).filter((element) => element.tabIndex >= 0);
    const backdrop = this.menuBackdrop?.nativeElement;
    if (backdrop) focusable.push(backdrop);
    const target = dialogTabTarget(
      focusable,
      sidebar.ownerDocument.activeElement,
      keyboardEvent.shiftKey
    );
    if (!target) return;
    keyboardEvent.preventDefault();
    target.focus();
  }

  @HostListener('window:resize')
  resetMenuForDesktop(): void {
    const view = this.sidebar?.nativeElement.ownerDocument.defaultView;
    if (this.menuOpen() && view?.matchMedia('(min-width: 981px)').matches) {
      this.closeMenu(false);
    }
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
    const path = this.pathFromUrl(url);
    const active =
      path === '/datasets' || path === '/dataset-review' || path === '/model-training';
    this.datasetSectionActive.set(active);
    this.vehicleSearchSectionActive.set(
      path === '/vehicle-search' || path.startsWith('/vehicles/')
    );
    if (active) this.datasetMenuOpen.set(true);
  }

  private firstMenuControl(): HTMLElement | null {
    return (
      this.sidebar?.nativeElement.querySelector<HTMLElement>(
        'a[href], button:not([disabled]), [tabindex]:not([tabindex="-1"])'
      ) ?? null
    );
  }

  private focusAfterRender(callback: () => void): void {
    afterNextRender(callback, { injector: this.injector });
  }

  private pathFromUrl(url: string): string {
    return url.split(/[?#]/, 1)[0];
  }
}
