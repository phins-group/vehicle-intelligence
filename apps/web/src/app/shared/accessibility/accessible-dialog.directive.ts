import { DOCUMENT } from '@angular/common';
import {
  AfterViewInit,
  Directive,
  ElementRef,
  HostListener,
  Injectable,
  OnDestroy,
  inject,
  output
} from '@angular/core';

const FOCUSABLE_SELECTOR = [
  'a[href]',
  'area[href]',
  'button:not([disabled])',
  'input:not([disabled]):not([type="hidden"])',
  'select:not([disabled])',
  'textarea:not([disabled])',
  'iframe',
  'object',
  'embed',
  'audio[controls]',
  'video[controls]',
  '[contenteditable="true"]',
  '[tabindex]'
].join(',');

interface InertRecord {
  count: number;
  originallyInert: boolean;
}

@Injectable({ providedIn: 'root' })
class DialogBackgroundManager {
  private readonly document = inject(DOCUMENT);
  private readonly records = new Map<HTMLElement, InertRecord>();
  private activeDialogs = 0;
  private originalBodyOverflow: string | null = null;

  acquire(dialog: HTMLElement): () => void {
    this.lockBodyScroll();
    const siblings = Array.from(dialog.parentElement?.children ?? [])
      .filter((element): element is HTMLElement => element !== dialog)
      .filter((element) => !element.hasAttribute('data-dialog-backdrop'));

    for (const sibling of siblings) {
      const current = this.records.get(sibling);
      if (current) {
        current.count += 1;
        continue;
      }
      this.records.set(sibling, {
        count: 1,
        originallyInert: sibling.hasAttribute('inert')
      });
      sibling.setAttribute('inert', '');
    }

    let released = false;
    return () => {
      if (released) return;
      released = true;
      for (const sibling of siblings) {
        const current = this.records.get(sibling);
        if (!current) continue;
        current.count -= 1;
        if (current.count > 0) continue;
        if (!current.originallyInert) sibling.removeAttribute('inert');
        this.records.delete(sibling);
      }
      this.unlockBodyScroll();
    };
  }

  private lockBodyScroll(): void {
    const body = this.document.body;
    if (this.activeDialogs === 0 && body) {
      this.originalBodyOverflow = body.style.overflow;
      body.style.overflow = 'hidden';
    }
    this.activeDialogs += 1;
  }

  private unlockBodyScroll(): void {
    if (this.activeDialogs === 0) return;
    this.activeDialogs -= 1;
    if (this.activeDialogs > 0) return;
    const body = this.document.body;
    if (body) body.style.overflow = this.originalBodyOverflow ?? '';
    this.originalBodyOverflow = null;
  }
}

/**
 * Returns the element that must receive focus to keep Tab navigation inside a dialog.
 * A null result lets the browser perform its normal in-dialog Tab movement.
 */
export function dialogTabTarget(
  focusable: readonly HTMLElement[],
  activeElement: Element | null,
  backwards: boolean
): HTMLElement | null {
  if (!focusable.length) return null;
  const activeIndex = focusable.findIndex((element) => element === activeElement);
  if (activeIndex === -1) return backwards ? focusable.at(-1)! : focusable[0];
  if (backwards && activeIndex === 0) return focusable.at(-1)!;
  if (!backwards && activeIndex === focusable.length - 1) return focusable[0];
  return null;
}

@Directive({
  selector: '[appAccessibleDialog]',
  standalone: true,
  host: {
    role: 'dialog',
    'aria-modal': 'true',
    tabindex: '-1',
    'data-accessible-dialog': ''
  }
})
export class AccessibleDialogDirective implements AfterViewInit, OnDestroy {
  private readonly document = inject(DOCUMENT);
  private readonly element = inject<ElementRef<HTMLElement>>(ElementRef).nativeElement;
  private readonly background = inject(DialogBackgroundManager);
  private readonly restoreTarget = this.document.activeElement as HTMLElement | null;
  private releaseBackground: (() => void) | null = null;
  private destroyed = false;

  readonly appDialogDismiss = output<void>();

  ngAfterViewInit(): void {
    this.releaseBackground = this.background.acquire(this.element);
    queueMicrotask(() => {
      if (this.destroyed) return;
      const preferred = this.element.querySelector<HTMLElement>('[data-dialog-initial-focus]');
      const target =
        (preferred && this.isFocusable(preferred) ? preferred : null) ??
        this.focusableElements()[0] ??
        this.element;
      target.focus();
    });
  }

  ngOnDestroy(): void {
    this.destroyed = true;
    this.releaseBackground?.();
    const target = this.restoreTarget;
    queueMicrotask(() => {
      if (target?.isConnected && !target.closest('[inert]')) target.focus();
    });
  }

  @HostListener('keydown', ['$event'])
  handleKeydown(event: KeyboardEvent): void {
    if (event.defaultPrevented || event.isComposing) return;
    if (event.key === 'Escape') {
      event.preventDefault();
      event.stopPropagation();
      this.appDialogDismiss.emit();
      return;
    }
    if (event.key !== 'Tab') return;

    const focusable = this.focusableElements();
    if (!focusable.length) {
      event.preventDefault();
      this.element.focus();
      return;
    }
    const target = dialogTabTarget(focusable, this.document.activeElement, event.shiftKey);
    if (!target) return;
    event.preventDefault();
    target.focus();
  }

  private focusableElements(): HTMLElement[] {
    return Array.from(this.element.querySelectorAll<HTMLElement>(FOCUSABLE_SELECTOR)).filter(
      (element) => this.isFocusable(element)
    );
  }

  private isFocusable(element: HTMLElement): boolean {
    if (element.tabIndex < 0 || element.closest('[hidden], [inert], [aria-hidden="true"]')) {
      return false;
    }
    const style = this.document.defaultView?.getComputedStyle(element);
    return style?.display !== 'none' && style?.visibility !== 'hidden';
  }
}
