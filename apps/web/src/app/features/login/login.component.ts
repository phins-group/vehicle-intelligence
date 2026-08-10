import { Component, OnInit, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { ActivatedRoute, Router } from '@angular/router';
import {
  LucideCar,
  LucideEye,
  LucideEyeOff,
  LucideKeyRound,
  LucideShieldCheck
} from '@lucide/angular';

import { AuthService } from '../../core/auth/auth.service';
import { apiErrorMessage } from '../../core/utils/api-error';

@Component({
  selector: 'app-login',
  imports: [FormsModule, LucideCar, LucideEye, LucideEyeOff, LucideKeyRound, LucideShieldCheck],
  templateUrl: './login.component.html'
})
export class LoginComponent implements OnInit {
  readonly auth = inject(AuthService);
  readonly busy = signal(false);
  readonly error = signal<string | null>(null);
  readonly showKey = signal(false);
  apiKey = '';
  private readonly router = inject(Router);
  private readonly route = inject(ActivatedRoute);

  ngOnInit(): void {
    if (this.auth.isAuthenticated()) void this.navigateAfterLogin();
  }

  async submit(): Promise<void> {
    if (this.busy()) return;
    this.busy.set(true);
    this.error.set(null);
    try {
      await this.auth.login(this.apiKey);
      this.apiKey = '';
      await this.navigateAfterLogin();
    } catch (error) {
      this.error.set(apiErrorMessage(error, 'API key không hợp lệ.'));
    } finally {
      this.busy.set(false);
    }
  }

  toggleVisibility(): void {
    this.showKey.update((visible) => !visible);
  }

  private navigateAfterLogin(): Promise<boolean> {
    const requested = this.route.snapshot.queryParamMap.get('returnUrl');
    const target = requested?.startsWith('/') && !requested.startsWith('//') ? requested : '/dashboard';
    return this.router.navigateByUrl(target);
  }
}
