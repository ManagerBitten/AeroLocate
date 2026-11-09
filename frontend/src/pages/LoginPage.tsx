import { FormEvent, useState } from "react";
import { login, register } from "../services/api";

export function LoginPage({ onSuccess }: { onSuccess: () => void }) {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [creating, setCreating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function submit(event: FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError(null);
    try {
      if (creating) await register(email, password);
      await login(email, password);
      onSuccess();
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : "Sign-in failed");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="flex min-h-full items-center justify-center p-4">
      <form onSubmit={submit} className="panel w-full max-w-sm space-y-4">
        <div>
          <h1 className="text-xl font-semibold">AeroLocate</h1>
          <p className="mt-1 text-sm text-ink-muted">Doppler analysis of recordings you own or are authorized to analyze.</p>
        </div>
        <div>
          <label className="label" htmlFor="email">
            Email
          </label>
          <input id="email" className="field" type="email" required value={email} onChange={(e) => setEmail(e.target.value)} />
        </div>
        <div>
          <label className="label" htmlFor="password">
            Password
          </label>
          <input
            id="password"
            className="field"
            type="password"
            required
            minLength={creating ? 12 : undefined}
            value={password}
            onChange={(e) => setPassword(e.target.value)}
          />
          {creating && <p className="mt-1 text-xs text-ink-muted">At least 12 characters.</p>}
        </div>
        {error && <p className="text-sm text-[#e66767]">{error}</p>}
        <button className="button w-full" disabled={busy}>
          {creating ? "Create account" : "Sign in"}
        </button>
        <button type="button" className="w-full text-sm text-ink-soft hover:text-ink" onClick={() => setCreating(!creating)}>
          {creating ? "I already have an account" : "Create an account"}
        </button>
      </form>
    </div>
  );
}
