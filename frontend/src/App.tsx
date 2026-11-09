import { lazy, Suspense, useCallback, useEffect, useState } from "react";
import { LoginPage } from "./pages/LoginPage";
import { listAnalyses, readToken, storeToken } from "./services/api";
import { STATUS_LABELS } from "./services/format";
import type { AnalysisSummary } from "./services/types";

const AnalysisPage = lazy(() => import("./pages/AnalysisPage").then((module) => ({ default: module.AnalysisPage })));
const UploadPage = lazy(() => import("./pages/UploadPage").then((module) => ({ default: module.UploadPage })));

export function App() {
  const [authenticated, setAuthenticated] = useState(Boolean(readToken()));
  const [selected, setSelected] = useState<string | null>(null);
  const [history, setHistory] = useState<AnalysisSummary[]>([]);
  const [revision, setRevision] = useState(0);

  const refreshHistory = useCallback(() => setRevision((value) => value + 1), []);

  useEffect(() => {
    if (!authenticated) return;
    listAnalyses()
      .then(setHistory)
      .catch(() => setAuthenticated(Boolean(readToken())));
  }, [authenticated, selected, revision]);

  if (!authenticated) return <LoginPage onSuccess={() => setAuthenticated(true)} />;

  return (
    <div className="flex min-h-full flex-col">
      <header className="flex items-center justify-between border-b border-surface-line bg-surface px-4 py-3">
        <button className="text-base font-semibold tracking-tight" onClick={() => setSelected(null)}>
          AeroLocate
        </button>
        <button
          className="button-quiet"
          onClick={() => {
            storeToken(null);
            setAuthenticated(false);
          }}
        >
          Sign out
        </button>
      </header>
      <div className="flex flex-1 flex-col lg:flex-row">
        <aside className="border-b border-surface-line bg-surface p-3 lg:w-64 lg:border-b-0 lg:border-r">
          <button className="button mb-3 w-full" onClick={() => setSelected(null)}>
            New analysis
          </button>
          <ul className="max-h-48 space-y-1 overflow-y-auto lg:max-h-none">
            {history.map((item) => (
              <li key={item.id}>
                <button
                  className={`w-full rounded-md px-2 py-1.5 text-left text-sm hover:bg-surface-raised ${
                    item.id === selected ? "bg-surface-raised" : ""
                  }`}
                  onClick={() => setSelected(item.id)}
                >
                  <span className="block truncate">{item.filename}</span>
                  <span className="block text-xs text-ink-muted">
                    {item.localization_status ? STATUS_LABELS[item.localization_status] : item.state}
                  </span>
                </button>
              </li>
            ))}
          </ul>
        </aside>
        <main className="min-w-0 flex-1 p-4">
          <Suspense fallback={<p className="text-ink-muted">Loading.</p>}>
            {selected ? (
              <AnalysisPage
                key={selected}
                analysisId={selected}
                onDeleted={() => setSelected(null)}
                onStateChanged={refreshHistory}
              />
            ) : (
              <UploadPage onCreated={setSelected} />
            )}
          </Suspense>
        </main>
      </div>
    </div>
  );
}
