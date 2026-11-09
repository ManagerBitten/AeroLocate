import { useEffect, useState } from "react";
import { readToken } from "../services/api";
import type { ProgressMessage } from "../services/types";

const RECONNECT_MS = 2000;
const TERMINAL = ["completed", "failed", "cancelled", "rejected"];

export function useProgress(analysisId: string | null): ProgressMessage | null {
  const [message, setMessage] = useState<ProgressMessage | null>(null);

  useEffect(() => {
    if (!analysisId) return;
    let closed = false;
    let finished = false;
    let socket: WebSocket | null = null;
    let timer: number | undefined;

    const connect = () => {
      const scheme = window.location.protocol === "https:" ? "wss" : "ws";
      socket = new WebSocket(`${scheme}://${window.location.host}/api/analyses/${analysisId}/progress`);
      socket.onopen = () => socket?.send(JSON.stringify({ token: readToken() }));
      socket.onmessage = (event) => {
        const update = JSON.parse(event.data as string) as ProgressMessage;
        finished = TERMINAL.includes(update.state);
        setMessage(update);
      };
      socket.onclose = (event) => {
        if (!closed && !finished && event.code !== 4401) timer = window.setTimeout(connect, RECONNECT_MS);
      };
    };

    connect();
    return () => {
      closed = true;
      window.clearTimeout(timer);
      socket?.close();
    };
  }, [analysisId]);

  return message;
}
