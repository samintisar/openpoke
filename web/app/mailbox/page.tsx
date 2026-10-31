'use client';

import Link from 'next/link';
import { useCallback, useEffect, useState } from 'react';

const POLL_INTERVAL_MS = 3000;

type Mail = {
  id: string;
  thread_id: string;
  from: string;
  to: string[];
  subject: string;
  body: string;
  timestamp: string;
  labels: string[];
};

type Draft = { id: string; to: string[]; cc: string[]; subject: string; body: string };

type Sent = { kind: string; to: string[]; cc: string[]; bcc: string[]; subject: string; body: string; reason?: string };

type Blocked = { kind: string; recipients: string[]; reason: string };

type Snapshot = { owner: string; inbox: Mail[]; drafts: Draft[]; sent: Sent[]; blocked: Blocked[] };

const fmtTime = (iso: string) => {
  const date = new Date(iso);
  return Number.isNaN(date.getTime()) ? iso : date.toLocaleString();
};

function Section({ title, count, children }: { title: string; count: number; children: React.ReactNode }) {
  return (
    <section className="mb-6">
      <h2 className="mb-2 text-sm font-semibold uppercase tracking-wide text-gray-500">
        {title} <span className="font-normal">({count})</span>
      </h2>
      {count === 0 ? <p className="text-sm text-gray-400">Nothing yet.</p> : <div className="space-y-2">{children}</div>}
    </section>
  );
}

function Card({ head, sub, body, tone = 'default' }: { head: string; sub: string; body: string; tone?: 'default' | 'ok' | 'bad' }) {
  const border = tone === 'ok' ? 'border-green-200' : tone === 'bad' ? 'border-red-200' : 'border-gray-200';
  return (
    <div className={`rounded-md border ${border} p-3`}>
      <div className="text-sm font-medium">{head}</div>
      <div className="mb-1 text-xs text-gray-500">{sub}</div>
      {body ? <div className="whitespace-pre-wrap text-sm text-gray-700">{body}</div> : null}
    </div>
  );
}

export default function MailboxPage() {
  const [snapshot, setSnapshot] = useState<Snapshot | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [sender, setSender] = useState('Carl Brenner <carl@brennerbuild.test>');
  const [subject, setSubject] = useState('');
  const [body, setBody] = useState('');
  const [threadId, setThreadId] = useState('');
  const [sending, setSending] = useState(false);

  const load = useCallback(async () => {
    try {
      const res = await fetch('/api/fake-gmail', { cache: 'no-store' });
      const data = await res.json().catch(() => ({}));
      if (!res.ok || !data?.ok) {
        setError(data?.error || `Failed (${res.status})`);
        return;
      }
      setError(null);
      setSnapshot(data as Snapshot);
    } catch (e: any) {
      setError(e?.message || 'Failed to load the mailbox');
    }
  }, []);

  useEffect(() => {
    void load();
    const timer = setInterval(() => void load(), POLL_INTERVAL_MS);
    return () => clearInterval(timer);
  }, [load]);

  const deliver = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!sender.trim() || !body.trim()) return;
    setSending(true);
    try {
      const res = await fetch('/api/fake-gmail', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sender, subject, body, threadId: threadId || null }),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok || !data?.ok) {
        setError(data?.error || `Failed (${res.status})`);
        return;
      }
      setSubject('');
      setBody('');
      setThreadId('');
      await load();
    } finally {
      setSending(false);
    }
  };

  const threads = snapshot
    ? Array.from(new Map(snapshot.inbox.map((m) => [m.thread_id, m.subject || '(no subject)'])).entries())
    : [];

  return (
    <main className="mx-auto max-w-3xl p-4">
      <header className="mb-4 flex items-center justify-between">
        <div>
          <h1 className="text-lg font-semibold">Fake mailbox</h1>
          {snapshot ? <p className="text-xs text-gray-500">{snapshot.owner} · nothing here leaves your machine</p> : null}
        </div>
        <Link className="rounded-md border border-gray-200 px-3 py-2 text-sm hover:bg-gray-50" href="/">
          Back to chat
        </Link>
      </header>

      {error ? <div className="mb-4 rounded-md border border-red-200 bg-red-50 p-3 text-sm text-red-700">{error}</div> : null}

      <form onSubmit={deliver} className="mb-8 space-y-2 rounded-md border border-gray-200 p-3">
        <div className="text-sm font-medium">Send an email to the inbox</div>
        <input
          className="w-full rounded-md border border-gray-200 px-3 py-2 text-sm"
          placeholder="From, e.g. Dana Ruiz <dana@maplecourt.test>"
          value={sender}
          onChange={(e) => setSender(e.target.value)}
        />
        <input
          className="w-full rounded-md border border-gray-200 px-3 py-2 text-sm"
          placeholder="Subject"
          value={subject}
          onChange={(e) => setSubject(e.target.value)}
        />
        <textarea
          className="h-28 w-full rounded-md border border-gray-200 px-3 py-2 text-sm"
          placeholder="Body"
          value={body}
          onChange={(e) => setBody(e.target.value)}
        />
        <select
          className="w-full rounded-md border border-gray-200 px-3 py-2 text-sm"
          value={threadId}
          onChange={(e) => setThreadId(e.target.value)}
        >
          <option value="">New thread</option>
          {threads.map(([id, title]) => (
            <option key={id} value={id}>
              Reply in: {title}
            </option>
          ))}
        </select>
        <button
          type="submit"
          disabled={sending || !sender.trim() || !body.trim()}
          className="rounded-md bg-gray-900 px-3 py-2 text-sm text-white disabled:opacity-40"
        >
          {sending ? 'Delivering…' : 'Deliver'}
        </button>
        <p className="text-xs text-gray-500">The important-email watcher checks every 60 seconds, so an alert can take a minute.</p>
      </form>

      {snapshot ? (
        <>
          <Section title="Sent by the assistant" count={snapshot.sent.length}>
            {snapshot.sent.map((s, i) => (
              <Card
                key={`sent-${i}`}
                tone="ok"
                head={`${s.kind}: ${s.subject || '(no subject)'}`}
                sub={`to ${[...s.to, ...s.cc, ...s.bcc].join(', ')} · approval: ${s.reason || 'unknown'}`}
                body={s.body}
              />
            ))}
          </Section>
          <Section title="Blocked by the send gate" count={snapshot.blocked.length}>
            {snapshot.blocked.map((b, i) => (
              <Card key={`blocked-${i}`} tone="bad" head={`${b.kind} blocked: ${b.reason}`} sub={`to ${b.recipients.join(', ')}`} body="" />
            ))}
          </Section>
          <Section title="Drafts" count={snapshot.drafts.length}>
            {snapshot.drafts.map((d) => (
              <Card key={d.id} head={d.subject || '(no subject)'} sub={`to ${[...d.to, ...d.cc].join(', ')}`} body={d.body} />
            ))}
          </Section>
          <Section title="Inbox" count={snapshot.inbox.length}>
            {snapshot.inbox.map((m) => (
              <Card key={m.id} head={m.subject || '(no subject)'} sub={`from ${m.from} · ${fmtTime(m.timestamp)}`} body={m.body} />
            ))}
          </Section>
        </>
      ) : null}
    </main>
  );
}
