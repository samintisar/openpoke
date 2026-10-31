export const runtime = 'nodejs';

const upstream = (path: string) => {
  const serverBase = process.env.PY_SERVER_URL || 'http://localhost:8001';
  return `${serverBase.replace(/\/$/, '')}/api/v1/fake-gmail${path}`;
};

const relay = async (resp: Response) => {
  const data = await resp.json().catch(() => ({}));
  return new Response(JSON.stringify(data), {
    status: resp.status,
    headers: { 'Content-Type': 'application/json; charset=utf-8' },
  });
};

const upstreamError = (e: any) =>
  new Response(JSON.stringify({ ok: false, error: 'Upstream error', detail: e?.message || String(e) }), {
    status: 502,
    headers: { 'Content-Type': 'application/json; charset=utf-8' },
  });

export async function GET() {
  try {
    return await relay(await fetch(upstream(''), { cache: 'no-store', headers: { Accept: 'application/json' } }));
  } catch (e: any) {
    return upstreamError(e);
  }
}

export async function POST(req: Request) {
  let body: any = {};
  try {
    body = await req.json();
  } catch {}
  try {
    const resp = await fetch(upstream('/messages'), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
      body: JSON.stringify({
        sender: body?.sender || '',
        subject: body?.subject || '',
        body: body?.body || '',
        thread_id: body?.threadId || null,
      }),
    });
    return await relay(resp);
  } catch (e: any) {
    return upstreamError(e);
  }
}
