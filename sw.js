const CACHE_VERSION = 'v1.0.4';
const STATIC_CACHE = `static-${CACHE_VERSION}`;
const STATIC_ASSETS = [
  './',
  './index.html',
  './styles.css',
  './app.js',
  './manifest.webmanifest',
  './icons/icon-192.png',
  './icons/icon-512.png',
  // CDN 스크립트는 오프라인 필수 아님. 초기엔 제외하고, 필요시 추가 캐시 가능.
];

self.addEventListener('install', e => {
  e.waitUntil(caches.open(STATIC_CACHE).then(c => c.addAll(STATIC_ASSETS)));
  self.skipWaiting();
});
self.addEventListener('activate', e => {
  e.waitUntil((async () => {
    const keys = await caches.keys();
    await Promise.all(keys.map(k => (k.startsWith('static-') && k !== STATIC_CACHE) ? caches.delete(k) : null));
    await self.clients.claim();
  })());
});
self.addEventListener('fetch', e => {
  const req = e.request;
  if (req.method !== 'GET') return;
  e.respondWith((async () => {
    // HTML 문서는 Network First (최신 UI)
    if (req.headers.get('accept')?.includes('text/html')) {
      try {
        const fresh = await fetch(req);
        const c = await caches.open(STATIC_CACHE);
        c.put(req, fresh.clone());
        return fresh;
      } catch {
        const cached = await caches.match(req);
        return cached ?? caches.match('./');
      }
    }
    // 정적 파일은 Cache First
    const cached = await caches.match(req);
    if (cached) return cached;
    const resp = await fetch(req);
    // 동일 출처 정적만 캐시
    if (new URL(req.url).origin === location.origin) {
      const c = await caches.open(STATIC_CACHE);
      c.put(req, resp.clone());
    }
    return resp;
  })());
});
