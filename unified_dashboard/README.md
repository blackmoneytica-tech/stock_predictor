# 📊 Trader Hub — 통합 대시보드 (US G5_22 + KR V25)

미국(G5_22)과 한국(V25) 두 전략 대시보드를 **단일 페이지 + 탭 전환**으로 통합한 사이트.
→ 배포: `trader-hub-bj0.pages.dev`

## 설계 원칙 — 백엔드 무수정

두 백엔드(Cloudflare Worker, GitHub Actions)는 **전혀 건드리지 않는다.** 통합 사이트는
프론트엔드 셸일 뿐이고, 데이터는 각자의 기존 출처에서 그대로 가져온다.

```
trader-hub-bj0.pages.dev (단일 index.html)
 ├─ 🇺🇸 US 탭  → fetch  g5-22-trader.blackmoneytica.workers.dev/api/*   (기존 worker, CORS 이미 작동)
 └─ 🇰🇷 KR 탭  → fetch  kr-v25.pages.dev/data/*.json                    (기존 데이터, _headers로 CORS 허용)
```

- US 연산: Cloudflare Worker (cron, 그대로)
- KR 연산: `stock_predictor` GitHub Actions → `kr_dashboard/data/*.json` (그대로). kr-v25는 이제 **데이터 SSOT 서버** 역할.

## 충돌 격리 (단일 DOM 병합)

| 충돌 | 해결 |
|---|---|
| CSS 클래스 (`.card .value .ticker .num ...` 양쪽 중복) | 모든 셀렉터를 `#view-us` / `#view-kr` 하위로 prefix |
| ID 3개 (`last-update` `picks-table` `picks-body`) | KR쪽만 `kr-` prefix로 rename |
| `saveCapital()` 전역 함수 충돌 | KR JS 전체를 IIFE로 격리 (US는 인라인 onclick 때문에 전역 유지) |
| KR `fetch('data/...')` 상대경로 | `fetch(KR_BASE + '/data/...')` cross-origin으로 재작성 |

`@keyframes pulse`(US) / `pulse-hb`(KR)는 이름이 달라 전역 유지.

## 빌드

`index.html`은 **생성물**이다. 직접 수정하지 말고 원본을 고친 뒤 재빌드한다.

```bash
cd unified_dashboard
python build.py     # → index.html 재생성
```

원본:
- US: `../../g5-22-trader/pages/public/index.html` (inline style/body/script)
- KR: `../kr_dashboard/index.html` + `style.css` + `script.js`

US/KR 대시보드를 각자 고치면, 재빌드만 하면 통합 사이트에 반영된다.

## 배포

`DEPLOY.md` 참조.
