# 배포 가이드 — Trader Hub

전제: Cloudflare 계정 `cf2d2ff56ad968ff1845660517b88fe7` (kr-v25 / g5-22-dashboard와 동일).
`wrangler` 로그인되어 있어야 함 (`npx wrangler login`).

## 1. 통합 사이트 새 배포 (trader-hub)

```bash
cd unified_dashboard
python build.py                                       # index.html 최신화
npx wrangler@4 pages deploy . --project-name=trader-hub --commit-dirty=true
# 첫 배포 시 프로젝트 자동 생성 → https://trader-hub-bj0.pages.dev
```

## 2. KR 데이터 CORS 허용 (kr-v25 재배포)

`kr_dashboard/_headers`(CORS) + `_redirects`(루트만 redirect, /data는 서빙 유지)를 추가했다.
다음 KR GitHub Actions(매일 KST 16:00) 실행 시 자동 반영되거나, 즉시 반영하려면:

```bash
cd ..                                                 # stock_predictor 루트
npx wrangler@4 pages deploy kr_dashboard --project-name=kr-v25 --commit-dirty=true
```

확인:
```bash
curl -sI https://kr-v25.pages.dev/data/daily.json | grep -i access-control   # → Access-Control-Allow-Origin: *
curl -sI https://kr-v25.pages.dev/                  | grep -i location        # → trader-hub-bj0.pages.dev
```

## 3. US 옛 사이트 redirect (g5-22-dashboard 재배포)

`g5-22-trader/pages/public/_redirects`를 `/* → trader-hub` 로 교체했다.

```bash
cd ../g5-22-trader/pages
npx wrangler pages deploy public --project-name=g5-22-dashboard
```

## 검증 체크리스트

- [ ] `trader-hub-bj0.pages.dev` 열림, 상단 탭 🇺🇸/🇰🇷 전환됨
- [ ] US 탭: VIX/SPY/QQQ·모드·picks 로딩 (worker fetch)
- [ ] KR 탭: Zone/Lev/Top-7 로딩 (kr-v25 data fetch, CORS 통과)
- [ ] 탭 선택이 새로고침 후에도 유지됨 (localStorage)
- [ ] `g5-22-dashboard.pages.dev` → trader-hub로 redirect
- [ ] `kr-v25.pages.dev/` → redirect, 단 `/data/daily.json`은 정상 서빙

## 주의

- 백엔드(worker, GitHub Actions)는 **변경 없음**. KR 데이터는 계속 kr-v25에 쌓인다.
- 대시보드 로직을 고치려면 각 원본(g5 index.html / kr_dashboard)을 고치고 `python build.py` 재실행 후 1번 재배포.
