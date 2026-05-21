# Scoliosis AI Frontend

This frontend is a small Next.js interface for a master's project on AI-powered scoliosis detection from spine X-ray images. It provides a polished home page, a research disclaimer, an image upload flow, and prepared integration with the FastAPI prediction endpoint.

## Setup

```bash
npm install
npm run dev
```

The app runs at `http://localhost:3000` by default.

## Commands

```bash
npm run dev
npm run build
npm run lint
npm run start
```

## Environment

Create `frontend/.env.local`:

```bash
BACKEND_API_BASE_URL=
NEXT_PUBLIC_BACKEND_API_BASE_URL=
BACKEND_API_AUTH_TOKEN=
LEGACY_BACKEND_API_BASE_URL=
DEMO_USERNAME=
DEMO_PASSWORD=
```

`BACKEND_API_BASE_URL` is required for image submission. Keep concrete server
URLs and demo credentials in `.env.local`, not in committed source code.
`NEXT_PUBLIC_BACKEND_API_BASE_URL` is an optional browser fallback for static
deployments. `BACKEND_API_AUTH_TOKEN` is optional and is only used server-side
by the Next.js API proxy when a backend such as a private Hugging Face Space
requires bearer authentication.
`LEGACY_BACKEND_API_BASE_URL` is optional and powers the separate original
RBUNet preview action without changing the main prediction backend.
The demo username and password are checked by the local Next.js API route before
the protected prediction workspace is shown.

## Backend Connection

All browser-side prediction calls go through same-origin Next.js API routes.
The helper lives in `src/lib/api.ts` and sends the uploaded image as multipart
`FormData` using the `file` field.

### Current DeepLab API

The main prediction workspace posts to:

```text
POST /api/model-predict?model=full|multiclass|binary
```

That route reads `BACKEND_API_BASE_URL` and forwards to the deployed DeepLab
backend:

```text
POST /predict-full
POST /predict-multiclass
POST /predict-binary
```

If the backend is a private Hugging Face Space, set `BACKEND_API_AUTH_TOKEN`
server-side so the proxy can add bearer authentication without exposing the
token to the browser.

### Legacy RBUNet Preview

The separate original-model preview action posts to:

```text
POST /api/legacy-segment-rbunet
```

That route reads `LEGACY_BACKEND_API_BASE_URL` and forwards to:

```text
POST /segment/rbunet?return_image=true
```

The legacy response is expected to include a base64 result image, usually under
`image_base64` or `data.image_base64`, which the frontend displays in a modal.

The older same-origin route `POST /api/segment-rbunet` still exists for RBUNet
compatibility, but the active workspace uses the DeepLab route for the main
prediction flow and the legacy route only for the extra preview.

## Demo Access

`/prediction` is protected by a lightweight demo session. Unauthenticated users
are redirected to `/login`. The login route posts to `POST /api/demo-login`,
which compares the submitted values with `DEMO_USERNAME` and `DEMO_PASSWORD` and
sets an HTTP-only demo cookie. `POST /api/demo-logout` clears that cookie.

## Folder Structure

```text
src/app/                  App Router pages and global styles
src/components/site/      Header and footer
src/components/prediction Upload and result components
src/components/ui/        Small reusable UI primitives
src/lib/                  API helpers and constants
src/types/                Shared TypeScript types
```

## Medical Disclaimer

This tool is for research and educational purposes only. It is not a medical diagnosis tool.
