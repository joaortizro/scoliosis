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
NEXT_PUBLIC_API_BASE_URL=
DEMO_USERNAME=
DEMO_PASSWORD=
```

`NEXT_PUBLIC_API_BASE_URL` is required for image submission. Keep concrete
server URLs and demo credentials in `.env.local`, not in committed source code.

## Backend Connection

The upload flow posts a `FormData` payload to:

```text
POST /segment/rbunet?return_image=true
```

The helper lives in `src/lib/api.ts` and sends the uploaded image as multipart
`FormData` using the `file` field. It accepts flexible response fields because
the backend prediction schema may evolve during research.

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
