# Frontend Architecture

## App Structure

The frontend is a Next.js App Router project using TypeScript, Tailwind CSS, and ESLint. The first screen is the usable research interface: project summary, workflow, upload flow, prediction result card, methodology, and disclaimer.

## Component Organization

- `src/app/` contains route-level files and global styles.
- `src/components/site/` contains layout elements such as the header and footer.
- `src/components/prediction/` contains upload and prediction display behavior.
- `src/components/ui/` contains small reusable primitives.
- `src/lib/` contains API boundaries and shared constants.
- `src/types/` contains shared TypeScript contracts.

## API Boundary

All backend communication should go through `src/lib/api.ts`. The current upload helper posts multipart `FormData` with a `file` field to `POST /segment/rbunet?return_image=true` using the required `NEXT_PUBLIC_API_BASE_URL` environment variable. Do not hardcode concrete server URLs in source code.

The frontend accepts a flexible prediction response to avoid coupling the UI too tightly to an evolving research API.

## Styling Approach

Styling uses Tailwind utility classes with a calm medical and AI research aesthetic: high contrast text, restrained borders, white and slate surfaces, cyan accents, and clear responsive spacing.

## Testing Approach

The current baseline is `npm run lint` and `npm run build`. If UI behavior expands, add focused Vitest and React Testing Library tests for upload state, API error handling, and result rendering.

## Boundaries

Frontend code should not mix in model training, DVC operations, MLflow orchestration, notebook logic, backend service logic, or private AI-assistant instructions.
