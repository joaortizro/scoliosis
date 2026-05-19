# Frontend Design System

This document defines the visual direction for the scoliosis frontend. It is inspired by the clean white, blue, and aviation-like clarity of Joby Aviation, adapted for a calmer medical research product.

## Personality

- Clean, calm, clinical, and confident.
- Spacious layouts with minimal decoration.
- Interfaces should feel precise and research-oriented, not busy or playful.
- Avoid heavy shadows, loud gradients, dense dashboards, or decorative clutter.

## Color Palette

| Role | Hex | Usage |
| --- | --- | --- |
| White | `#ffffff` | Primary page and navigation surfaces |
| Off white | `#fbfcfd` | App background |
| Soft blue surface | `#f4f8fb` | Secondary bands and quiet panels |
| Line | `#d9e5ee` | Borders and dividers |
| Primary blue | `#0a5f9e` | Primary actions, links, focus rings |
| Deep blue | `#102a43` | Logo, hero panels, major headings |
| Orange accent | `#f97316` | Small status accents and research callouts |
| Near black | `#111827` | Body text when strong contrast is needed |

Do not use pure black for large surfaces. Use `#102a43` or `#111827`.

## Typography

- Use the system UI stack for now: `Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif`.
- Large landing headings should be confident and simple.
- Body copy should stay readable with generous line height.
- Avoid tight letter spacing except for the centered logo wordmark.

## Layout

- Max content width: `max-w-7xl`.
- Page padding: `px-5` on mobile and `lg:px-8` on desktop.
- Use full-width white or soft-blue sections instead of nested page cards.
- Cards are reserved for individual tools, repeated items, and result panels.
- Border radius should usually be `rounded-md` or `rounded-lg`.

## Navigation

- Header structure:
  - Left: hamburger menu.
  - Center: logo wordmark.
  - Right: login button.
- The hamburger menu opens simple placeholder navigation links.
- The login button currently links to `/prediction` and does not perform authentication.

## Components

- Primary buttons use deep/primary blue with white text.
- Secondary buttons are white with a soft blue-gray border.
- Badges use a pale blue background with primary blue text.
- Warning and disclaimer blocks use orange lightly, not as the dominant color.

## Page Structure

- `/` is the landing page.
- `/prediction` is the prediction workspace.
- Prediction UI should remain independent from backend internals.

## Medical Disclaimer

The disclaimer must remain visible anywhere prediction behavior is presented:

> This tool is for research and educational purposes only. It is not a medical diagnosis tool.
