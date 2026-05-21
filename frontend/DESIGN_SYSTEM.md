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
| White | `#ffffff` | Primary page, workspace, and empty structural backgrounds |
| Warm white | `#f5f4df` | Reserved warm tint; use sparingly, not for large empty gaps |
| Pale blue | `#f2f8ff` | Soft selected states, controls, and quiet workspace surfaces |
| Primary blue | `#007ae5` | Primary actions, links, focus rings, general spine fill |
| Dark blue | `#1c3f9a` | Logo, hero panels, major headings, general spine border |
| Alternate dark blue | `#073f73` | Deeper UI surfaces and hover states |
| Orange accent | `#ff5c00` | Small status accents and research callouts |
| Near black | `#0d1620` | Body text and strong contrast surfaces |
| Gray | `#c7c6b7` | Borders, dividers, muted controls |
| Medical teal | `#00d1c1` | Optional supporting accent when a second highlight is needed |

Do not use pure black for large surfaces. Use `#0d1620`, `#182433`, `#1c3f9a`, or `#073f73`.

For vertebra-level segmentation overlays, use the exported `VERTEBRA_COLORS` map in `src/lib/constants.ts`. Multiclass colors should follow a continuous neon-spectrum progression down the spine, with darker fills and brighter same-family borders. Keep mask and bounding-box strokes thin enough that they do not overpower the radiograph. Unknown labels fall back to the general spine colors: fill `#0078E5`, border `#1C3F98`.

## Typography

- Prefer Clarity City if the font file is later added locally. Until then, use Poppins from Google Fonts, then the system UI stack.
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
