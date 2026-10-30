# Privacy Policy — AI Reply

**Last updated: 2026-10-31**

AI Reply ("the app") is an AI-assisted keyboard and reply suggestion tool. This policy
explains exactly what the app does with your data. The short version: **the app has no
backend, stores almost nothing, and the only party that sees your conversations is the
AI provider you personally configure.**

## What the app accesses

1. **On-screen text (Accessibility Service).** To suggest replies, the app reads the
   visible text of the app currently on screen (for example, a chat conversation).
   - The text is held **only in memory**, never written to storage, and is automatically
     cleared when you open an app on the blocklist you define in Settings.
   - Screenshots are only processed when **you** explicitly pick one in the Screenshot
     Analyzer; the app never takes screenshots on its own.
2. **What you type (Keyboard / Input Method).** Like any keyboard, the app receives the
   keys you press to insert text. The app does **not** log keystrokes.
3. **Settings you enter.** API provider, API key, models, style profile, tone choices,
   and saved replies are stored **locally on your device** in the app's private storage.
   They never leave the device except as described below.

## What leaves your device

- When you tap the generate (✦) / rewrite (✎) buttons, or use the Screenshot Analyzer,
  the relevant text (and for screenshots, the image) is sent over the internet to the
  **AI provider you configured** (e.g. OpenAI, Google Gemini, OpenRouter, or your own
  self-hosted server). That provider processes the request under **its own privacy
  policy** — read it before sending anything sensitive.
- The app's developers do **not** operate any server, do **not** receive copies, and do
  **not** have access to your API key or conversations.

## What is collected

- **No analytics. No ads. No trackers. No account.**
- Data stored locally: API settings, tone-per-app preferences, saved replies, and the
  exclusion list. You can erase everything by uninstalling the app or clearing its data.

## Children

The app is not directed at children under 13. The "Spicy" tone is intended for adults
and is off by default in the sense that it is only used when you select it.

## Self-hosted providers

If you configure a personal server (such as gemini-web2api), conversation text is sent
to that server. Its handling of data depends entirely on your own setup.

## Changes

Material changes to this policy will be noted in the app's release notes.

## Contact

Open an issue on the project's repository, or contact the developer directly.
