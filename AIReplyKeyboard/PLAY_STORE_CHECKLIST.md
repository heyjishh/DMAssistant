# Play Store Compliance Checklist

Everything below maps to what Google Play requires for this app category. Work through
it before submitting.

## 1. Accessibility API declaration (the big one)

The app reads screen content via `AccessibilityService` for a **non-accessibility**
purpose (capturing chat text). Play allows this but requires a declaration:

- Play Console → App content → **Sensitive app categories → Accessibility API**
- Declare the use and provide:
  - **What** the data is used for: reading visible chat text to generate reply
    suggestions, only on explicit user action (✦ tap).
  - **Why** an accessibility service is required: there is no public API to read
    another app's visible text; screenshot+OCR would be strictly worse for privacy
    (full-screen capture) and battery.
  - A **demo video** showing the feature and the in-app disclosure.
- In-app **prominent disclosure** (required *before* the Accessibility prompt): add a
  dialog on first launch that states "AI Reply reads on-screen text to suggest replies.
  Nothing is stored; text is only sent to the AI provider you configure" with
  Accept/Decline. *(Not yet implemented — add before submitting.)*

## 2. Data safety form (Play Console → App content → Data safety)

| Data type | Collected | Shared | Notes |
|---|---|---|---|
| Text of conversations | Transiently, on-device | Yes — to the user-configured AI provider | Only on ✦/✎/analyzer action; never stored by the app |
| Screenshots | Transiently, on-device | Yes — same as above | Only when user picks one |
| App interactions | No | No | No analytics SDKs exist in the app |
| Device ID / ads ID | No | No | — |

Declare **encryption in transit: yes** (HTTPS except user-configured http servers),
**deletion mechanism: yes** (uninstall / clear data), **no data sold**.

## 3. Content rating

Answer the IARC questionnaire; the "Spicy" tone implies **Mature 17+** is the likely
outcome. Mention user-selectable adult-toned text generation.

## 4. Target audience

Do **not** select under-13 audiences given the Spicy tone.

## 5. Release signing

- `release.keystore` + `keystore.properties` are gitignored — **back both up**; losing
  the key means you can never update the app under the same listing.
- Change the placeholder password `AIR3PLY-CHANGE-ME` in `keystore.properties` and
  re-generate the keystore with your own password before real use:
  ```bash
  keytool -genkeypair -v -keystore release.keystore -alias aireply \
      -keyalg RSA -keysize 2048 -validity 10000 -dname "CN=AI Reply"
  ```
- The signed bundle: `./gradlew bundleRelease` → `app/build/outputs/bundle/release/`.

## 6. Privacy policy URL

Host `PRIVACY_POLICY.md` publicly (GitHub Pages / a gist URL works) and paste that URL
into Play Console → App content → **Privacy policy**.

## 7. Before-release housekeeping

- `TestHookReceiver` already exists **only in debug builds** (`app/src/debug/`) — verify
  with `aapt2 dump xmltree --file AndroidManifest.xml app-release.aab | grep TestHook`
  (should output nothing).
- Bump `versionCode` / `versionName` for every submission.
