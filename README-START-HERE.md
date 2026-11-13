# TWINS EYE website source

This archive contains the TWINS EYE web app source, its workspace package dependency, and the package-manager configuration needed to install it. It does not include installed dependencies or the separate animated-film artifact.

## Run locally

Requires Node.js and pnpm.

```sh
pnpm install
PORT=5173 BASE_PATH=/ pnpm --filter @workspace/twins-eye run dev
```

Open the local address printed by Vite. The app runs pose analysis in the browser and downloads the MediaPipe model and WebAssembly runtime on first use; an internet connection is needed for that initial load.

## Project files

- `artifacts/twins-eye/` — website pages, components, styling, pose analysis, and app configuration.
- `lib/api-client-react/` — shared workspace package referenced by the app.
- `package.json`, `pnpm-workspace.yaml`, `pnpm-lock.yaml` — workspace and dependency setup.

For Replit-specific preview configuration, the source also includes the artifact metadata under `artifacts/twins-eye/.replit-artifact/`.
