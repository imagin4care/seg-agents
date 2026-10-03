// The manifest of a base agent as a server registers it: its image named by
// the digest it was pushed as (a tag can be moved, a digest cannot). The
// workflow "Agent images" runs it after each push. Whether the gateway takes
// the manifest is checked where it is registered (its tests read
// gateway/agents/). Node, nothing else.
//
//   node pin-manifest.mjs <manifest.json> <repository> <digest> > <name>.json
import { readFileSync } from 'node:fs';
import { pathToFileURL } from 'node:url';

/** `manifest` with its image at `repository@digest`. */
export function pinned(manifest, repository, digest) {
  if (!/^sha256:[0-9a-f]{64}$/.test(digest)) throw new Error(`not a digest: ${digest}`);
  if (/[@:]/.test(repository.slice(repository.lastIndexOf('/') + 1))) throw new Error(`a repository, without its tag: ${repository}`);
  if (manifest?.runtime?.type !== 'container') throw new Error('not the manifest of an agent in a container');
  return { ...manifest, runtime: { ...manifest.runtime, image: `${repository}@${digest}` } };
}

if (import.meta.url === pathToFileURL(process.argv[1] ?? '').href) {
  const [file, repository, digest] = process.argv.slice(2);
  process.stdout.write(`${JSON.stringify(pinned(JSON.parse(readFileSync(file, 'utf8')), repository, digest), null, 2)}\n`);
}
