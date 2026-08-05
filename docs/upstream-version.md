# Upstream Version Audit

## Finding

The repository currently declares itself as `p2pfl` version `0.4.4` in both project metadata and the lockfile:

- `pyproject.toml` has `name = "p2pfl"` and `version = "0.4.4"`.
- `uv.lock` records the editable local package as `p2pfl` version `0.4.4` with `source = { editable = "." }`.

The configured package metadata points to the upstream project at `https://github.com/p2pfl/p2pfl`, but this clone has no configured Git remotes and no tags. The local branch is `work` at commit `0497364 final version`.

## Evidence collected

Commands run:

```sh
git remote -v
```

Result: succeeded, with no output; no remotes are configured.

```sh
git branch -vv
```

Result: succeeded, showing `* work 0497364 final version`.

```sh
git log --oneline -5
```

Result: succeeded:

```text
0497364 final version
35270a1 final correct model replacement
9a06873 model_replacement attack module
48d8e68 add scale attack module
ad8a103 test analyze_results
```

```sh
rg -n "version = \"0.4.4\"|p2pfl|source|revision" pyproject.toml uv.lock
```

Result: succeeded and found the local project version in both files.

```sh
git ls-remote --tags https://github.com/p2pfl/p2pfl.git 'refs/tags/*'
```

Result: failed due environment network/proxy restriction:

```text
fatal: unable to access 'https://github.com/p2pfl/p2pfl.git/': CONNECT tunnel failed, response 403
```

## Conclusion

The exact upstream commit cannot be proven from local metadata alone because this clone lacks remotes, tags, and an upstream commit reference. The strongest local evidence is that the thesis repository is based on upstream P2PFL release/package version `0.4.4`, then modified on local branch `work` through thesis commits beginning with `cb3abe5 base set up of p2pfl and labelflip and sign flip attacks`.

Recommended follow-up when network access is available:

1. Add or fetch upstream without changing working files: `git fetch https://github.com/p2pfl/p2pfl.git --tags upstream-audit`.
2. Compare this tree with the upstream `v0.4.4` tag if present: `git diff --stat v0.4.4...HEAD`.
3. Use `git merge-base HEAD v0.4.4` or `git describe --tags --contains <base>` to identify the exact fork point.
