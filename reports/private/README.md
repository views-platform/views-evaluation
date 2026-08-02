# reports/private/ — non-public, gitignored

Everything in this directory is **gitignored** (except this README). It is the safe place
for report material that must not be published: anything containing credentials, API keys,
tokens, connection strings, partner data, personal/identifying information, or otherwise
non-public content.

The rest of `reports/` **is tracked** (engineering reports, the technical risk register,
post-mortems, plans). Do not paste secrets into those — put such files here instead.

Rules (`.gitignore`):

    reports/private/*
    !reports/private/README.md

So files you drop here are ignored automatically; only this README is committed.
