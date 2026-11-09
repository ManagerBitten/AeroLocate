# Security

Please report vulnerabilities privately through GitHub's "Report a vulnerability" button on the Security tab, not in a public issue. Include the steps to reproduce the problem and the version or commit you tested.

AeroLocate is meant for trusted, private deployments. Before exposing it to a network:

- Put TLS in front of nginx
- Set `AEROLOCATE_ALLOW_REGISTRATION=false` once your accounts exist
- Keep `.env` out of version control. It holds your database password, token secret and API keys.
