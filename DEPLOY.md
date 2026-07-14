# Deploy: scheduled daily digest (AWS Lambda + EventBridge + SES)

A once-a-day serverless job that runs the sentiment pipeline and emails you a brief.

```
EventBridge Scheduler (cron)  ──▶  Lambda (container image)  ──▶  SES  ──▶  your inbox
                                   runs marketsentiment.scripts.daily_digest
```

It runs on the **LLM backend** (`MS_SENTIMENT_BACKEND=llm`) so there's no torch/FinBERT to
package — the image stays small and the daily OpenAI cost is a few cents.

## Prerequisites

- AWS account + [AWS CLI](https://aws.amazon.com/cli/) configured (`aws configure`)
- [AWS SAM CLI](https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/install-sam-cli.html)
- Docker running (SAM builds the Lambda container image locally)
- An OpenAI API key

## 1. Verify your email in SES

SES starts in **sandbox** mode, where both sender and recipient must be verified. For a
personal digest that's fine — verify one address and use it for both `From` and `To`:

```bash
aws ses verify-email-identity --email-address you@example.com --region us-east-1
# then click the confirmation link SES emails you
```

Use the **same region** for SES and the Lambda (the template deploys to your CLI's default
region; pass `--region` to override).

## 2. Build & deploy

From the repo root:

```bash
sam build                      # builds the Dockerfile.lambda image
sam deploy --guided            # first time: prompts for the parameters below, saves them
```

Parameters it asks for:

| Parameter | Value |
|-----------|-------|
| `OpenAIApiKey` | your `sk-...` key (stored encrypted; `NoEcho`) |
| `DigestFrom` | the SES-verified address |
| `DigestTo` | where the digest goes (same verified address while in sandbox) |
| `ScheduleExpression` | default `cron(0 9 ? * MON-FRI *)` — 9am weekdays |
| `ScheduleTimezone` | default `America/New_York` |

Subsequent deploys are just `sam build && sam deploy`.

## 3. Test it without waiting for 9am

```bash
sam remote invoke DigestFunction --stack-name <your-stack-name>
# or in the Lambda console: Test → send an empty event
```

You should get the email within a minute. Check `sam logs -n DigestFunction --stack-name <name> --tail` if not.

## Cost

Effectively free: one Lambda invocation/day (well within the free tier) + ~a few hundred
`gpt-4o-mini` calls (~$0.02/day) + SES ($0.10 per 1,000 emails). Call it a few cents a month.

## Optional: cross-day "mention velocity"

The pipeline can report how fast a ticker's mentions are moving vs. the previous run, but
that needs state to survive between (ephemeral) runs. To enable it, persist the DuckDB file
in S3:

1. Create a bucket, e.g. `s3://my-marketsentiment/db.duckdb`.
2. Add `MS_DB_S3_URI: s3://my-marketsentiment/db.duckdb` to the function's `Environment.Variables` in `template.yaml`.
3. Grant the function S3 access — add to `Policies`:
   ```yaml
   - S3CrudPolicy:
       BucketName: my-marketsentiment
   ```

The digest downloads the DB at start and uploads it at the end; the first run just starts empty.

## Notes / limits

- **15-min Lambda cap** — the job classifies each ingested post with one LLM call, so it's
  bounded by post volume (~200 posts → a few minutes). If you widen ingestion, batch the
  calls or switch that step to async.
- **StockTwits rate limits** may thin the pull; add `MS_STOCKTWITS_ACCESS_TOKEN` as another
  env var if you have a token.
- **Leaving sandbox** — to email addresses you haven't verified, request SES production access.
