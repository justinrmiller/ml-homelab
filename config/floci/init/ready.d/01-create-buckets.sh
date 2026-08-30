#!/bin/sh
# Floci runs the scripts in this directory once the S3 API is live. The compat
# image pre-sets credentials and the local endpoint, so no --endpoint-url flag
# is needed. Hooks are fail-fast: a non-zero exit skips the rest of the phase,
# so every command here has to stay safe to re-run against persisted state.
set -eu

for bucket in app-bucket ray-bucket; do
  aws s3api head-bucket --bucket "$bucket" >/dev/null 2>&1 || aws s3 mb "s3://$bucket"
done

# Parity with the `mc anonymous set public app-bucket` this replaced. Nothing in
# the repo relies on anonymous reads — the dashboard uses presigned URLs — but
# it keeps app-bucket behaving the way it always has.
aws s3api put-bucket-policy --bucket app-bucket --policy '{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": "*",
    "Action": "s3:GetObject",
    "Resource": "arn:aws:s3:::app-bucket/*"
  }]
}'

echo "Buckets ready"
