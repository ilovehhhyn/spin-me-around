#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "Usage: $0 <subscription-id> <region> <quota-family-fragment> <required-vcpus>" >&2
  exit 2
fi

subscription_id="$1"
region="$2"
quota_family_fragment="$3"
required_vcpus="$4"

az account set --subscription "$subscription_id"

# Required human-readable check from the provisioning protocol.
az vm list-usage --location "$region" --output table

usage_json="$(az vm list-usage --location "$region" --output json)"
matching_usage="$(
  jq \
    --arg family "$quota_family_fragment" \
    '[.[] | select(((.localName // .name.localizedValue // .name.value) | ascii_downcase) | contains($family | ascii_downcase))]' \
    <<<"$usage_json"
)"

match_count="$(jq 'length' <<<"$matching_usage")"
if [[ "$match_count" -ne 1 ]]; then
  echo "Expected one quota family matching '$quota_family_fragment'; found $match_count." >&2
  echo "Choose the exact family name from the table above and rerun." >&2
  exit 3
fi

quota_limit="$(jq -r '.[0].limit' <<<"$matching_usage")"
quota_used="$(jq -r '.[0].currentValue' <<<"$matching_usage")"
quota_available=$((quota_limit - quota_used))

echo "Matched quota record:"
jq '.[0]' <<<"$matching_usage"

if [[ "$quota_limit" -eq 0 ]]; then
  echo "STOP: target GPU-family quota limit is 0 in $region. File a quota request." >&2
  exit 4
fi

if [[ "$quota_available" -lt "$required_vcpus" ]]; then
  echo "STOP: $quota_available vCPUs remain, but the target VM requires $required_vcpus." >&2
  exit 5
fi

echo "Quota gate passed: $quota_available family vCPUs are currently available."
