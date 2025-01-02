#!/bin/bash

set -Eeuo pipefail

if [[ ${GITHUB_TAG} =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    curl -X POST -H "Content-type: application/json" \
	 --data "{\"text\":\"💥 New *Rasa Open Source* version ${GITHUB_TAG} has been released! https://github.com/RasaHQ/rasa/releases/tag/${GITHUB_TAG}\"}" \
	 "https://hooks.slack.com/services/T0GHWFTS8/BMTQQL47K/${SLACK_WEBHOOK_TOKEN}"
fi
