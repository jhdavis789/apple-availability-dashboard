# Apple dashboard repository

Read `../AGENTS.md` for the project context and data contracts. This is an independent repository, so parent discovery may stop here. Retrieve shared rules with `brain_context` for this absolute repository path and expand relevant nodes.

`FEATURES.md` describes the frontend. `build_data.py` generates `data.json`; `cron_update.sh` collects and deploys; `staleness_alarm.sh` monitors freshness. Use `bash -n` for shell syntax checks. Do not execute collection, deployment, or notification scripts as harmless smoke tests.

Apple pickup state is three-valued: `available`, `unavailable`, and `ineligible`. An all-`ineligible` SKU is retired or invalid, not 0% available. Remove it from active tracking and end its historical line at the last valid observation; never publish `ineligible` as zero. Source: Jackson correction, 2026-09-07.
