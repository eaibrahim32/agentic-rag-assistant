# Observability Stack

## Prometheus and Grafana

The kube-prometheus-stack Helm chart bundles Prometheus, Grafana, Alertmanager,
and node-exporter with sensible default dashboards and alert rules. Prometheus
scrapes metrics on a pull model; targets expose a /metrics endpoint.

## ELK via the ECK operator

Elastic Cloud on Kubernetes (ECK) is an operator that manages Elasticsearch and
Kibana as custom resources. Filebeat runs as a DaemonSet, tails container logs
from each node, and ships them to Elasticsearch for querying in Kibana.

## The distinction

Metrics tell you that something is wrong. Logs tell you what. Traces tell you
where. A stack with only metrics leaves you alerting on a symptom with no path
to the cause.
