# Kubernetes Autoscaling

## Horizontal Pod Autoscaler

The Horizontal Pod Autoscaler (HPA) automatically adjusts the number of pod
replicas in a Deployment based on observed metrics, most commonly CPU
utilization. The HPA controller queries the metrics server on a fixed interval
(15 seconds by default) and compares current utilization against the target.

Scaling up is fast; scaling down is deliberately slow. The default
stabilization window for scale-down is 300 seconds, which prevents thrashing
when load is spiky.

## Load testing observations

Under a sustained synthetic load driving CPU to 231% of the configured request,
a deployment with `minReplicas: 1` and `maxReplicas: 5` scaled from 1 pod to 5
pods within roughly 90 seconds, then returned to 1 pod approximately 5 minutes
after load stopped.

## Requests and limits

HPA calculates utilization as a percentage of the CPU *request*, not the limit.
A deployment with no CPU request will not autoscale on CPU at all.
