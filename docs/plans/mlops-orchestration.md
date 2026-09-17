# Plan: MLOps Orchestration Roadmap — Index

Master index for the MLOps orchestration roadmap. This project is a **template
for practicing production-level MLOps** (see `AGENTS.md`) — the actual data
volume doesn't require any of this, but we adopt these tools deliberately to
practice real production patterns.

Split into two companion docs, in the order they'll be implemented:

1. **`kubernetes-plan.md`** — detailed, ready-to-implement plan for Phase 1
   (CD: build & push images to GHCR) and Phase 2 (Kubernetes: local `kind`
   cluster running this project's jobs + stateful services). Includes the
   concrete "real-world Kubernetes use cases" reference and the K8s design
   rationale.
2. **`airflow-terraform-overview.md`** — general concepts and use cases for
   Airflow (orchestrating the K8s jobs with real dependencies/branching) and
   Terraform (provisioning real cloud infrastructure once we move off local
   kind). Deliberately kept high-level; details get filled in once
   Kubernetes (and then Airflow) are actually implemented.

## Overall phase order

1. **CD → GHCR** (`kubernetes-plan.md` Phase 1)
2. **Kubernetes on kind** (`kubernetes-plan.md` Phase 2)
3. **Airflow** orchestrating the K8s jobs (`airflow-terraform-overview.md` Part 1)
4. **Terraform** provisioning cloud infra (`airflow-terraform-overview.md` Part 2)

## Cross-cutting risks / considerations (apply across phases)
- Heavy CUDA base images make CD builds slow/large — rely on buildx GHA cache + path filters.
- Baking training code into its image changes the current bind-mount dev workflow — keep both paths working.
- kind has no GPU — training/inference-on-kind is a manifest-shape exercise; real GPU runs wait for the cloud phase or stay on host compose.
- In-repo postgres manifest is authored fresh (no existing compose to mirror) — must match the credentials/schema the app expects.
- Private GHCR packages require an imagePullSecret in every namespace that pulls them.
- Secrets must never be committed — generate K8s Secrets from gitignored env files locally; document the cloud Secret Manager path.

## Progress log
- [ ] `kubernetes-plan.md` Phase 1: CD → GHCR
- [ ] `kubernetes-plan.md` Phase 2: Kubernetes
- [ ] `airflow-terraform-overview.md` Part 1: Airflow
- [ ] `airflow-terraform-overview.md` Part 2: Terraform
