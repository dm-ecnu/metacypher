.PHONY: help install smoke neo4j-up neo4j-down analyze retrieve generate correct

# Default target
help:
	@echo "MetaCypher artifact targets:"
	@echo "  install    pip install -r requirements.txt"
	@echo "  smoke      run offline smoke test (no Neo4j required)"
	@echo "  neo4j-up   docker compose up -d (start Neo4j)"
	@echo "  neo4j-down docker compose down (stop Neo4j)"
	@echo "  analyze    run query_analyze.py   (pass extra args via ARGS=...)"
	@echo "  retrieve   run all_subgraph_set.py (pass extra args via ARGS=...)"
	@echo "  generate   run generation.py       (pass extra args via ARGS=...)"
	@echo "  correct    run correction.py       (pass extra args via ARGS=...)"

install:
	pip install -r requirements.txt

smoke:
	cd metacypher && python ../examples/smoke_offline.py

neo4j-up:
	docker compose -f docker-compose.yml up -d

neo4j-down:
	docker compose -f docker-compose.yml down

analyze:
	cd metacypher && python query_analyze.py $(ARGS)

retrieve:
	cd metacypher && python all_subgraph_set.py $(ARGS)

generate:
	cd metacypher && python generation.py $(ARGS)

correct:
	cd metacypher && python correction.py $(ARGS)
