.PHONY: setup install models index run test clean

# Full setup
setup: install models
	@echo "Setup complete. Place PDFs in knowledge_base/ and run: make index"

# Install Python deps
install:
	pip install -r requirements.txt

# Pull Ollama models
models:
	ollama pull llava:7b
	ollama pull mistral:7b

# Index all PDFs into vector stores
index:
	python indexer.py

# Index specific domain
index-ceramics:
	python indexer.py ceramics

index-paintings:
	python indexer.py paintings

index-architecture:
	python indexer.py architecture

# Run the Gradio app
run:
	python app.py

# Run setup verification
test:
	python test_setup.py

# Clean generated outputs (keeps knowledge base and vector stores)
clean-outputs:
	rm -f outputs/metadata/*.json
	rm -f outputs/annotations/*.json
	rm -f outputs/annotations/*.csv
	rm -f outputs/images/*.jpg
	rm -f outputs/images/*.png

# Clean vector stores (will need re-indexing)
clean-vectors:
	find vector_stores -name "chroma*" -exec rm -rf {} + 2>/dev/null || true
	find vector_stores -name "*.bin" -delete 2>/dev/null || true

# Full clean
clean: clean-outputs clean-vectors
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# Docker
docker-build:
	docker-compose build

docker-up:
	docker-compose up -d

docker-index:
	docker-compose exec app python indexer.py

docker-down:
	docker-compose down
