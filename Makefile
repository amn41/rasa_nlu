.PHONY: clean test lint init docs format formatter build-docker build-docker-full build-docker-mitie-en build-docker-spacy-en build-docker-spacy-de

JOBS ?= 1
INTEGRATION_TEST_FOLDER = tests/integration_tests/
INTEGRATION_TEST_PYTEST_MARKERS ?= "sequential or broker or ((not sequential) and (not broker))"
PLATFORM ?= "linux/amd64"

help:
	@echo "make"
	@echo "    clean"
	@echo "        Remove Python/build artifacts."
	@echo "    install"
	@echo "        Install rasa."
	@echo "    install-full"
	@echo "        Install rasa with all extras (transformers, tensorflow_text, spacy, jieba)."
	@echo "    formatter"
	@echo "        Apply black formatting to code."
	@echo "    lint"
	@echo "        Lint code with ruff, and check if black formatter should be applied."
	@echo "    lint-docstrings"
	@echo "        Check docstring conventions in changed files."
	@echo "    types"
	@echo "        Check for type errors using mypy."
	@echo "    static-checks"
	@echo "        Run all python static checks."
	@echo "    prepare-tests-ubuntu"
	@echo "        Install system requirements for running tests on Ubuntu and Debian based systems."
	@echo "    prepare-tests-macos"
	@echo "        Install system requirements for running tests on macOS."
	@echo "    prepare-tests-windows"
	@echo "        Install system requirements for running tests on Windows."
	@echo "    prepare-spacy"
	@echo "        Download models needed for spacy tests."
	@echo "    prepare-mitie"
	@echo "        Download the standard english mitie model."
	@echo "    prepare-transformers"
	@echo "        Download all models needed for testing LanguageModelFeaturizer."
	@echo "    test"
	@echo "        Run pytest on tests/."
	@echo "        Use the JOBS environment variable to configure number of workers (default: 1)."
	@echo "    test-integration"
	@echo "        Run integration tests using pytest."
	@echo "        Use the JOBS environment variable to configure number of workers (default: 1)."
	@echo "    livedocs"
	@echo "        Build the docs locally."
	@echo "    release"
	@echo "        Prepare a release."
	@echo "    build-docker"
	@echo "        Build Rasa Open Source Docker image."
	@echo "    run-integration-containers"
	@echo "        Run the integration test containers."
	@echo "    stop-integration-containers"
	@echo "        Stop the integration test containers."

clean:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f  {} +
	rm -rf build/
	rm -rf .mypy_cache/
	rm -rf dist/
	rm -rf docs/build
	rm -rf docs/.docusaurus

install:
	poetry run python -m pip install -U pip
	poetry install

install-mitie:
	poetry run python -m pip install -U pip
	poetry run python -m pip install -U git+https://github.com/tmbo/MITIE.git#egg=mitie

install-full: install-mitie
	poetry install -E full

install-docs:
	cd docs/ && yarn install

formatter:
	poetry run black rasa tests

format: formatter

lint:
     # Ignore docstring errors when running on the entire project
	poetry run ruff check rasa tests --ignore D
	poetry run black --check rasa tests
	make lint-docstrings

# Compare against `main` if no branch was provided
BRANCH ?= main
lint-docstrings:
	./scripts/lint_python_docstrings.sh $(BRANCH)

lint-changelog:
	./scripts/lint_changelog_files.sh

lint-security:
	poetry run bandit -ll -ii -r --config pyproject.toml rasa/*

types:
	poetry run mypy rasa

static-checks: lint lint-security types

prepare-spacy:
	poetry run python -m spacy download en_core_web_md
	poetry run python -m spacy download de_core_news_sm

prepare-mitie:
	wget --progress=dot:giga -N -P data/ https://github.com/mit-nlp/MITIE/releases/download/v0.4/MITIE-models-v0.2.tar.bz2
ifeq ($(OS),Windows_NT)
	7z x data/MITIE-models-v0.2.tar.bz2 -bb3
	7z x MITIE-models-v0.2.tar -bb3
	cp MITIE-models/english/total_word_feature_extractor.dat data/
	rm -r MITIE-models
	rm MITIE-models-v0.2.tar
else
	tar -xvjf data/MITIE-models-v0.2.tar.bz2 --strip-components 2 -C data/ MITIE-models/english/total_word_feature_extractor.dat
endif
	rm data/MITIE*.bz2

prepare-transformers:
	while read -r MODEL; do poetry run python scripts/download_transformer_model.py $$MODEL ; done < data/test/hf_transformers_models.txt
	if ! [ $(CI) ]; then poetry run python scripts/download_transformer_model.py rasa/LaBSE; fi
prepare-tests-macos:
	brew install wget graphviz || true

prepare-tests-ubuntu:
	sudo apt-get -y install graphviz graphviz-dev python-tk

prepare-tests-windows:
	choco install wget graphviz

# GitHub Action has pre-installed a helper function for installing Chocolatey packages
# It will retry the installation 5 times if it fails
# See: https://github.com/actions/virtual-environments/blob/main/images/win/scripts/ImageHelpers/ChocoHelpers.ps1
prepare-tests-windows-gha:
	powershell -command "Install-ChocoPackage wget graphviz"

test: clean
	# OMP_NUM_THREADS can improve overall performance using one thread by process (on tensorflow), avoiding overload
	# TF_CPP_MIN_LOG_LEVEL=2 sets C code log level for tensorflow to error suppressing lower log events
	OMP_NUM_THREADS=1 TF_CPP_MIN_LOG_LEVEL=2 poetry run pytest tests -n $(JOBS) --dist loadscope --cov rasa --ignore $(INTEGRATION_TEST_FOLDER)

test-integration:
	# OMP_NUM_THREADS can improve overall performance using one thread by process (on tensorflow), avoiding overload
	# TF_CPP_MIN_LOG_LEVEL=2 sets C code log level for tensorflow to error suppressing lower log events
ifeq (,$(wildcard tests_deployment/.env))
	OMP_NUM_THREADS=1 TF_CPP_MIN_LOG_LEVEL=2 poetry run pytest $(INTEGRATION_TEST_FOLDER) -n $(JOBS) -m $(INTEGRATION_TEST_PYTEST_MARKERS) --dist loadgroup
else
	set -o allexport; source tests_deployment/.env && OMP_NUM_THREADS=1 TF_CPP_MIN_LOG_LEVEL=2 poetry run pytest $(INTEGRATION_TEST_FOLDER) -n $(JOBS) -m $(INTEGRATION_TEST_PYTEST_MARKERS) --dist loadgroup && set +o allexport
endif

test-cli: PYTEST_MARKER=category_cli and (not flaky)
test-cli: DD_ARGS := $(or $(DD_ARGS),)
test-cli: test-marker

test-core-featurizers: PYTEST_MARKER=category_core_featurizers and (not flaky)
test-core-featurizers: DD_ARGS := $(or $(DD_ARGS),)
test-core-featurizers: test-marker

test-policies: PYTEST_MARKER=category_policies and (not flaky)
test-policies: DD_ARGS := $(or $(DD_ARGS),)
test-policies: test-marker

test-nlu-featurizers: PYTEST_MARKER=category_nlu_featurizers and (not flaky)
test-nlu-featurizers: DD_ARGS := $(or $(DD_ARGS),)
test-nlu-featurizers: prepare-spacy prepare-mitie prepare-transformers test-marker

test-nlu-predictors: PYTEST_MARKER=category_nlu_predictors and (not flaky)
test-nlu-predictors: DD_ARGS := $(or $(DD_ARGS),)
test-nlu-predictors: prepare-spacy prepare-mitie test-marker

test-full-model-training: PYTEST_MARKER=category_full_model_training and (not flaky)
test-full-model-training: DD_ARGS := $(or $(DD_ARGS),)
test-full-model-training: prepare-spacy prepare-mitie prepare-transformers test-marker

test-other-unit-tests: PYTEST_MARKER=category_other_unit_tests and (not flaky)
test-other-unit-tests: DD_ARGS := $(or $(DD_ARGS),)
test-other-unit-tests: prepare-spacy prepare-mitie test-marker

test-performance: PYTEST_MARKER=category_performance and (not flaky)
test-performance: DD_ARGS := $(or $(DD_ARGS),)
test-performance: test-marker

test-flaky: PYTEST_MARKER=flaky
test-flaky: DD_ARGS := $(or $(DD_ARGS),)
test-flaky: prepare-spacy prepare-mitie test-marker

test-gh-actions:
	OMP_NUM_THREADS=1 TF_CPP_MIN_LOG_LEVEL=2 poetry run pytest .github/tests --cov .github/scripts

test-marker: clean
    # OMP_NUM_THREADS can improve overall performance using one thread by process (on tensorflow), avoiding overload
	# TF_CPP_MIN_LOG_LEVEL=2 sets C code log level for tensorflow to error suppressing lower log events
	TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 TF_CPP_MIN_LOG_LEVEL=2 poetry run pytest tests -n $(JOBS) --dist loadscope -m "$(PYTEST_MARKER)" --cov rasa --ignore $(INTEGRATION_TEST_FOLDER) $(DD_ARGS)

generate-pending-changelog:
	poetry run python -c "from scripts import release; release.generate_changelog('major.minor.patch')"

cleanup-generated-changelog:
	# this is a helper to cleanup your git status locally after running "make test-docs"
	# it's not run on CI at the moment
	git status --porcelain | sed -n '/^D */s///p' | xargs git reset HEAD
	git reset HEAD CHANGELOG.mdx
	git ls-files --deleted | xargs git checkout
	git checkout CHANGELOG.mdx

test-docs: generate-pending-changelog docs
	poetry run pytest tests/docs/*

lint-docs: generate-pending-changelog docs
	cd docs/ && yarn mdx-lint

prepare-docs:
	cd docs/ && poetry run yarn pre-build

docs: prepare-docs
	cd docs/ && yarn build

livedocs:
	cd docs/ && poetry run yarn start

preview-docs:
	cd docs/ && yarn build && yarn deploy-preview --alias=${PULL_REQUEST_NUMBER} --message="Preview for Pull Request #${PULL_REQUEST_NUMBER}"

publish-docs:
	cd docs/ && yarn build && yarn deploy

release:
	poetry run python scripts/release.py

build-docker:
	export IMAGE_NAME=rasa && \
	docker buildx use default && \
	docker buildx bake --set default.platform=${PLATFORM} -f docker/docker-bake.hcl base && \
	docker buildx bake --set default.platform=${PLATFORM} -f docker/docker-bake.hcl base-poetry && \
	docker buildx bake --set default.platform=${PLATFORM} -f docker/docker-bake.hcl base-builder && \
	docker buildx bake --set default.platform=${PLATFORM} -f docker/docker-bake.hcl default

build-docker-full:
	export IMAGE_NAME=rasa && \
	docker buildx use default && \
	docker buildx bake --set default.platform=${PLATFORM} -f docker/docker-bake.hcl base-images && \
	docker buildx bake --set default.platform=${PLATFORM} -f docker/docker-bake.hcl base-builder && \
	docker buildx bake --set default.platform=${PLATFORM} -f docker/docker-bake.hcl full

build-docker-mitie-en:
	export IMAGE_NAME=rasa && \
	docker buildx use default && \
	docker buildx bake --set default.platform=${PLATFORM} -f docker/docker-bake.hcl base-images && \
	docker buildx bake --set default.platform=${PLATFORM} -f docker/docker-bake.hcl base-builder && \
	docker buildx bake --set default.platform=${PLATFORM} -f docker/docker-bake.hcl mitie-en

build-docker-spacy-en:
	export IMAGE_NAME=rasa && \
	docker buildx use default && \
	docker buildx bake --set default.platform=${PLATFORM} -f docker/docker-bake.hcl base && \
	docker buildx bake --set default.platform=${PLATFORM} -f docker/docker-bake.hcl base-poetry && \
	docker buildx bake --set default.platform=${PLATFORM} -f docker/docker-bake.hcl base-builder && \
	docker buildx bake --set default.platform=${PLATFORM} -f docker/docker-bake.hcl spacy-en

build-docker-spacy-de:
	export IMAGE_NAME=rasa && \
	docker buildx use default && \
	docker buildx bake --set default.platform=${PLATFORM} -f docker/docker-bake.hcl base && \
	docker buildx bake --set default.platform=${PLATFORM} -f docker/docker-bake.hcl base-poetry && \
	docker buildx bake --set default.platform=${PLATFORM} -f docker/docker-bake.hcl base-builder && \
	docker buildx bake --set default.platform=${PLATFORM} -f docker/docker-bake.hcl spacy-de

build-docker-spacy-it:
	export IMAGE_NAME=rasa && \
	docker buildx use default && \
	docker buildx bake --set default.platform=${PLATFORM} -f docker/docker-bake.hcl base && \
	docker buildx bake --set default.platform=${PLATFORM} -f docker/docker-bake.hcl base-poetry && \
	docker buildx bake --set default.platform=${PLATFORM} -f docker/docker-bake.hcl base-builder && \
	docker buildx bake --set default.platform=${PLATFORM} -f docker/docker-bake.hcl spacy-it

build-tests-deployment-env: ## Create environment files (.env) for docker-compose.
	cd tests_deployment && \
	test -f .env || cat .env.example >> .env

run-integration-containers: build-tests-deployment-env ## Run the integration test containers.
	cd tests_deployment && \
	docker-compose -f docker-compose.integration.yml up &

stop-integration-containers: ## Stop the integration test containers.
	cd tests_deployment && \
	docker-compose -f docker-compose.integration.yml down
