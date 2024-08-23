# Guide for E2E Tests

This directory contains end-to-end (E2E) tests for various endpoints of the application. The tests are organized into different subdirectories corresponding to the different types of endpoints: `container`, `dedicated_endpoints`, and `serverless_endpoints`.

## Directory Structure

- `container/`: Contains E2E tests for containerized endpoints.
- `dedicated_endpoints/`: Contains E2E tests for dedicated endpoints.
- `serverless_endpoints/`: Contains E2E tests for serverless endpoints.

Each subdirectory may contain the following files:

- `conftest.py`: Provides fixture definitions for the tests.
- `test_chat_completions.py`: Tests related to chat completions.
- `test_completions.py`: Tests related to other types of completions.

## Running Tests

You can run the tests using `pytest`. Below are the commands to run the tests for each type of endpoint and to run all tests together.

### Running Container Tests

> [!IMPORTANT]
> You should run Friendli Container in your local to run container E2E tests.

```sh
export CONTAINER_HTTP_BASE_URL=http://localhost:8000
export CONTAINER_GRPC_BASE_URL=0.0.0.0:8001
pytest -v -s tests/e2e/container
```

### Running Dedicated Endpoints Tests

> [!IMPORTANT]
> You should create an endpoint in advance to run E2E tests.

```sh
export ENDPOINT_ID=XXXXX
export TEAM_ID=XXXXX
export FRIENDLI_TOKEN=XXXXX
pytest -v -s tests/e2e/dedicated_endpoints
```

### Running Serverless Endpoints Tests

```sh
export MODEL_ID=XXXXX
export TEAM_ID=XXXXX
export FRIENDLI_TOKEN=XXXXX
pytest -v -s tests/e2e/serverless_endpoints
```

### Running All Tests

```sh
export CONTAINER_HTTP_BASE_URL=http://localhost:8000
export CONTAINER_GRPC_BASE_URL=0.0.0.0:8001
export ENDPOINT_ID=XXXXX
export MODEL_ID=XXXXX
export TEAM_ID=XXXXX
export FRIENDLI_TOKEN=XXXXX
pytest -v -s tests/e2e
```

## Environment Variables

The tests rely on several environment variables for configuration:

- `CONTAINER_HTTP_BASE_URL`: Base URL for HTTP requests to the container.
- `CONTAINER_GRPC_BASE_URL`: Base URL for gRPC requests to the container.
- `ENDPOINT_ID`: The ID of the dedicated endpoint being tested.
- `MODEL_ID`: The ID of the model used for serverless endpoints.
- `TEAM_ID`: The team identifier.
- `FRIENDLI_TOKEN`: Authentication token for API access.

**Make sure to replace the placeholders (`XXXXX`) with appropriate values before running the tests.**
