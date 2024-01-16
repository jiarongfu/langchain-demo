from langsmith import Client
from langchain.chains.base import Chain    
from uuid import uuid4
from langchain.smith import RunEvalConfig, run_on_dataset

def run_test(chain:Chain):
    evaluation_config = RunEvalConfig(
        evaluators=[
            "qa",
            "context_qa",
            "cot_qa",
        ]
    )

    client = Client()
    uuid = uuid4()
    run_on_dataset(
        dataset_name="ds-demo-kv-1",
        llm_or_chain_factory=chain,
        client=client,
        evaluation=evaluation_config,
        project_name=f"test-demo-{uuid.hex[0:8]}",
    )