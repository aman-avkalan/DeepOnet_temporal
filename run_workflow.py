import asyncio
from temporalio.client import Client

async def main():
    client = await Client.connect("localhost:7233")
    result = await client.execute_workflow(
        "DeepONetWorkflow",
        50,
        id="ldc-deeponet-training",
        task_queue="deeponet-queue",
    )
    print("Workflow finished")
    print("Saved plot:", result)

if __name__ == "__main__":
    asyncio.run(main())
