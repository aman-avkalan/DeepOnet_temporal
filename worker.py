import asyncio
from concurrent.futures import ThreadPoolExecutor
from temporalio.client import Client
from temporalio.worker import Worker

from workflows import DeepONetWorkflow
from activities import train_deeponet

async def main():
    client = await Client.connect("localhost:7233")

    worker = Worker(
        client,
        task_queue="deeponet-queue",
        workflows=[DeepONetWorkflow],
        activities=[train_deeponet],
        activity_executor=ThreadPoolExecutor(max_workers=1),
    )

    print("DeepONet Temporal worker started")
    await worker.run()

if __name__ == "__main__":
    asyncio.run(main())
