from temporalio import workflow
from datetime import timedelta

@workflow.defn
class DeepONetWorkflow:
    @workflow.run
    async def run(self, epochs: int = 50):
        return await workflow.execute_activity(
            "train_deeponet",
            epochs,
            start_to_close_timeout=timedelta(hours=6),
        )
